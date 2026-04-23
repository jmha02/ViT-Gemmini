#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "onnxruntime_c_api.h"
#include "ivit_gemmini_ops.h"

#ifdef IVIT_USE_GEMMINI
#define SYSTOLIC_INT8
#include "systolic_include.h"
#endif

namespace {

int g_ivit_execution_mode = 0;

inline int positive_mod(int value, int mod) {
    return (value % mod + mod) % mod;
}

#ifdef IVIT_USE_GEMMINI
inline tiled_matmul_type_t get_accelerator_mode(int mode) {
    return static_cast<tiled_matmul_type_t>(positive_mod(mode - 1, static_cast<int>(CPU) + 1));
}
#endif

inline int clamp_mode(int mode) {
    if (mode < 0 || mode > 2) {
        return 0;
    }
    return mode;
}

static inline void ort_profile_cycle_barrier(void) {
    asm volatile ("" ::: "memory");
}

static inline uint64_t ort_profile_read_cycles(void) {
#if defined(__riscv)
    uint64_t cycles = 0;
    ort_profile_cycle_barrier();
    asm volatile("rdcycle %0" : "=r"(cycles) : : "memory");
    ort_profile_cycle_barrier();
    return cycles;
#else
    return 0;
#endif
}

int64_t get_optional_int64_attribute(const OrtApi* api, const OrtKernelInfo* info, const char* name, int64_t default_value) {
    int64_t value = default_value;
    OrtStatus* status = api->KernelInfoGetAttribute_int64(info, name, &value);
    if (status != nullptr) {
        api->ReleaseStatus(status);
        return default_value;
    }
    return value;
}

std::string get_optional_string_attribute(const OrtApi* api, const OrtKernelInfo* info, const char* name) {
    size_t size = 0;
    OrtStatus* status = api->KernelInfoGetAttribute_string(info, name, nullptr, &size);
    if (status != nullptr) {
        api->ReleaseStatus(status);
        return "";
    }
    std::string value(size, '\0');
    status = api->KernelInfoGetAttribute_string(info, name, value.data(), &size);
    if (status != nullptr) {
        api->ReleaseStatus(status);
        return "";
    }
    if (!value.empty() && value.back() == '\0') {
        value.pop_back();
    }
    return value;
}

void ort_log_node_cycles(int64_t node_index, const std::string& node_name, const char* op_type, uint64_t cycles) {
    if (node_index < 0 || node_name.empty()) {
        return;
    }
    std::printf(
        "[ORT_NODE_CYCLES],%lld,%s,%s,CPUExecutionProvider,%llu\n",
        static_cast<long long>(node_index),
        node_name.c_str(),
        op_type,
        static_cast<unsigned long long>(cycles)
    );
}

void cpu_matmul_int32(const int8_t* a, const int8_t* b, int32_t* y,
        int64_t m, int64_t n, int64_t k) {
    for (int64_t row = 0; row < m; ++row) {
        for (int64_t col = 0; col < n; ++col) {
            int32_t acc = 0;
            for (int64_t depth = 0; depth < k; ++depth) {
                acc += static_cast<int32_t>(a[row * k + depth]) *
                       static_cast<int32_t>(b[depth * n + col]);
            }
            y[row * n + col] = acc;
        }
    }
}

#ifdef IVIT_USE_GEMMINI
void gemmini_matmul_int32(const int8_t* a, const int8_t* b, int32_t* y,
        int64_t m, int64_t n, int64_t k, int mode) {
    if (m == 0 || n == 0 || k == 0) {
        return;
    }

    tiled_matmul_auto(
        static_cast<size_t>(m),
        static_cast<size_t>(n),
        static_cast<size_t>(k),
        a,
        b,
        nullptr,
        y,
        static_cast<size_t>(k),
        static_cast<size_t>(n),
        static_cast<size_t>(n),
        static_cast<size_t>(n),
        MVIN_SCALE_IDENTITY,
        MVIN_SCALE_IDENTITY,
        ACC_SCALE_IDENTITY,
        NO_ACTIVATION,
        ACC_SCALE_IDENTITY,
        0,
        false,
        false,
        false,
        true,
        false,
        3,
        get_accelerator_mode(mode));
}
#endif

std::vector<int64_t> get_dims(const OrtApi* ort, const OrtValue* value) {
    OrtTensorTypeAndShapeInfo* info = nullptr;
    ort->GetTensorTypeAndShape(value, &info);
    size_t ndim = 0;
    ort->GetDimensionsCount(info, &ndim);
    std::vector<int64_t> dims(ndim, 0);
    ort->GetDimensions(info, dims.data(), ndim);
    ort->ReleaseTensorTypeAndShapeInfo(info);
    return dims;
}

int64_t numel(const std::vector<int64_t>& dims) {
    int64_t total = 1;
    for (int64_t dim : dims) {
        total *= dim;
    }
    return total;
}

template <typename T>
const T* get_tensor_data(const OrtApi* ort, const OrtValue* value) {
    void* raw = nullptr;
    ort->GetTensorMutableData(const_cast<OrtValue*>(value), &raw);
    return static_cast<const T*>(raw);
}

template <typename T>
T* get_mutable_tensor_data(const OrtApi* ort, OrtValue* value) {
    void* raw = nullptr;
    ort->GetTensorMutableData(value, &raw);
    return static_cast<T*>(raw);
}

int8_t require_scalar_zero_point(const OrtApi* ort, const OrtValue* value, const char* input_name) {
    const auto dims = get_dims(ort, value);
    int64_t size = 1;
    for (int64_t dim : dims) {
        size *= dim;
    }
    if (size != 1) {
        throw std::runtime_error(std::string(input_name) + " zero-point must be a scalar tensor");
    }

    return get_tensor_data<int8_t>(ort, value)[0];
}

template <typename T>
T require_scalar_tensor_value(const OrtApi* ort, const OrtValue* value, const char* input_name) {
    const auto dims = get_dims(ort, value);
    int64_t size = 1;
    for (int64_t dim : dims) {
        size *= dim;
    }
    if (size != 1) {
        throw std::runtime_error(std::string(input_name) + " must be a scalar tensor");
    }
    return get_tensor_data<T>(ort, value)[0];
}

template <typename OutT>
OutT clamp_to_type(long long value) {
    const long long lo = static_cast<long long>(std::numeric_limits<OutT>::min());
    const long long hi = static_cast<long long>(std::numeric_limits<OutT>::max());
    if (value < lo) {
        value = lo;
    }
    if (value > hi) {
        value = hi;
    }
    return static_cast<OutT>(value);
}

std::vector<int64_t> compute_matmul_output_shape(
        const std::vector<int64_t>& a_dims, const std::vector<int64_t>& b_dims) {
    if (a_dims.size() < 2) {
        throw std::runtime_error("GemminiMatMulInteger expects A to have rank >= 2");
    }

    if (b_dims.size() == 2) {
        if (a_dims.back() != b_dims.front()) {
            throw std::runtime_error("GemminiMatMulInteger shape mismatch");
        }

        auto out_dims = a_dims;
        out_dims.back() = b_dims.back();
        return out_dims;
    }

    if (b_dims.size() != a_dims.size()) {
        throw std::runtime_error("GemminiMatMulInteger expects B to have rank 2 or match A rank");
    }
    if (a_dims.back() != b_dims[b_dims.size() - 2]) {
        throw std::runtime_error("GemminiMatMulInteger shape mismatch");
    }
    for (size_t i = 0; i + 2 < a_dims.size(); ++i) {
        if (a_dims[i] != b_dims[i]) {
            throw std::runtime_error("GemminiMatMulInteger batch dimensions must match");
        }
    }

    auto out_dims = a_dims;
    out_dims.back() = b_dims.back();
    return out_dims;
}

bool scale_is_scalar(const OrtApi* ort, const OrtValue* scale_value) {
    const auto dims = get_dims(ort, scale_value);
    const size_t ndim = dims.size();
    if (ndim == 0) {
        return true;
    }
    if (ndim == 1 && dims[0] == 1) {
        return true;
    }
    return false;
}

inline float safe_scale(float scale) {
    return std::max(scale, 1e-8f);
}

float repq_sym_scale_from_uniform_params(
        const float* scales,
        bool scalar_scale,
        const float* zero_points,
        bool scalar_zero_point,
        int64_t cols,
        int32_t n_bits) {
    const float levels = static_cast<float>((1 << n_bits) - 1);
    float max_abs = 0.0f;
    for (int64_t col = 0; col < cols; ++col) {
        const float scale = scalar_scale ? scales[0] : scales[col];
        const float zp = scalar_zero_point ? zero_points[0] : zero_points[col];
        const float qmin = -zp;
        const float qmax = levels - zp;
        max_abs = std::max(max_abs, std::abs(qmin * scale));
        max_abs = std::max(max_abs, std::abs(qmax * scale));
    }
    return safe_scale(max_abs / 127.0f);
}

inline float repq_sym_scale_from_log_delta(float delta) {
    return safe_scale(std::abs(delta) / 127.0f);
}

inline int8_t quantize_sym_int8_scalar(float value, float scale) {
    const long long q = std::llround(static_cast<double>(value) / static_cast<double>(safe_scale(scale)));
    return clamp_to_type<int8_t>(std::clamp(q, -127LL, 127LL));
}

struct FixedPointScale {
    int64_t mantissa;
    int32_t exponent;
};

FixedPointScale decompose_fixedpoint_scale(float scale) {
    if (scale == 0.0f) {
        return {0, 0};
    }

    int exp = 0;
    const double mant = std::frexp(static_cast<double>(scale), &exp);
    const int64_t mantissa = std::llround(std::ldexp(mant, 31));
    return {mantissa, static_cast<int32_t>(31 - exp)};
}

template <typename InT, typename OutT>
void requantize_tensor_fixedpoint_same_shape(const OrtApi* ort, OrtKernelContext* ctx) {
    const OrtValue* x_value = nullptr;
    const OrtValue* scale_value = nullptr;
    ort->KernelContext_GetInput(ctx, 0, &x_value);
    ort->KernelContext_GetInput(ctx, 1, &scale_value);

    const auto dims = get_dims(ort, x_value);
    if (dims.empty()) {
        throw std::runtime_error("Requantize op expects rank >= 1");
    }

    int64_t outer = 1;
    for (size_t i = 0; i + 1 < dims.size(); ++i) {
        outer *= dims[i];
    }
    const int64_t last = dims.back();

    const InT* x = get_tensor_data<InT>(ort, x_value);
    const float* scales = get_tensor_data<float>(ort, scale_value);
    const bool scalar_scale = scale_is_scalar(ort, scale_value);

    std::vector<FixedPointScale> fixed_scales(static_cast<size_t>(scalar_scale ? 1 : last));
    for (int64_t col = 0; col < static_cast<int64_t>(fixed_scales.size()); ++col) {
        fixed_scales[static_cast<size_t>(col)] = decompose_fixedpoint_scale(scales[scalar_scale ? 0 : col]);
    }

    OrtValue* out_value = nullptr;
    ort->KernelContext_GetOutput(ctx, 0, dims.data(), dims.size(), &out_value);
    OutT* out = get_mutable_tensor_data<OutT>(ort, out_value);

    for (int64_t row = 0; row < outer; ++row) {
        for (int64_t col = 0; col < last; ++col) {
            const FixedPointScale fp = fixed_scales[static_cast<size_t>(scalar_scale ? 0 : col)];
            if (fp.mantissa == 0) {
                out[row * last + col] = static_cast<OutT>(0);
                continue;
            }

            const double scaled = static_cast<double>(x[row * last + col]) * static_cast<double>(fp.mantissa);
            const double divided = std::ldexp(scaled, -fp.exponent);
            const long long rounded = std::llrint(divided);
            out[row * last + col] = clamp_to_type<OutT>(rounded);
        }
    }
}

struct GemminiMatMulIntegerKernel {
    const OrtApi* ort;
    int64_t profile_node_index;
    std::string profile_label;
};

void* ORT_API_CALL GemminiMatMulInteger_CreateKernel(
        const OrtCustomOp* op, const OrtApi* api, const OrtKernelInfo* info) {
    (void)op;
    auto* kernel = new GemminiMatMulIntegerKernel();
    kernel->ort = api;
    kernel->profile_node_index = get_optional_int64_attribute(api, info, "profile_node_index", -1);
    kernel->profile_label = get_optional_string_attribute(api, info, "profile_label");
    return kernel;
}

void ORT_API_CALL GemminiMatMulInteger_Destroy(void* op_kernel) {
    delete static_cast<GemminiMatMulIntegerKernel*>(op_kernel);
}

void ORT_API_CALL GemminiMatMulInteger_Compute(void* op_kernel, OrtKernelContext* ctx) {
    auto* kernel = static_cast<GemminiMatMulIntegerKernel*>(op_kernel);
    const OrtApi* ort = kernel->ort;
    const uint64_t profile_start = ort_profile_read_cycles();
    const auto finish_profile = [&]() {
        ort_log_node_cycles(
            kernel->profile_node_index,
            kernel->profile_label,
            "GemminiMatMulInteger",
            ort_profile_read_cycles() - profile_start
        );
    };

    const OrtValue* a_value = nullptr;
    const OrtValue* b_value = nullptr;
    const OrtValue* a_zp_value = nullptr;
    const OrtValue* b_zp_value = nullptr;
    ort->KernelContext_GetInput(ctx, 0, &a_value);
    ort->KernelContext_GetInput(ctx, 1, &b_value);
    ort->KernelContext_GetInput(ctx, 2, &a_zp_value);
    ort->KernelContext_GetInput(ctx, 3, &b_zp_value);

    if (require_scalar_zero_point(ort, a_zp_value, "A") != 0 ||
            require_scalar_zero_point(ort, b_zp_value, "B") != 0) {
        throw std::runtime_error("GemminiMatMulInteger only supports zero-points of 0");
    }

    const auto a_dims = get_dims(ort, a_value);
    const auto b_dims = get_dims(ort, b_value);
    const auto out_dims = compute_matmul_output_shape(a_dims, b_dims);

    const int8_t* a = get_tensor_data<int8_t>(ort, a_value);
    const int8_t* b = get_tensor_data<int8_t>(ort, b_value);

    OrtValue* out_value = nullptr;
    ort->KernelContext_GetOutput(ctx, 0, out_dims.data(), out_dims.size(), &out_value);
    int32_t* out = get_mutable_tensor_data<int32_t>(ort, out_value);

    const int64_t rows = a_dims[a_dims.size() - 2];
    const int64_t cols = b_dims.back();
    const int64_t depth = a_dims.back();
    const int64_t batch = numel(a_dims) / (rows * depth);
    const int64_t a_stride = rows * depth;
    const int64_t b_stride = (b_dims.size() == 2) ? 0 : (depth * cols);
    const int64_t out_stride = rows * cols;
    std::memset(out, 0, static_cast<size_t>(batch * out_stride) * sizeof(int32_t));

    const int mode = clamp_mode(g_ivit_execution_mode);
#ifdef IVIT_USE_GEMMINI
    for (int64_t batch_idx = 0; batch_idx < batch; ++batch_idx) {
        const int8_t* a_ptr = a + batch_idx * a_stride;
        const int8_t* b_ptr = (b_dims.size() == 2) ? b : (b + batch_idx * b_stride);
        int32_t* out_ptr = out + batch_idx * out_stride;
        if (mode == 0) {
            cpu_matmul_int32(a_ptr, b_ptr, out_ptr, rows, cols, depth);
        } else {
            gemmini_matmul_int32(a_ptr, b_ptr, out_ptr, rows, cols, depth, mode);
        }
    }
#else
    (void)mode;
    for (int64_t batch_idx = 0; batch_idx < batch; ++batch_idx) {
        const int8_t* a_ptr = a + batch_idx * a_stride;
        const int8_t* b_ptr = (b_dims.size() == 2) ? b : (b + batch_idx * b_stride);
        int32_t* out_ptr = out + batch_idx * out_stride;
        cpu_matmul_int32(a_ptr, b_ptr, out_ptr, rows, cols, depth);
    }
#endif
    finish_profile();
}

const char* ORT_API_CALL GemminiMatMulInteger_GetName(const OrtCustomOp* op) {
    (void)op;
    return "GemminiMatMulInteger";
}

const char* ORT_API_CALL GemminiMatMulInteger_GetExecutionProviderType(const OrtCustomOp* op) {
    (void)op;
    return "CPUExecutionProvider";
}

ONNXTensorElementDataType ORT_API_CALL GemminiMatMulInteger_GetInputType(
        const OrtCustomOp* op, size_t idx) {
    (void)op;
    (void)idx;
    return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8;
}

size_t ORT_API_CALL GemminiMatMulInteger_GetInputTypeCount(const OrtCustomOp* op) {
    (void)op;
    return 4;
}

ONNXTensorElementDataType ORT_API_CALL GemminiMatMulInteger_GetOutputType(
        const OrtCustomOp* op, size_t idx) {
    (void)op;
    (void)idx;
    return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32;
}

size_t ORT_API_CALL GemminiMatMulInteger_GetOutputTypeCount(const OrtCustomOp* op) {
    (void)op;
    return 1;
}

OrtCustomOpInputOutputCharacteristic ORT_API_CALL GemminiMatMulInteger_GetInputCharacteristic(
        const OrtCustomOp* op, size_t idx) {
    (void)op;
    (void)idx;
    return INPUT_OUTPUT_REQUIRED;
}

OrtCustomOpInputOutputCharacteristic ORT_API_CALL GemminiMatMulInteger_GetOutputCharacteristic(
        const OrtCustomOp* op, size_t idx) {
    (void)op;
    (void)idx;
    return INPUT_OUTPUT_REQUIRED;
}

OrtCustomOp make_gemmini_matmul_integer_op() {
    OrtCustomOp op{};
    op.version = ORT_API_VERSION;
    op.CreateKernel = GemminiMatMulInteger_CreateKernel;
    op.GetName = GemminiMatMulInteger_GetName;
    op.GetExecutionProviderType = GemminiMatMulInteger_GetExecutionProviderType;
    op.GetInputType = GemminiMatMulInteger_GetInputType;
    op.GetInputTypeCount = GemminiMatMulInteger_GetInputTypeCount;
    op.GetOutputType = GemminiMatMulInteger_GetOutputType;
    op.GetOutputTypeCount = GemminiMatMulInteger_GetOutputTypeCount;
    op.KernelCompute = GemminiMatMulInteger_Compute;
    op.KernelDestroy = GemminiMatMulInteger_Destroy;
    op.GetInputCharacteristic = GemminiMatMulInteger_GetInputCharacteristic;
    op.GetOutputCharacteristic = GemminiMatMulInteger_GetOutputCharacteristic;
    return op;
}

struct RequantizeInt32Kernel {
    const OrtApi* ort;
};

void* ORT_API_CALL RequantizeInt32_CreateKernel(
        const OrtCustomOp* op, const OrtApi* api, const OrtKernelInfo* info) {
    (void)op;
    (void)info;
    auto* kernel = new RequantizeInt32Kernel();
    kernel->ort = api;
    return kernel;
}

void ORT_API_CALL RequantizeInt32_Destroy(void* op_kernel) {
    delete static_cast<RequantizeInt32Kernel*>(op_kernel);
}

void ORT_API_CALL RequantizeInt32_Compute(void* op_kernel, OrtKernelContext* ctx) {
    auto* kernel = static_cast<RequantizeInt32Kernel*>(op_kernel);
    const OrtApi* ort = kernel->ort;

    const OrtValue* x_value = nullptr;
    const OrtValue* scale_value = nullptr;
    ort->KernelContext_GetInput(ctx, 0, &x_value);
    ort->KernelContext_GetInput(ctx, 1, &scale_value);

    OrtTensorTypeAndShapeInfo* x_info = nullptr;
    ort->GetTensorTypeAndShape(x_value, &x_info);
    size_t ndim = 0;
    ort->GetDimensionsCount(x_info, &ndim);
    std::vector<int64_t> dims(ndim, 0);
    ort->GetDimensions(x_info, dims.data(), ndim);
    ort->ReleaseTensorTypeAndShapeInfo(x_info);

    if (ndim == 0) {
        throw std::runtime_error("RequantizeInt32 expects rank >= 1");
    }

    int64_t outer = 1;
    for (size_t i = 0; i + 1 < ndim; ++i) {
        outer *= dims[i];
    }
    const int64_t last = dims[ndim - 1];

    const int32_t* x = get_tensor_data<int32_t>(ort, x_value);
    const float* scales = get_tensor_data<float>(ort, scale_value);
    const bool scalar_scale = scale_is_scalar(ort, scale_value);

    OrtValue* out_value = nullptr;
    ort->KernelContext_GetOutput(ctx, 0, dims.data(), ndim, &out_value);
    int8_t* out = get_mutable_tensor_data<int8_t>(ort, out_value);

    for (int64_t row = 0; row < outer; ++row) {
        for (int64_t col = 0; col < last; ++col) {
            const float scale = scalar_scale ? scales[0] : scales[col];
            const float scaled = static_cast<float>(x[row * last + col]) * scale;
            long rounded = lrintf(scaled);
            rounded = std::clamp(rounded, -128L, 127L);
            out[row * last + col] = static_cast<int8_t>(rounded);
        }
    }
}

template <typename InT, typename OutT>
void requantize_tensor_same_shape(const OrtApi* ort, OrtKernelContext* ctx) {
    const OrtValue* x_value = nullptr;
    const OrtValue* scale_value = nullptr;
    ort->KernelContext_GetInput(ctx, 0, &x_value);
    ort->KernelContext_GetInput(ctx, 1, &scale_value);

    const auto dims = get_dims(ort, x_value);
    if (dims.empty()) {
        throw std::runtime_error("Requantize op expects rank >= 1");
    }

    int64_t outer = 1;
    for (size_t i = 0; i + 1 < dims.size(); ++i) {
        outer *= dims[i];
    }
    const int64_t last = dims.back();

    const InT* x = get_tensor_data<InT>(ort, x_value);
    const float* scales = get_tensor_data<float>(ort, scale_value);
    const bool scalar_scale = scale_is_scalar(ort, scale_value);

    OrtValue* out_value = nullptr;
    ort->KernelContext_GetOutput(ctx, 0, dims.data(), dims.size(), &out_value);
    OutT* out = get_mutable_tensor_data<OutT>(ort, out_value);

    for (int64_t row = 0; row < outer; ++row) {
        for (int64_t col = 0; col < last; ++col) {
            const float scale = scalar_scale ? scales[0] : scales[col];
            const float scaled = static_cast<float>(x[row * last + col]) * scale;
            const long rounded = lrintf(scaled);
            out[row * last + col] = clamp_to_type<OutT>(rounded);
        }
    }
}

struct RequantizeInt64Kernel {
    const OrtApi* ort;
};

void* ORT_API_CALL RequantizeInt64_CreateKernel(
        const OrtCustomOp* op, const OrtApi* api, const OrtKernelInfo* info) {
    (void)op;
    (void)info;
    auto* kernel = new RequantizeInt64Kernel();
    kernel->ort = api;
    return kernel;
}

void ORT_API_CALL RequantizeInt64_Destroy(void* op_kernel) {
    delete static_cast<RequantizeInt64Kernel*>(op_kernel);
}

void ORT_API_CALL RequantizeInt64_Compute(void* op_kernel, OrtKernelContext* ctx) {
    auto* kernel = static_cast<RequantizeInt64Kernel*>(op_kernel);
    requantize_tensor_fixedpoint_same_shape<int64_t, int8_t>(kernel->ort, ctx);
}

const char* ORT_API_CALL RequantizeInt64_GetName(const OrtCustomOp* op) {
    (void)op;
    return "RequantizeInt64";
}

const char* ORT_API_CALL RequantizeInt64_GetExecutionProviderType(const OrtCustomOp* op) {
    (void)op;
    return "CPUExecutionProvider";
}

ONNXTensorElementDataType ORT_API_CALL RequantizeInt64_GetInputType(
        const OrtCustomOp* op, size_t idx) {
    (void)op;
    return idx == 0 ? ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64
                    : ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
}

size_t ORT_API_CALL RequantizeInt64_GetInputTypeCount(const OrtCustomOp* op) {
    (void)op;
    return 2;
}

ONNXTensorElementDataType ORT_API_CALL RequantizeInt64_GetOutputType(
        const OrtCustomOp* op, size_t idx) {
    (void)op;
    (void)idx;
    return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8;
}

size_t ORT_API_CALL RequantizeInt64_GetOutputTypeCount(const OrtCustomOp* op) {
    (void)op;
    return 1;
}

OrtCustomOpInputOutputCharacteristic ORT_API_CALL RequantizeInt64_GetInputCharacteristic(
        const OrtCustomOp* op, size_t idx) {
    (void)op;
    (void)idx;
    return INPUT_OUTPUT_REQUIRED;
}

OrtCustomOpInputOutputCharacteristic ORT_API_CALL RequantizeInt64_GetOutputCharacteristic(
        const OrtCustomOp* op, size_t idx) {
    (void)op;
    (void)idx;
    return INPUT_OUTPUT_REQUIRED;
}

OrtCustomOp make_requantize_int64_op() {
    OrtCustomOp op{};
    op.version = ORT_API_VERSION;
    op.CreateKernel = RequantizeInt64_CreateKernel;
    op.GetName = RequantizeInt64_GetName;
    op.GetExecutionProviderType = RequantizeInt64_GetExecutionProviderType;
    op.GetInputType = RequantizeInt64_GetInputType;
    op.GetInputTypeCount = RequantizeInt64_GetInputTypeCount;
    op.GetOutputType = RequantizeInt64_GetOutputType;
    op.GetOutputTypeCount = RequantizeInt64_GetOutputTypeCount;
    op.KernelCompute = RequantizeInt64_Compute;
    op.KernelDestroy = RequantizeInt64_Destroy;
    op.GetInputCharacteristic = RequantizeInt64_GetInputCharacteristic;
    op.GetOutputCharacteristic = RequantizeInt64_GetOutputCharacteristic;
    return op;
}

struct RequantizeInt64ToInt16Kernel {
    const OrtApi* ort;
};

void* ORT_API_CALL RequantizeInt64ToInt16_CreateKernel(
        const OrtCustomOp* op, const OrtApi* api, const OrtKernelInfo* info) {
    (void)op;
    (void)info;
    auto* kernel = new RequantizeInt64ToInt16Kernel();
    kernel->ort = api;
    return kernel;
}

void ORT_API_CALL RequantizeInt64ToInt16_Destroy(void* op_kernel) {
    delete static_cast<RequantizeInt64ToInt16Kernel*>(op_kernel);
}

void ORT_API_CALL RequantizeInt64ToInt16_Compute(void* op_kernel, OrtKernelContext* ctx) {
    auto* kernel = static_cast<RequantizeInt64ToInt16Kernel*>(op_kernel);
    requantize_tensor_fixedpoint_same_shape<int64_t, int16_t>(kernel->ort, ctx);
}

const char* ORT_API_CALL RequantizeInt64ToInt16_GetName(const OrtCustomOp* op) {
    (void)op;
    return "RequantizeInt64ToInt16";
}

const char* ORT_API_CALL RequantizeInt64ToInt16_GetExecutionProviderType(const OrtCustomOp* op) {
    (void)op;
    return "CPUExecutionProvider";
}

ONNXTensorElementDataType ORT_API_CALL RequantizeInt64ToInt16_GetInputType(
        const OrtCustomOp* op, size_t idx) {
    (void)op;
    return idx == 0 ? ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64
                    : ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
}

size_t ORT_API_CALL RequantizeInt64ToInt16_GetInputTypeCount(const OrtCustomOp* op) {
    (void)op;
    return 2;
}

ONNXTensorElementDataType ORT_API_CALL RequantizeInt64ToInt16_GetOutputType(
        const OrtCustomOp* op, size_t idx) {
    (void)op;
    (void)idx;
    return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16;
}

size_t ORT_API_CALL RequantizeInt64ToInt16_GetOutputTypeCount(const OrtCustomOp* op) {
    (void)op;
    return 1;
}

OrtCustomOpInputOutputCharacteristic ORT_API_CALL RequantizeInt64ToInt16_GetInputCharacteristic(
        const OrtCustomOp* op, size_t idx) {
    (void)op;
    (void)idx;
    return INPUT_OUTPUT_REQUIRED;
}

OrtCustomOpInputOutputCharacteristic ORT_API_CALL RequantizeInt64ToInt16_GetOutputCharacteristic(
        const OrtCustomOp* op, size_t idx) {
    (void)op;
    (void)idx;
    return INPUT_OUTPUT_REQUIRED;
}

OrtCustomOp make_requantize_int64_to_int16_op() {
    OrtCustomOp op{};
    op.version = ORT_API_VERSION;
    op.CreateKernel = RequantizeInt64ToInt16_CreateKernel;
    op.GetName = RequantizeInt64ToInt16_GetName;
    op.GetExecutionProviderType = RequantizeInt64ToInt16_GetExecutionProviderType;
    op.GetInputType = RequantizeInt64ToInt16_GetInputType;
    op.GetInputTypeCount = RequantizeInt64ToInt16_GetInputTypeCount;
    op.GetOutputType = RequantizeInt64ToInt16_GetOutputType;
    op.GetOutputTypeCount = RequantizeInt64ToInt16_GetOutputTypeCount;
    op.KernelCompute = RequantizeInt64ToInt16_Compute;
    op.KernelDestroy = RequantizeInt64ToInt16_Destroy;
    op.GetInputCharacteristic = RequantizeInt64ToInt16_GetInputCharacteristic;
    op.GetOutputCharacteristic = RequantizeInt64ToInt16_GetOutputCharacteristic;
    return op;
}

struct RequantizeInt32ToInt16Kernel {
    const OrtApi* ort;
};

void* ORT_API_CALL RequantizeInt32ToInt16_CreateKernel(
        const OrtCustomOp* op, const OrtApi* api, const OrtKernelInfo* info) {
    (void)op;
    (void)info;
    auto* kernel = new RequantizeInt32ToInt16Kernel();
    kernel->ort = api;
    return kernel;
}

void ORT_API_CALL RequantizeInt32ToInt16_Destroy(void* op_kernel) {
    delete static_cast<RequantizeInt32ToInt16Kernel*>(op_kernel);
}

void ORT_API_CALL RequantizeInt32ToInt16_Compute(void* op_kernel, OrtKernelContext* ctx) {
    auto* kernel = static_cast<RequantizeInt32ToInt16Kernel*>(op_kernel);
    requantize_tensor_same_shape<int32_t, int16_t>(kernel->ort, ctx);
}

const char* ORT_API_CALL RequantizeInt32ToInt16_GetName(const OrtCustomOp* op) {
    (void)op;
    return "RequantizeInt32ToInt16";
}

const char* ORT_API_CALL RequantizeInt32ToInt16_GetExecutionProviderType(const OrtCustomOp* op) {
    (void)op;
    return "CPUExecutionProvider";
}

ONNXTensorElementDataType ORT_API_CALL RequantizeInt32ToInt16_GetInputType(
        const OrtCustomOp* op, size_t idx) {
    (void)op;
    return idx == 0 ? ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32
                    : ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
}

size_t ORT_API_CALL RequantizeInt32ToInt16_GetInputTypeCount(const OrtCustomOp* op) {
    (void)op;
    return 2;
}

ONNXTensorElementDataType ORT_API_CALL RequantizeInt32ToInt16_GetOutputType(
        const OrtCustomOp* op, size_t idx) {
    (void)op;
    (void)idx;
    return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16;
}

size_t ORT_API_CALL RequantizeInt32ToInt16_GetOutputTypeCount(const OrtCustomOp* op) {
    (void)op;
    return 1;
}

OrtCustomOpInputOutputCharacteristic ORT_API_CALL RequantizeInt32ToInt16_GetInputCharacteristic(
        const OrtCustomOp* op, size_t idx) {
    (void)op;
    (void)idx;
    return INPUT_OUTPUT_REQUIRED;
}

OrtCustomOpInputOutputCharacteristic ORT_API_CALL RequantizeInt32ToInt16_GetOutputCharacteristic(
        const OrtCustomOp* op, size_t idx) {
    (void)op;
    (void)idx;
    return INPUT_OUTPUT_REQUIRED;
}

OrtCustomOp make_requantize_int32_to_int16_op() {
    OrtCustomOp op{};
    op.version = ORT_API_VERSION;
    op.CreateKernel = RequantizeInt32ToInt16_CreateKernel;
    op.GetName = RequantizeInt32ToInt16_GetName;
    op.GetExecutionProviderType = RequantizeInt32ToInt16_GetExecutionProviderType;
    op.GetInputType = RequantizeInt32ToInt16_GetInputType;
    op.GetInputTypeCount = RequantizeInt32ToInt16_GetInputTypeCount;
    op.GetOutputType = RequantizeInt32ToInt16_GetOutputType;
    op.GetOutputTypeCount = RequantizeInt32ToInt16_GetOutputTypeCount;
    op.KernelCompute = RequantizeInt32ToInt16_Compute;
    op.KernelDestroy = RequantizeInt32ToInt16_Destroy;
    op.GetInputCharacteristic = RequantizeInt32ToInt16_GetInputCharacteristic;
    op.GetOutputCharacteristic = RequantizeInt32ToInt16_GetOutputCharacteristic;
    return op;
}

struct RequantizeInt8ToInt16Kernel {
    const OrtApi* ort;
};

void* ORT_API_CALL RequantizeInt8ToInt16_CreateKernel(
        const OrtCustomOp* op, const OrtApi* api, const OrtKernelInfo* info) {
    (void)op;
    (void)info;
    auto* kernel = new RequantizeInt8ToInt16Kernel();
    kernel->ort = api;
    return kernel;
}

void ORT_API_CALL RequantizeInt8ToInt16_Destroy(void* op_kernel) {
    delete static_cast<RequantizeInt8ToInt16Kernel*>(op_kernel);
}

void ORT_API_CALL RequantizeInt8ToInt16_Compute(void* op_kernel, OrtKernelContext* ctx) {
    auto* kernel = static_cast<RequantizeInt8ToInt16Kernel*>(op_kernel);
    requantize_tensor_same_shape<int8_t, int16_t>(kernel->ort, ctx);
}

const char* ORT_API_CALL RequantizeInt8ToInt16_GetName(const OrtCustomOp* op) {
    (void)op;
    return "RequantizeInt8ToInt16";
}

const char* ORT_API_CALL RequantizeInt8ToInt16_GetExecutionProviderType(const OrtCustomOp* op) {
    (void)op;
    return "CPUExecutionProvider";
}

ONNXTensorElementDataType ORT_API_CALL RequantizeInt8ToInt16_GetInputType(
        const OrtCustomOp* op, size_t idx) {
    (void)op;
    return idx == 0 ? ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8
                    : ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
}

size_t ORT_API_CALL RequantizeInt8ToInt16_GetInputTypeCount(const OrtCustomOp* op) {
    (void)op;
    return 2;
}

ONNXTensorElementDataType ORT_API_CALL RequantizeInt8ToInt16_GetOutputType(
        const OrtCustomOp* op, size_t idx) {
    (void)op;
    (void)idx;
    return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16;
}

size_t ORT_API_CALL RequantizeInt8ToInt16_GetOutputTypeCount(const OrtCustomOp* op) {
    (void)op;
    return 1;
}

OrtCustomOpInputOutputCharacteristic ORT_API_CALL RequantizeInt8ToInt16_GetInputCharacteristic(
        const OrtCustomOp* op, size_t idx) {
    (void)op;
    (void)idx;
    return INPUT_OUTPUT_REQUIRED;
}

OrtCustomOpInputOutputCharacteristic ORT_API_CALL RequantizeInt8ToInt16_GetOutputCharacteristic(
        const OrtCustomOp* op, size_t idx) {
    (void)op;
    (void)idx;
    return INPUT_OUTPUT_REQUIRED;
}

OrtCustomOp make_requantize_int8_to_int16_op() {
    OrtCustomOp op{};
    op.version = ORT_API_VERSION;
    op.CreateKernel = RequantizeInt8ToInt16_CreateKernel;
    op.GetName = RequantizeInt8ToInt16_GetName;
    op.GetExecutionProviderType = RequantizeInt8ToInt16_GetExecutionProviderType;
    op.GetInputType = RequantizeInt8ToInt16_GetInputType;
    op.GetInputTypeCount = RequantizeInt8ToInt16_GetInputTypeCount;
    op.GetOutputType = RequantizeInt8ToInt16_GetOutputType;
    op.GetOutputTypeCount = RequantizeInt8ToInt16_GetOutputTypeCount;
    op.KernelCompute = RequantizeInt8ToInt16_Compute;
    op.KernelDestroy = RequantizeInt8ToInt16_Destroy;
    op.GetInputCharacteristic = RequantizeInt8ToInt16_GetInputCharacteristic;
    op.GetOutputCharacteristic = RequantizeInt8ToInt16_GetOutputCharacteristic;
    return op;
}

struct RequantizeInt16Kernel {
    const OrtApi* ort;
};

void* ORT_API_CALL RequantizeInt16_CreateKernel(
        const OrtCustomOp* op, const OrtApi* api, const OrtKernelInfo* info) {
    (void)op;
    (void)info;
    auto* kernel = new RequantizeInt16Kernel();
    kernel->ort = api;
    return kernel;
}

void ORT_API_CALL RequantizeInt16_Destroy(void* op_kernel) {
    delete static_cast<RequantizeInt16Kernel*>(op_kernel);
}

void ORT_API_CALL RequantizeInt16_Compute(void* op_kernel, OrtKernelContext* ctx) {
    auto* kernel = static_cast<RequantizeInt16Kernel*>(op_kernel);
    requantize_tensor_same_shape<int16_t, int16_t>(kernel->ort, ctx);
}

const char* ORT_API_CALL RequantizeInt16_GetName(const OrtCustomOp* op) {
    (void)op;
    return "RequantizeInt16";
}

const char* ORT_API_CALL RequantizeInt16_GetExecutionProviderType(const OrtCustomOp* op) {
    (void)op;
    return "CPUExecutionProvider";
}

ONNXTensorElementDataType ORT_API_CALL RequantizeInt16_GetInputType(
        const OrtCustomOp* op, size_t idx) {
    (void)op;
    return idx == 0 ? ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16
                    : ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
}

size_t ORT_API_CALL RequantizeInt16_GetInputTypeCount(const OrtCustomOp* op) {
    (void)op;
    return 2;
}

ONNXTensorElementDataType ORT_API_CALL RequantizeInt16_GetOutputType(
        const OrtCustomOp* op, size_t idx) {
    (void)op;
    (void)idx;
    return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16;
}

size_t ORT_API_CALL RequantizeInt16_GetOutputTypeCount(const OrtCustomOp* op) {
    (void)op;
    return 1;
}

OrtCustomOpInputOutputCharacteristic ORT_API_CALL RequantizeInt16_GetInputCharacteristic(
        const OrtCustomOp* op, size_t idx) {
    (void)op;
    (void)idx;
    return INPUT_OUTPUT_REQUIRED;
}

OrtCustomOpInputOutputCharacteristic ORT_API_CALL RequantizeInt16_GetOutputCharacteristic(
        const OrtCustomOp* op, size_t idx) {
    (void)op;
    (void)idx;
    return INPUT_OUTPUT_REQUIRED;
}

OrtCustomOp make_requantize_int16_op() {
    OrtCustomOp op{};
    op.version = ORT_API_VERSION;
    op.CreateKernel = RequantizeInt16_CreateKernel;
    op.GetName = RequantizeInt16_GetName;
    op.GetExecutionProviderType = RequantizeInt16_GetExecutionProviderType;
    op.GetInputType = RequantizeInt16_GetInputType;
    op.GetInputTypeCount = RequantizeInt16_GetInputTypeCount;
    op.GetOutputType = RequantizeInt16_GetOutputType;
    op.GetOutputTypeCount = RequantizeInt16_GetOutputTypeCount;
    op.KernelCompute = RequantizeInt16_Compute;
    op.KernelDestroy = RequantizeInt16_Destroy;
    op.GetInputCharacteristic = RequantizeInt16_GetInputCharacteristic;
    op.GetOutputCharacteristic = RequantizeInt16_GetOutputCharacteristic;
    return op;
}

struct RequantizeInt16ToInt8Kernel {
    const OrtApi* ort;
};

void* ORT_API_CALL RequantizeInt16ToInt8_CreateKernel(
        const OrtCustomOp* op, const OrtApi* api, const OrtKernelInfo* info) {
    (void)op;
    (void)info;
    auto* kernel = new RequantizeInt16ToInt8Kernel();
    kernel->ort = api;
    return kernel;
}

void ORT_API_CALL RequantizeInt16ToInt8_Destroy(void* op_kernel) {
    delete static_cast<RequantizeInt16ToInt8Kernel*>(op_kernel);
}

void ORT_API_CALL RequantizeInt16ToInt8_Compute(void* op_kernel, OrtKernelContext* ctx) {
    auto* kernel = static_cast<RequantizeInt16ToInt8Kernel*>(op_kernel);
    requantize_tensor_same_shape<int16_t, int8_t>(kernel->ort, ctx);
}

const char* ORT_API_CALL RequantizeInt16ToInt8_GetName(const OrtCustomOp* op) {
    (void)op;
    return "RequantizeInt16ToInt8";
}

const char* ORT_API_CALL RequantizeInt16ToInt8_GetExecutionProviderType(const OrtCustomOp* op) {
    (void)op;
    return "CPUExecutionProvider";
}

ONNXTensorElementDataType ORT_API_CALL RequantizeInt16ToInt8_GetInputType(
        const OrtCustomOp* op, size_t idx) {
    (void)op;
    return idx == 0 ? ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16
                    : ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
}

size_t ORT_API_CALL RequantizeInt16ToInt8_GetInputTypeCount(const OrtCustomOp* op) {
    (void)op;
    return 2;
}

ONNXTensorElementDataType ORT_API_CALL RequantizeInt16ToInt8_GetOutputType(
        const OrtCustomOp* op, size_t idx) {
    (void)op;
    (void)idx;
    return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8;
}

size_t ORT_API_CALL RequantizeInt16ToInt8_GetOutputTypeCount(const OrtCustomOp* op) {
    (void)op;
    return 1;
}

OrtCustomOpInputOutputCharacteristic ORT_API_CALL RequantizeInt16ToInt8_GetInputCharacteristic(
        const OrtCustomOp* op, size_t idx) {
    (void)op;
    (void)idx;
    return INPUT_OUTPUT_REQUIRED;
}

OrtCustomOpInputOutputCharacteristic ORT_API_CALL RequantizeInt16ToInt8_GetOutputCharacteristic(
        const OrtCustomOp* op, size_t idx) {
    (void)op;
    (void)idx;
    return INPUT_OUTPUT_REQUIRED;
}

OrtCustomOp make_requantize_int16_to_int8_op() {
    OrtCustomOp op{};
    op.version = ORT_API_VERSION;
    op.CreateKernel = RequantizeInt16ToInt8_CreateKernel;
    op.GetName = RequantizeInt16ToInt8_GetName;
    op.GetExecutionProviderType = RequantizeInt16ToInt8_GetExecutionProviderType;
    op.GetInputType = RequantizeInt16ToInt8_GetInputType;
    op.GetInputTypeCount = RequantizeInt16ToInt8_GetInputTypeCount;
    op.GetOutputType = RequantizeInt16ToInt8_GetOutputType;
    op.GetOutputTypeCount = RequantizeInt16ToInt8_GetOutputTypeCount;
    op.KernelCompute = RequantizeInt16ToInt8_Compute;
    op.KernelDestroy = RequantizeInt16ToInt8_Destroy;
    op.GetInputCharacteristic = RequantizeInt16ToInt8_GetInputCharacteristic;
    op.GetOutputCharacteristic = RequantizeInt16ToInt8_GetOutputCharacteristic;
    return op;
}

struct QLinearAddInt16Kernel {
    const OrtApi* ort;
};

void* ORT_API_CALL QLinearAddInt16_CreateKernel(
        const OrtCustomOp* op, const OrtApi* api, const OrtKernelInfo* info) {
    (void)op;
    (void)info;
    auto* kernel = new QLinearAddInt16Kernel();
    kernel->ort = api;
    return kernel;
}

void ORT_API_CALL QLinearAddInt16_Destroy(void* op_kernel) {
    delete static_cast<QLinearAddInt16Kernel*>(op_kernel);
}

void ORT_API_CALL QLinearAddInt16_Compute(void* op_kernel, OrtKernelContext* ctx) {
    auto* kernel = static_cast<QLinearAddInt16Kernel*>(op_kernel);
    const OrtApi* ort = kernel->ort;

    const OrtValue* x1_value = nullptr;
    const OrtValue* x1_scale_value = nullptr;
    const OrtValue* x1_zp_value = nullptr;
    const OrtValue* x2_value = nullptr;
    const OrtValue* x2_scale_value = nullptr;
    const OrtValue* x2_zp_value = nullptr;
    const OrtValue* y_scale_value = nullptr;
    const OrtValue* y_zp_value = nullptr;
    ort->KernelContext_GetInput(ctx, 0, &x1_value);
    ort->KernelContext_GetInput(ctx, 1, &x1_scale_value);
    ort->KernelContext_GetInput(ctx, 2, &x1_zp_value);
    ort->KernelContext_GetInput(ctx, 3, &x2_value);
    ort->KernelContext_GetInput(ctx, 4, &x2_scale_value);
    ort->KernelContext_GetInput(ctx, 5, &x2_zp_value);
    ort->KernelContext_GetInput(ctx, 6, &y_scale_value);
    ort->KernelContext_GetInput(ctx, 7, &y_zp_value);

    if (require_scalar_tensor_value<int16_t>(ort, x1_zp_value, "x1_zero_point") != 0 ||
            require_scalar_tensor_value<int16_t>(ort, x2_zp_value, "x2_zero_point") != 0 ||
            require_scalar_tensor_value<int16_t>(ort, y_zp_value, "y_zero_point") != 0) {
        throw std::runtime_error("QLinearAddInt16 only supports zero-points of 0");
    }

    const float x1_scale = require_scalar_tensor_value<float>(ort, x1_scale_value, "x1_scale");
    const float x2_scale = require_scalar_tensor_value<float>(ort, x2_scale_value, "x2_scale");
    const float y_scale = require_scalar_tensor_value<float>(ort, y_scale_value, "y_scale");
    if (y_scale == 0.0f) {
        throw std::runtime_error("QLinearAddInt16 requires non-zero y_scale");
    }

    const auto x1_dims = get_dims(ort, x1_value);
    const auto x2_dims = get_dims(ort, x2_value);
    if (x1_dims != x2_dims) {
        throw std::runtime_error("QLinearAddInt16 expects matching input shapes");
    }

    const int16_t* x1 = get_tensor_data<int16_t>(ort, x1_value);
    const int16_t* x2 = get_tensor_data<int16_t>(ort, x2_value);

    OrtValue* out_value = nullptr;
    ort->KernelContext_GetOutput(ctx, 0, x1_dims.data(), x1_dims.size(), &out_value);
    int16_t* out = get_mutable_tensor_data<int16_t>(ort, out_value);

    const float x1_ratio = x1_scale / y_scale;
    const float x2_ratio = x2_scale / y_scale;
    const int64_t total = numel(x1_dims);
    for (int64_t idx = 0; idx < total; ++idx) {
        const long lhs = lrintf(static_cast<float>(x1[idx]) * x1_ratio);
        const long rhs = lrintf(static_cast<float>(x2[idx]) * x2_ratio);
        out[idx] = clamp_to_type<int16_t>(static_cast<long long>(lhs) + static_cast<long long>(rhs));
    }
}

const char* ORT_API_CALL QLinearAddInt16_GetName(const OrtCustomOp* op) {
    (void)op;
    return "QLinearAddInt16";
}

const char* ORT_API_CALL QLinearAddInt16_GetExecutionProviderType(const OrtCustomOp* op) {
    (void)op;
    return "CPUExecutionProvider";
}

ONNXTensorElementDataType ORT_API_CALL QLinearAddInt16_GetInputType(
        const OrtCustomOp* op, size_t idx) {
    (void)op;
    switch (idx) {
        case 0:
        case 3:
            return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16;
        case 1:
        case 4:
        case 6:
            return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
        case 2:
        case 5:
        case 7:
            return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16;
        default:
            return ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
    }
}

size_t ORT_API_CALL QLinearAddInt16_GetInputTypeCount(const OrtCustomOp* op) {
    (void)op;
    return 8;
}

ONNXTensorElementDataType ORT_API_CALL QLinearAddInt16_GetOutputType(
        const OrtCustomOp* op, size_t idx) {
    (void)op;
    (void)idx;
    return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16;
}

size_t ORT_API_CALL QLinearAddInt16_GetOutputTypeCount(const OrtCustomOp* op) {
    (void)op;
    return 1;
}

OrtCustomOpInputOutputCharacteristic ORT_API_CALL QLinearAddInt16_GetInputCharacteristic(
        const OrtCustomOp* op, size_t idx) {
    (void)op;
    (void)idx;
    return INPUT_OUTPUT_REQUIRED;
}

OrtCustomOpInputOutputCharacteristic ORT_API_CALL QLinearAddInt16_GetOutputCharacteristic(
        const OrtCustomOp* op, size_t idx) {
    (void)op;
    (void)idx;
    return INPUT_OUTPUT_REQUIRED;
}

OrtCustomOp make_qlinear_add_int16_op() {
    OrtCustomOp op{};
    op.version = ORT_API_VERSION;
    op.CreateKernel = QLinearAddInt16_CreateKernel;
    op.GetName = QLinearAddInt16_GetName;
    op.GetExecutionProviderType = QLinearAddInt16_GetExecutionProviderType;
    op.GetInputType = QLinearAddInt16_GetInputType;
    op.GetInputTypeCount = QLinearAddInt16_GetInputTypeCount;
    op.GetOutputType = QLinearAddInt16_GetOutputType;
    op.GetOutputTypeCount = QLinearAddInt16_GetOutputTypeCount;
    op.KernelCompute = QLinearAddInt16_Compute;
    op.KernelDestroy = QLinearAddInt16_Destroy;
    op.GetInputCharacteristic = QLinearAddInt16_GetInputCharacteristic;
    op.GetOutputCharacteristic = QLinearAddInt16_GetOutputCharacteristic;
    return op;
}

const char* ORT_API_CALL RequantizeInt32_GetName(const OrtCustomOp* op) {
    (void)op;
    return "RequantizeInt32";
}

const char* ORT_API_CALL RequantizeInt32_GetExecutionProviderType(const OrtCustomOp* op) {
    (void)op;
    return "CPUExecutionProvider";
}

ONNXTensorElementDataType ORT_API_CALL RequantizeInt32_GetInputType(
        const OrtCustomOp* op, size_t idx) {
    (void)op;
    return idx == 0 ? ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32
                    : ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
}

size_t ORT_API_CALL RequantizeInt32_GetInputTypeCount(const OrtCustomOp* op) {
    (void)op;
    return 2;
}

ONNXTensorElementDataType ORT_API_CALL RequantizeInt32_GetOutputType(
        const OrtCustomOp* op, size_t idx) {
    (void)op;
    (void)idx;
    return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8;
}

size_t ORT_API_CALL RequantizeInt32_GetOutputTypeCount(const OrtCustomOp* op) {
    (void)op;
    return 1;
}

OrtCustomOpInputOutputCharacteristic ORT_API_CALL RequantizeInt32_GetInputCharacteristic(
        const OrtCustomOp* op, size_t idx) {
    (void)op;
    (void)idx;
    return INPUT_OUTPUT_REQUIRED;
}

OrtCustomOpInputOutputCharacteristic ORT_API_CALL RequantizeInt32_GetOutputCharacteristic(
        const OrtCustomOp* op, size_t idx) {
    (void)op;
    (void)idx;
    return INPUT_OUTPUT_REQUIRED;
}

OrtCustomOp make_requantize_int32_op() {
    OrtCustomOp op{};
    op.version = ORT_API_VERSION;
    op.CreateKernel = RequantizeInt32_CreateKernel;
    op.GetName = RequantizeInt32_GetName;
    op.GetExecutionProviderType = RequantizeInt32_GetExecutionProviderType;
    op.GetInputType = RequantizeInt32_GetInputType;
    op.GetInputTypeCount = RequantizeInt32_GetInputTypeCount;
    op.GetOutputType = RequantizeInt32_GetOutputType;
    op.GetOutputTypeCount = RequantizeInt32_GetOutputTypeCount;
    op.KernelCompute = RequantizeInt32_Compute;
    op.KernelDestroy = RequantizeInt32_Destroy;
    op.GetInputCharacteristic = RequantizeInt32_GetInputCharacteristic;
    op.GetOutputCharacteristic = RequantizeInt32_GetOutputCharacteristic;
    return op;
}

struct RepQLogQuantKernel {
    const OrtApi* ort;
    int32_t n_bits;
};

void* ORT_API_CALL RepQLogQuant_CreateKernel(
        const OrtCustomOp* op, const OrtApi* api, const OrtKernelInfo* info) {
    (void)op;
    auto* kernel = new RepQLogQuantKernel();
    kernel->ort = api;
    int64_t n_bits = 8;
    api->KernelInfoGetAttribute_int64(info, "n_bits", &n_bits);
    kernel->n_bits = static_cast<int32_t>(n_bits);
    return kernel;
}

void ORT_API_CALL RepQLogQuant_Destroy(void* op_kernel) {
    delete static_cast<RepQLogQuantKernel*>(op_kernel);
}

void ORT_API_CALL RepQLogQuant_Compute(void* op_kernel, OrtKernelContext* ctx) {
    auto* kernel = static_cast<RepQLogQuantKernel*>(op_kernel);
    const OrtApi* ort = kernel->ort;

    const OrtValue* x_value = nullptr;
    const OrtValue* delta_value = nullptr;
    ort->KernelContext_GetInput(ctx, 0, &x_value);
    ort->KernelContext_GetInput(ctx, 1, &delta_value);
    const auto x_dims = get_dims(ort, x_value);
    const auto delta_dims = get_dims(ort, delta_value);
    const int64_t x_elements = numel(x_dims);
    const int64_t delta_elements = numel(delta_dims);
    if (delta_elements != 1) {
        throw std::runtime_error("RepQLogQuant expects scalar delta");
    }

    const float* x = get_tensor_data<float>(ort, x_value);
    const float* delta_ptr = get_tensor_data<float>(ort, delta_value);
    const float delta = delta_ptr[0];
    const float sqrt2_minus_one = std::sqrt(2.0f) - 1.0f;
    const int32_t levels = 1 << kernel->n_bits;

    OrtValue* y_value = nullptr;
    ort->KernelContext_GetOutput(ctx, 0, x_dims.data(), x_dims.size(), &y_value);
    float* y = get_mutable_tensor_data<float>(ort, y_value);

    if (delta <= 0.0f) {
        std::fill(y, y + x_elements, 0.0f);
        return;
    }

    for (int64_t idx = 0; idx < x_elements; ++idx) {
        const float x_val = x[idx];
        if (!(x_val > 0.0f)) {
            y[idx] = 0.0f;
            continue;
        }

        float q_float = std::round(-std::log2(x_val / delta) * 2.0f);
        if (q_float >= static_cast<float>(levels)) {
            y[idx] = 0.0f;
            continue;
        }

        q_float = std::clamp(q_float, 0.0f, static_cast<float>(levels - 1));
        const int32_t q = static_cast<int32_t>(q_float);
        const float odd_mask = (q % 2 == 0) ? 1.0f : (1.0f + sqrt2_minus_one);
        const float exponent = -std::ceil(static_cast<float>(q) / 2.0f);
        y[idx] = std::exp2(exponent) * odd_mask * delta;
    }
}

const char* ORT_API_CALL RepQLogQuant_GetName(const OrtCustomOp* op) {
    (void)op;
    return "RepQLogQuant";
}

const char* ORT_API_CALL RepQLogQuant_GetExecutionProviderType(const OrtCustomOp* op) {
    (void)op;
    return "CPUExecutionProvider";
}

ONNXTensorElementDataType ORT_API_CALL RepQLogQuant_GetInputType(
        const OrtCustomOp* op, size_t idx) {
    (void)op;
    if (idx > 1) {
        throw std::runtime_error("RepQLogQuant expects two inputs");
    }
    return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
}

size_t ORT_API_CALL RepQLogQuant_GetInputTypeCount(const OrtCustomOp* op) {
    (void)op;
    return 2;
}

ONNXTensorElementDataType ORT_API_CALL RepQLogQuant_GetOutputType(
        const OrtCustomOp* op, size_t idx) {
    (void)op;
    (void)idx;
    return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
}

size_t ORT_API_CALL RepQLogQuant_GetOutputTypeCount(const OrtCustomOp* op) {
    (void)op;
    return 1;
}

OrtCustomOpInputOutputCharacteristic ORT_API_CALL RepQLogQuant_GetInputCharacteristic(
        const OrtCustomOp* op, size_t idx) {
    (void)op;
    (void)idx;
    return INPUT_OUTPUT_REQUIRED;
}

OrtCustomOpInputOutputCharacteristic ORT_API_CALL RepQLogQuant_GetOutputCharacteristic(
        const OrtCustomOp* op, size_t idx) {
    (void)op;
    (void)idx;
    return INPUT_OUTPUT_REQUIRED;
}

OrtCustomOp make_repq_log_quant_op() {
    OrtCustomOp op{};
    op.version = ORT_API_VERSION;
    op.CreateKernel = RepQLogQuant_CreateKernel;
    op.GetName = RepQLogQuant_GetName;
    op.GetExecutionProviderType = RepQLogQuant_GetExecutionProviderType;
    op.GetInputType = RepQLogQuant_GetInputType;
    op.GetInputTypeCount = RepQLogQuant_GetInputTypeCount;
    op.GetOutputType = RepQLogQuant_GetOutputType;
    op.GetOutputTypeCount = RepQLogQuant_GetOutputTypeCount;
    op.KernelCompute = RepQLogQuant_Compute;
    op.KernelDestroy = RepQLogQuant_Destroy;
    op.GetInputCharacteristic = RepQLogQuant_GetInputCharacteristic;
    op.GetOutputCharacteristic = RepQLogQuant_GetOutputCharacteristic;
    return op;
}

inline int32_t quantize_repq_log_code(float x_val, float delta, int32_t levels) {
    if (!(x_val > 0.0f) || !(delta > 0.0f)) {
        return -1;
    }

    float q_float = std::round(-std::log2(x_val / delta) * 2.0f);
    if (q_float >= static_cast<float>(levels)) {
        return -1;
    }

    q_float = std::clamp(q_float, 0.0f, static_cast<float>(levels - 1));
    return static_cast<int32_t>(q_float);
}

inline float repq_log_value_from_code(int32_t q, float delta) {
    const float odd_mask = (q % 2 == 0) ? 1.0f : std::sqrt(2.0f);
    const float exponent = -std::ceil(static_cast<float>(q) / 2.0f);
    return std::exp2(exponent) * odd_mask * delta;
}

struct RepQLogMatMulKernel {
    const OrtApi* ort;
    int32_t n_bits;
    bool approximate;
    int64_t profile_node_index;
    std::string profile_label;
};

void* ORT_API_CALL RepQLogMatMul_CreateKernel(
        const OrtCustomOp* op, const OrtApi* api, const OrtKernelInfo* info) {
    (void)op;
    auto* kernel = new RepQLogMatMulKernel();
    kernel->ort = api;
    int64_t n_bits = get_optional_int64_attribute(api, info, "n_bits", 8);
    kernel->n_bits = static_cast<int32_t>(n_bits);
    const int64_t approximate = get_optional_int64_attribute(api, info, "approximate", 0);
    kernel->approximate = approximate != 0;
    kernel->profile_node_index = get_optional_int64_attribute(api, info, "profile_node_index", -1);
    kernel->profile_label = get_optional_string_attribute(api, info, "profile_label");
    return kernel;
}

void ORT_API_CALL RepQLogMatMul_Destroy(void* op_kernel) {
    delete static_cast<RepQLogMatMulKernel*>(op_kernel);
}

void ORT_API_CALL RepQLogMatMul_Compute(void* op_kernel, OrtKernelContext* ctx) {
    auto* kernel = static_cast<RepQLogMatMulKernel*>(op_kernel);
    const OrtApi* ort = kernel->ort;
    const uint64_t profile_start = ort_profile_read_cycles();
    const auto finish_profile = [&]() {
        ort_log_node_cycles(
            kernel->profile_node_index,
            kernel->profile_label,
            "RepQLogMatMul",
            ort_profile_read_cycles() - profile_start
        );
    };

    const OrtValue* a_value = nullptr;
    const OrtValue* delta_value = nullptr;
    const OrtValue* b_value = nullptr;
    const OrtValue* b_scale_value = nullptr;
    const OrtValue* b_zp_value = nullptr;
    ort->KernelContext_GetInput(ctx, 0, &a_value);
    ort->KernelContext_GetInput(ctx, 1, &delta_value);
    ort->KernelContext_GetInput(ctx, 2, &b_value);
    ort->KernelContext_GetInput(ctx, 3, &b_scale_value);
    ort->KernelContext_GetInput(ctx, 4, &b_zp_value);

    const auto a_dims = get_dims(ort, a_value);
    const auto b_dims = get_dims(ort, b_value);
    const auto out_dims = compute_matmul_output_shape(a_dims, b_dims);
    const int64_t rows = a_dims[a_dims.size() - 2];
    const int64_t depth = a_dims.back();
    const int64_t cols = b_dims.back();
    const int64_t batch = numel(a_dims) / (rows * depth);
    const int64_t a_stride = rows * depth;
    const int64_t b_stride = (b_dims.size() == 2) ? 0 : (depth * cols);
    const int64_t out_stride = rows * cols;

    const float delta = require_scalar_tensor_value<float>(ort, delta_value, "delta");
    const float b_scale = require_scalar_tensor_value<float>(ort, b_scale_value, "b_scale");
    const float b_zp_f = require_scalar_tensor_value<float>(ort, b_zp_value, "b_zero_point");
    const int32_t b_zp = std::clamp(static_cast<int32_t>(std::lrintf(b_zp_f)), 0, 255);
    const int32_t levels = 1 << kernel->n_bits;

    OrtValue* out_value = nullptr;
    ort->KernelContext_GetOutput(ctx, 0, out_dims.data(), out_dims.size(), &out_value);
    float* out = get_mutable_tensor_data<float>(ort, out_value);
    std::fill(out, out + batch * out_stride, 0.0f);

    if (!(delta > 0.0f) || !(b_scale > 0.0f)) {
        finish_profile();
        return;
    }

    const float* a = get_tensor_data<float>(ort, a_value);
    const float* b = get_tensor_data<float>(ort, b_value);
    const int mode = clamp_mode(g_ivit_execution_mode);

    if (kernel->approximate) {
        const float attn_sym_scale = repq_sym_scale_from_log_delta(delta);
        const float value_sym_scale = repq_sym_scale_from_uniform_params(
            &b_scale,
            true,
            &b_zp_f,
            true,
            1,
            kernel->n_bits
        );

        std::vector<int8_t> a_quant(static_cast<size_t>(a_stride), 0);
        std::vector<int8_t> b_quant(static_cast<size_t>(depth * cols), 0);
        std::vector<int32_t> temp(static_cast<size_t>(out_stride), 0);

        for (int64_t batch_idx = 0; batch_idx < batch; ++batch_idx) {
            const float* a_ptr = a + batch_idx * a_stride;
            const float* b_ptr = (b_dims.size() == 2) ? b : (b + batch_idx * b_stride);
            float* out_ptr = out + batch_idx * out_stride;

            std::fill(a_quant.begin(), a_quant.end(), static_cast<int8_t>(0));
            std::fill(b_quant.begin(), b_quant.end(), static_cast<int8_t>(0));
            std::fill(temp.begin(), temp.end(), 0);

            for (int64_t idx = 0; idx < a_stride; ++idx) {
                const int32_t q_code = quantize_repq_log_code(a_ptr[idx], delta, levels);
                const float q_value = (q_code < 0) ? 0.0f : repq_log_value_from_code(q_code, delta);
                a_quant[static_cast<size_t>(idx)] = quantize_sym_int8_scalar(q_value, attn_sym_scale);
            }
            for (int64_t idx = 0; idx < depth * cols; ++idx) {
                b_quant[static_cast<size_t>(idx)] = quantize_sym_int8_scalar(b_ptr[idx], value_sym_scale);
            }

#ifdef IVIT_USE_GEMMINI
            static int repq_log_matmul_approx_prints = 0;
            if (mode != 0 && repq_log_matmul_approx_prints < 8) {
                std::printf("RepQLogMatMul approx using Gemmini (%lld, %lld, %lld)\n",
                            static_cast<long long>(rows),
                            static_cast<long long>(cols),
                            static_cast<long long>(depth));
                ++repq_log_matmul_approx_prints;
            }
            if (mode == 0) {
                cpu_matmul_int32(a_quant.data(), b_quant.data(), temp.data(), rows, cols, depth);
            } else {
                gemmini_matmul_int32(a_quant.data(), b_quant.data(), temp.data(), rows, cols, depth, mode);
            }
#else
            cpu_matmul_int32(a_quant.data(), b_quant.data(), temp.data(), rows, cols, depth);
#endif

            for (int64_t idx = 0; idx < out_stride; ++idx) {
                out_ptr[idx] = attn_sym_scale * value_sym_scale * static_cast<float>(temp[static_cast<size_t>(idx)]);
            }
        }
        finish_profile();
        return;
    }

    std::vector<int32_t> q_codes(static_cast<size_t>(a_stride), -1);
    std::vector<uint8_t> active(levels, 0);
    std::vector<int32_t> active_levels;
    std::vector<int8_t> b_shifted(static_cast<size_t>(depth * cols), 0);
    std::vector<int8_t> mask(static_cast<size_t>(a_stride), 0);
    std::vector<int32_t> temp(static_cast<size_t>(out_stride), 0);
    std::vector<float> row_sum(static_cast<size_t>(rows), 0.0f);

    for (int64_t batch_idx = 0; batch_idx < batch; ++batch_idx) {
        const float* a_ptr = a + batch_idx * a_stride;
        const float* b_ptr = (b_dims.size() == 2) ? b : (b + batch_idx * b_stride);
        float* out_ptr = out + batch_idx * out_stride;

        std::fill(q_codes.begin(), q_codes.end(), -1);
        std::fill(active.begin(), active.end(), 0);
        active_levels.clear();
        std::fill(row_sum.begin(), row_sum.end(), 0.0f);

        for (int64_t idx = 0; idx < a_stride; ++idx) {
            const int32_t q = quantize_repq_log_code(a_ptr[idx], delta, levels);
            q_codes[static_cast<size_t>(idx)] = q;
            if (q >= 0 && active[static_cast<size_t>(q)] == 0) {
                active[static_cast<size_t>(q)] = 1;
                active_levels.push_back(q);
            }
        }
        std::sort(active_levels.begin(), active_levels.end());

        for (int64_t idx = 0; idx < depth * cols; ++idx) {
            long long u = std::llround(static_cast<double>(b_ptr[idx]) / static_cast<double>(b_scale)) + b_zp;
            u = std::clamp(u, 0LL, 255LL);
            b_shifted[static_cast<size_t>(idx)] = static_cast<int8_t>(u - 128LL);
        }

#ifdef IVIT_USE_GEMMINI
        static int repq_log_matmul_prints = 0;
        if (mode != 0 && repq_log_matmul_prints < 8) {
            std::printf("RepQLogMatMul exact using Gemmini with %zu active log levels (%lld, %lld, %lld)\n",
                        active_levels.size(),
                        static_cast<long long>(rows),
                        static_cast<long long>(cols),
                        static_cast<long long>(depth));
            ++repq_log_matmul_prints;
        }
#endif

        for (int32_t q : active_levels) {
            const float alpha = repq_log_value_from_code(q, delta);
            std::fill(mask.begin(), mask.end(), static_cast<int8_t>(0));
            std::fill(temp.begin(), temp.end(), 0);

            for (int64_t row = 0; row < rows; ++row) {
                int32_t count = 0;
                const int64_t row_offset = row * depth;
                for (int64_t col = 0; col < depth; ++col) {
                    const size_t idx = static_cast<size_t>(row_offset + col);
                    if (q_codes[idx] == q) {
                        mask[idx] = 1;
                        ++count;
                    }
                }
                row_sum[static_cast<size_t>(row)] += alpha * static_cast<float>(count);
            }

#ifdef IVIT_USE_GEMMINI
            if (mode == 0) {
                cpu_matmul_int32(mask.data(), b_shifted.data(), temp.data(), rows, cols, depth);
            } else {
                gemmini_matmul_int32(mask.data(), b_shifted.data(), temp.data(), rows, cols, depth, mode);
            }
#else
            cpu_matmul_int32(mask.data(), b_shifted.data(), temp.data(), rows, cols, depth);
#endif

            for (int64_t idx = 0; idx < out_stride; ++idx) {
                out_ptr[idx] += alpha * static_cast<float>(temp[static_cast<size_t>(idx)]);
            }
        }

        const float correction = b_scale * static_cast<float>(128 - b_zp);
        for (int64_t row = 0; row < rows; ++row) {
            const float row_bias = correction * row_sum[static_cast<size_t>(row)];
            for (int64_t col = 0; col < cols; ++col) {
                const size_t out_idx = static_cast<size_t>(row * cols + col);
                out_ptr[out_idx] = b_scale * out_ptr[out_idx] + row_bias;
            }
        }
    }
    finish_profile();
}

const char* ORT_API_CALL RepQLogMatMul_GetName(const OrtCustomOp* op) {
    (void)op;
    return "RepQLogMatMul";
}

const char* ORT_API_CALL RepQLogMatMul_GetExecutionProviderType(const OrtCustomOp* op) {
    (void)op;
    return "CPUExecutionProvider";
}

ONNXTensorElementDataType ORT_API_CALL RepQLogMatMul_GetInputType(
        const OrtCustomOp* op, size_t idx) {
    (void)op;
    if (idx > 4) {
        throw std::runtime_error("RepQLogMatMul expects five inputs");
    }
    return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
}

size_t ORT_API_CALL RepQLogMatMul_GetInputTypeCount(const OrtCustomOp* op) {
    (void)op;
    return 5;
}

ONNXTensorElementDataType ORT_API_CALL RepQLogMatMul_GetOutputType(
        const OrtCustomOp* op, size_t idx) {
    (void)op;
    (void)idx;
    return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
}

size_t ORT_API_CALL RepQLogMatMul_GetOutputTypeCount(const OrtCustomOp* op) {
    (void)op;
    return 1;
}

OrtCustomOpInputOutputCharacteristic ORT_API_CALL RepQLogMatMul_GetInputCharacteristic(
        const OrtCustomOp* op, size_t idx) {
    (void)op;
    (void)idx;
    return INPUT_OUTPUT_REQUIRED;
}

OrtCustomOpInputOutputCharacteristic ORT_API_CALL RepQLogMatMul_GetOutputCharacteristic(
        const OrtCustomOp* op, size_t idx) {
    (void)op;
    (void)idx;
    return INPUT_OUTPUT_REQUIRED;
}

OrtCustomOp make_repq_log_matmul_op() {
    OrtCustomOp op{};
    op.version = ORT_API_VERSION;
    op.CreateKernel = RepQLogMatMul_CreateKernel;
    op.GetName = RepQLogMatMul_GetName;
    op.GetExecutionProviderType = RepQLogMatMul_GetExecutionProviderType;
    op.GetInputType = RepQLogMatMul_GetInputType;
    op.GetInputTypeCount = RepQLogMatMul_GetInputTypeCount;
    op.GetOutputType = RepQLogMatMul_GetOutputType;
    op.GetOutputTypeCount = RepQLogMatMul_GetOutputTypeCount;
    op.KernelCompute = RepQLogMatMul_Compute;
    op.KernelDestroy = RepQLogMatMul_Destroy;
    op.GetInputCharacteristic = RepQLogMatMul_GetInputCharacteristic;
    op.GetOutputCharacteristic = RepQLogMatMul_GetOutputCharacteristic;
    return op;
}

struct RepQUniformMatMulKernel {
    const OrtApi* ort;
    int n_bits;
    bool approximate;
    int64_t profile_node_index;
    std::string profile_label;
};

void* ORT_API_CALL RepQUniformMatMul_CreateKernel(
        const OrtCustomOp* op, const OrtApi* api, const OrtKernelInfo* info) {
    auto* kernel = new RepQUniformMatMulKernel();
    kernel->ort = api;
    int64_t n_bits = get_optional_int64_attribute(api, info, "n_bits", 8);
    kernel->n_bits = static_cast<int>(n_bits);
    const int64_t approximate = get_optional_int64_attribute(api, info, "approximate", 0);
    kernel->approximate = approximate != 0;
    kernel->profile_node_index = get_optional_int64_attribute(api, info, "profile_node_index", -1);
    kernel->profile_label = get_optional_string_attribute(api, info, "profile_label");
    return kernel;
}

void ORT_API_CALL RepQUniformMatMul_Destroy(void* op_kernel) {
    delete static_cast<RepQUniformMatMulKernel*>(op_kernel);
}

void ORT_API_CALL RepQUniformMatMul_Compute(void* op_kernel, OrtKernelContext* ctx) {
    auto* kernel = static_cast<RepQUniformMatMulKernel*>(op_kernel);
    const OrtApi* ort = kernel->ort;
    const uint64_t profile_start = ort_profile_read_cycles();
    const auto finish_profile = [&]() {
        ort_log_node_cycles(
            kernel->profile_node_index,
            kernel->profile_label,
            "RepQUniformMatMul",
            ort_profile_read_cycles() - profile_start
        );
    };

    const OrtValue* a_value = nullptr;
    const OrtValue* a_scale_value = nullptr;
    const OrtValue* a_zp_value = nullptr;
    const OrtValue* b_value = nullptr;
    const OrtValue* b_scale_value = nullptr;
    const OrtValue* b_zp_value = nullptr;
    ort->KernelContext_GetInput(ctx, 0, &a_value);
    ort->KernelContext_GetInput(ctx, 1, &a_scale_value);
    ort->KernelContext_GetInput(ctx, 2, &a_zp_value);
    ort->KernelContext_GetInput(ctx, 3, &b_value);
    ort->KernelContext_GetInput(ctx, 4, &b_scale_value);
    ort->KernelContext_GetInput(ctx, 5, &b_zp_value);

    const auto a_dims = get_dims(ort, a_value);
    const auto b_dims = get_dims(ort, b_value);
    const auto out_dims = compute_matmul_output_shape(a_dims, b_dims);
    const int64_t rows = a_dims[a_dims.size() - 2];
    const int64_t depth = a_dims.back();
    const int64_t cols = b_dims.back();
    const int64_t batch = numel(a_dims) / (rows * depth);
    const int64_t a_stride = rows * depth;
    const int64_t b_stride = (b_dims.size() == 2) ? 0 : (depth * cols);
    const int64_t out_stride = rows * cols;

    const float a_scale = require_scalar_tensor_value<float>(ort, a_scale_value, "a_scale");
    const float a_zp = require_scalar_tensor_value<float>(ort, a_zp_value, "a_zero_point");
    const auto b_scale_dims = get_dims(ort, b_scale_value);
    const auto b_zp_dims = get_dims(ort, b_zp_value);
    const bool b_scale_scalar = scale_is_scalar(ort, b_scale_value);
    const bool b_zp_scalar = scale_is_scalar(ort, b_zp_value);
    if (!b_scale_scalar && !(b_scale_dims.size() == 1 && b_scale_dims[0] == cols)) {
        throw std::runtime_error("RepQUniformMatMul expects B scale to be scalar or length-N");
    }
    if (!b_zp_scalar && !(b_zp_dims.size() == 1 && b_zp_dims[0] == cols)) {
        throw std::runtime_error("RepQUniformMatMul expects B zero-point to be scalar or length-N");
    }

    OrtValue* out_value = nullptr;
    ort->KernelContext_GetOutput(ctx, 0, out_dims.data(), out_dims.size(), &out_value);
    float* out = get_mutable_tensor_data<float>(ort, out_value);
    std::fill(out, out + batch * out_stride, 0.0f);

    if (!(a_scale > 0.0f)) {
        finish_profile();
        return;
    }

    const float* a = get_tensor_data<float>(ort, a_value);
    const float* b = get_tensor_data<float>(ort, b_value);
    const float* b_scales = get_tensor_data<float>(ort, b_scale_value);
    const float* b_zps = get_tensor_data<float>(ort, b_zp_value);
    const int levels = 1 << kernel->n_bits;
    const int mode = clamp_mode(g_ivit_execution_mode);

    if (kernel->approximate) {
        const float a_sym_scale = repq_sym_scale_from_uniform_params(
            &a_scale,
            true,
            &a_zp,
            true,
            1,
            kernel->n_bits
        );
        const float b_sym_scale = repq_sym_scale_from_uniform_params(
            b_scales,
            b_scale_scalar,
            b_zps,
            b_zp_scalar,
            cols,
            kernel->n_bits
        );

        std::vector<int8_t> a_quant(static_cast<size_t>(a_stride), 0);
        std::vector<int8_t> b_quant(static_cast<size_t>(depth * cols), 0);
        std::vector<int32_t> temp(static_cast<size_t>(out_stride), 0);

        for (int64_t batch_idx = 0; batch_idx < batch; ++batch_idx) {
            const float* a_ptr = a + batch_idx * a_stride;
            const float* b_ptr = (b_dims.size() == 2) ? b : (b + batch_idx * b_stride);
            float* out_ptr = out + batch_idx * out_stride;

            std::fill(a_quant.begin(), a_quant.end(), static_cast<int8_t>(0));
            std::fill(b_quant.begin(), b_quant.end(), static_cast<int8_t>(0));
            std::fill(temp.begin(), temp.end(), 0);

            for (int64_t idx = 0; idx < a_stride; ++idx) {
                a_quant[static_cast<size_t>(idx)] = quantize_sym_int8_scalar(a_ptr[idx], a_sym_scale);
            }
            for (int64_t idx = 0; idx < depth * cols; ++idx) {
                b_quant[static_cast<size_t>(idx)] = quantize_sym_int8_scalar(b_ptr[idx], b_sym_scale);
            }

#ifdef IVIT_USE_GEMMINI
            static int repq_uniform_matmul_approx_prints = 0;
            if (mode != 0 && repq_uniform_matmul_approx_prints < 8) {
                std::printf("RepQUniformMatMul approx using Gemmini (%lld, %lld, %lld)\n",
                            static_cast<long long>(rows),
                            static_cast<long long>(cols),
                            static_cast<long long>(depth));
                ++repq_uniform_matmul_approx_prints;
            }
            if (mode == 0) {
                cpu_matmul_int32(a_quant.data(), b_quant.data(), temp.data(), rows, cols, depth);
            } else {
                gemmini_matmul_int32(a_quant.data(), b_quant.data(), temp.data(), rows, cols, depth, mode);
            }
#else
            cpu_matmul_int32(a_quant.data(), b_quant.data(), temp.data(), rows, cols, depth);
#endif

            for (int64_t idx = 0; idx < out_stride; ++idx) {
                out_ptr[idx] = a_sym_scale * b_sym_scale * static_cast<float>(temp[static_cast<size_t>(idx)]);
            }
        }
        finish_profile();
        return;
    }

    const long long a_qmin = static_cast<long long>(std::ceil(-static_cast<double>(a_zp)));
    const long long a_qmax = static_cast<long long>(std::floor(static_cast<double>(levels - 1) - static_cast<double>(a_zp)));
    const long long a_shift_center = a_qmin + 128LL;

    std::vector<float> b_scale_vec(static_cast<size_t>(cols), 0.0f);
    std::vector<float> b_zp_vec(static_cast<size_t>(cols), 0.0f);
    std::vector<long long> b_qmin(static_cast<size_t>(cols), 0);
    std::vector<long long> b_qmax(static_cast<size_t>(cols), 0);
    std::vector<long long> b_shift_center(static_cast<size_t>(cols), 0);
    for (int64_t col = 0; col < cols; ++col) {
        const float scale = b_scale_scalar ? b_scales[0] : b_scales[col];
        const float zp = b_zp_scalar ? b_zps[0] : b_zps[col];
        b_scale_vec[static_cast<size_t>(col)] = scale;
        b_zp_vec[static_cast<size_t>(col)] = zp;
        b_qmin[static_cast<size_t>(col)] = static_cast<long long>(std::ceil(-static_cast<double>(zp)));
        b_qmax[static_cast<size_t>(col)] =
            static_cast<long long>(std::floor(static_cast<double>(levels - 1) - static_cast<double>(zp)));
        b_shift_center[static_cast<size_t>(col)] = b_qmin[static_cast<size_t>(col)] + 128LL;
    }

    std::vector<int8_t> a_shifted(static_cast<size_t>(a_stride), 0);
    std::vector<int8_t> b_shifted(static_cast<size_t>(depth * cols), 0);
    std::vector<int32_t> temp(static_cast<size_t>(out_stride), 0);
    std::vector<int64_t> row_sum(static_cast<size_t>(rows), 0);
    std::vector<int64_t> col_sum(static_cast<size_t>(cols), 0);

    for (int64_t batch_idx = 0; batch_idx < batch; ++batch_idx) {
        const float* a_ptr = a + batch_idx * a_stride;
        const float* b_ptr = (b_dims.size() == 2) ? b : (b + batch_idx * b_stride);
        float* out_ptr = out + batch_idx * out_stride;

        std::fill(a_shifted.begin(), a_shifted.end(), static_cast<int8_t>(0));
        std::fill(b_shifted.begin(), b_shifted.end(), static_cast<int8_t>(0));
        std::fill(temp.begin(), temp.end(), 0);
        std::fill(row_sum.begin(), row_sum.end(), 0);
        std::fill(col_sum.begin(), col_sum.end(), 0);

        for (int64_t row = 0; row < rows; ++row) {
            for (int64_t k = 0; k < depth; ++k) {
                const size_t idx = static_cast<size_t>(row * depth + k);
                long long q = std::llround(static_cast<double>(a_ptr[idx]) / static_cast<double>(a_scale));
                q = std::clamp(q, a_qmin, a_qmax);
                const long long shifted = q - a_shift_center;
                a_shifted[idx] = static_cast<int8_t>(shifted);
                row_sum[static_cast<size_t>(row)] += shifted;
            }
        }

        for (int64_t k = 0; k < depth; ++k) {
            for (int64_t col = 0; col < cols; ++col) {
                const size_t idx = static_cast<size_t>(k * cols + col);
                const float b_scale = b_scale_vec[static_cast<size_t>(col)];
                if (!(b_scale > 0.0f)) {
                    continue;
                }
                long long q = std::llround(static_cast<double>(b_ptr[idx]) / static_cast<double>(b_scale));
                q = std::clamp(q, b_qmin[static_cast<size_t>(col)], b_qmax[static_cast<size_t>(col)]);
                const long long shifted = q - b_shift_center[static_cast<size_t>(col)];
                b_shifted[idx] = static_cast<int8_t>(shifted);
                col_sum[static_cast<size_t>(col)] += shifted;
            }
        }

#ifdef IVIT_USE_GEMMINI
        static int repq_uniform_matmul_prints = 0;
        if (mode != 0 && repq_uniform_matmul_prints < 8) {
            std::printf("RepQUniformMatMul exact using Gemmini (%lld, %lld, %lld)\n",
                        static_cast<long long>(rows),
                        static_cast<long long>(cols),
                        static_cast<long long>(depth));
            ++repq_uniform_matmul_prints;
        }
#endif

#ifdef IVIT_USE_GEMMINI
        if (mode == 0) {
            cpu_matmul_int32(a_shifted.data(), b_shifted.data(), temp.data(), rows, cols, depth);
        } else {
            gemmini_matmul_int32(a_shifted.data(), b_shifted.data(), temp.data(), rows, cols, depth, mode);
        }
#else
        cpu_matmul_int32(a_shifted.data(), b_shifted.data(), temp.data(), rows, cols, depth);
#endif

        for (int64_t row = 0; row < rows; ++row) {
            const double row_sum_d = static_cast<double>(row_sum[static_cast<size_t>(row)]);
            for (int64_t col = 0; col < cols; ++col) {
                const size_t out_idx = static_cast<size_t>(row * cols + col);
                const double b_scale = static_cast<double>(b_scale_vec[static_cast<size_t>(col)]);
                const double c_b = static_cast<double>(b_shift_center[static_cast<size_t>(col)]);
                const double col_sum_d = static_cast<double>(col_sum[static_cast<size_t>(col)]);
                const double acc = static_cast<double>(temp[out_idx]);
                const double exact =
                    acc +
                    c_b * row_sum_d +
                    static_cast<double>(a_shift_center) * col_sum_d +
                    static_cast<double>(depth) * static_cast<double>(a_shift_center) * c_b;
                out_ptr[out_idx] = static_cast<float>(static_cast<double>(a_scale) * b_scale * exact);
            }
        }
    }
    finish_profile();
}

const char* ORT_API_CALL RepQUniformMatMul_GetName(const OrtCustomOp* op) {
    (void)op;
    return "RepQUniformMatMul";
}

const char* ORT_API_CALL RepQUniformMatMul_GetExecutionProviderType(const OrtCustomOp* op) {
    (void)op;
    return "CPUExecutionProvider";
}

ONNXTensorElementDataType ORT_API_CALL RepQUniformMatMul_GetInputType(
        const OrtCustomOp* op, size_t idx) {
    (void)op;
    (void)idx;
    return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
}

size_t ORT_API_CALL RepQUniformMatMul_GetInputTypeCount(const OrtCustomOp* op) {
    (void)op;
    return 6;
}

ONNXTensorElementDataType ORT_API_CALL RepQUniformMatMul_GetOutputType(
        const OrtCustomOp* op, size_t idx) {
    (void)op;
    (void)idx;
    return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
}

size_t ORT_API_CALL RepQUniformMatMul_GetOutputTypeCount(const OrtCustomOp* op) {
    (void)op;
    return 1;
}

OrtCustomOpInputOutputCharacteristic ORT_API_CALL RepQUniformMatMul_GetInputCharacteristic(
        const OrtCustomOp* op, size_t idx) {
    (void)op;
    (void)idx;
    return INPUT_OUTPUT_REQUIRED;
}

OrtCustomOpInputOutputCharacteristic ORT_API_CALL RepQUniformMatMul_GetOutputCharacteristic(
        const OrtCustomOp* op, size_t idx) {
    (void)op;
    (void)idx;
    return INPUT_OUTPUT_REQUIRED;
}

OrtCustomOp make_repq_uniform_matmul_op() {
    OrtCustomOp op{};
    op.version = ORT_API_VERSION;
    op.CreateKernel = RepQUniformMatMul_CreateKernel;
    op.GetName = RepQUniformMatMul_GetName;
    op.GetExecutionProviderType = RepQUniformMatMul_GetExecutionProviderType;
    op.GetInputType = RepQUniformMatMul_GetInputType;
    op.GetInputTypeCount = RepQUniformMatMul_GetInputTypeCount;
    op.GetOutputType = RepQUniformMatMul_GetOutputType;
    op.GetOutputTypeCount = RepQUniformMatMul_GetOutputTypeCount;
    op.KernelCompute = RepQUniformMatMul_Compute;
    op.KernelDestroy = RepQUniformMatMul_Destroy;
    op.GetInputCharacteristic = RepQUniformMatMul_GetInputCharacteristic;
    op.GetOutputCharacteristic = RepQUniformMatMul_GetOutputCharacteristic;
    return op;
}

OrtCustomOp g_gemmini_matmul_integer_op{};
OrtCustomOp g_requantize_int32_op{};
OrtCustomOp g_requantize_int32_to_int16_op{};
OrtCustomOp g_requantize_int8_to_int16_op{};
OrtCustomOp g_requantize_int16_op{};
OrtCustomOp g_requantize_int16_to_int8_op{};
OrtCustomOp g_requantize_int64_op{};
OrtCustomOp g_requantize_int64_to_int16_op{};
OrtCustomOp g_qlinear_add_int16_op{};
OrtCustomOp g_repq_log_quant_op{};
OrtCustomOp g_repq_log_matmul_op{};
OrtCustomOp g_repq_uniform_matmul_op{};
bool g_custom_ops_initialized = false;

void ensure_custom_ops_initialized() {
    if (g_custom_ops_initialized) {
        return;
    }
    g_gemmini_matmul_integer_op = make_gemmini_matmul_integer_op();
    g_requantize_int32_op = make_requantize_int32_op();
    g_requantize_int32_to_int16_op = make_requantize_int32_to_int16_op();
    g_requantize_int8_to_int16_op = make_requantize_int8_to_int16_op();
    g_requantize_int16_op = make_requantize_int16_op();
    g_requantize_int16_to_int8_op = make_requantize_int16_to_int8_op();
    g_requantize_int64_op = make_requantize_int64_op();
    g_requantize_int64_to_int16_op = make_requantize_int64_to_int16_op();
    g_qlinear_add_int16_op = make_qlinear_add_int16_op();
    g_repq_log_quant_op = make_repq_log_quant_op();
    g_repq_log_matmul_op = make_repq_log_matmul_op();
    g_repq_uniform_matmul_op = make_repq_uniform_matmul_op();
    g_custom_ops_initialized = true;
}

}  // namespace

extern "C" void ORT_API_CALL SetIvitExecutionMode(int mode) {
    g_ivit_execution_mode = clamp_mode(mode);
}

extern "C" OrtStatus* ORT_API_CALL AddIvitGemminiOps(
        OrtCustomOpDomain* domain, const OrtApi* ort) {
    ensure_custom_ops_initialized();
    OrtStatus* status = ort->CustomOpDomain_Add(domain, &g_gemmini_matmul_integer_op);
    if (status != nullptr) {
        return status;
    }
    status = ort->CustomOpDomain_Add(domain, &g_requantize_int32_op);
    if (status != nullptr) {
        return status;
    }
    status = ort->CustomOpDomain_Add(domain, &g_requantize_int32_to_int16_op);
    if (status != nullptr) {
        return status;
    }
    status = ort->CustomOpDomain_Add(domain, &g_requantize_int8_to_int16_op);
    if (status != nullptr) {
        return status;
    }
    status = ort->CustomOpDomain_Add(domain, &g_requantize_int16_op);
    if (status != nullptr) {
        return status;
    }
    status = ort->CustomOpDomain_Add(domain, &g_requantize_int16_to_int8_op);
    if (status != nullptr) {
        return status;
    }
    status = ort->CustomOpDomain_Add(domain, &g_requantize_int64_op);
    if (status != nullptr) {
        return status;
    }
    status = ort->CustomOpDomain_Add(domain, &g_requantize_int64_to_int16_op);
    if (status != nullptr) {
        return status;
    }
    status = ort->CustomOpDomain_Add(domain, &g_qlinear_add_int16_op);
    if (status != nullptr) {
        return status;
    }
    status = ort->CustomOpDomain_Add(domain, &g_repq_log_quant_op);
    if (status != nullptr) {
        return status;
    }
    status = ort->CustomOpDomain_Add(domain, &g_repq_log_matmul_op);
    if (status != nullptr) {
        return status;
    }
    status = ort->CustomOpDomain_Add(domain, &g_repq_uniform_matmul_op);
    return status;
}
