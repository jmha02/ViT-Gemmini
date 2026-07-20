/*
 * ivit_ops.c — ORT custom ops for I-ViT integer-only operations.
 *
 * Implements three custom ops in domain "ivit":
 *
 *   ivit.Shiftmax      [int8 x]  → [int8]    attr: x0 (int64)
 *   ivit.ShiftmaxInt32 [int32 x] → [int8]    attr: x0 (int64)
 *   ivit.ShiftGELU  [int8 x] → [int32]   attr: x0 (int64)
 *   ivit.QLayernorm      [int8 x, int32 bias]  → [int32]
 *   ivit.QLayernormInt16 [int16 x, int32 bias] → [int32]
 *   ivit.QLayernormInt32 [int32 x, int32 bias] → [int32]
 *   ivit.QLayernormI64      [int8 x, int64 bias]  → [int64]
 *   ivit.QLayernormInt16I64 [int16 x, int64 bias] → [int64]
 *   ivit.QLayernormInt32I64 [int32 x, int64 bias] → [int64]
 *
 * All algorithms match the I-ViT integer reference implementations.
 *
 * Build (RISC-V Linux):
 *   riscv64-unknown-linux-gnu-gcc -O2 -march=rv64imafdc -mabi=lp64d \
 *       -I<ort_riscv>/include/onnxruntime/core/session \
 *       -c ivit_ops.c -o ivit_ops.o
 *   ar rcs build/ort/ort_ivit_ops/libivit_ops.a ivit_ops.o
 */

#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "onnxruntime_c_api.h"
#include "ivit_gemmini_ops.h"

/* ── ORT API global ───────────────────────────────────────────────────────── */
static const OrtApi *g_ort = NULL;

/* ── Integer shift-exp (shared by Shiftmax and ShiftGELU) ────────────────── */
/*
 * shift_exp(data, x0, n):
 *   data = data + (data >> 1) - (data >> 4)
 *   data = max(data, n * x0)
 *   q    = data / x0   (truncating toward zero)
 *   r    = data - q * x0
 *   e    = (r >> 1) - x0
 *   return e << (n - q)         [n - q is always >= 0]
 */
static int32_t shift_exp(int32_t data, int32_t x0, int32_t n)
{
    /* polynomial adjustment */
    data = data + (data >> 1) - (data >> 4);

    int32_t floor_val = n * x0;   /* x0 < 0, so floor_val < 0 */
    if (data < floor_val) data = floor_val;

    /* truncated integer division: C99 truncates toward zero */
    int32_t q = data / x0;
    int32_t r = data - q * x0;

    int32_t e = (r >> 1) - x0;
    int32_t shift = n - q;
    if (shift < 0) shift = 0;   /* safety clamp */
    if (shift > 31) return 0;   /* underflow */
    return e << shift;
}

/* ══════════════════════════════════════════════════════════════════════════
 *  ivit.Shiftmax  —  integer softmax
 *  Input : int8  [*, N]
 *  Output: int8  [*, N]
 *  Attr  : x0 (int64) = int(-1/input_scale - 1)
 * ══════════════════════════════════════════════════════════════════════════ */

typedef struct {
    const OrtApi *ort;
    int32_t       x0;  /* x0 for shift_exp */
} ShiftmaxKernel;

static void *ORT_API_CALL Shiftmax_CreateKernel(
        const OrtCustomOp *op, const OrtApi *api, const OrtKernelInfo *info)
{
    (void)op;
    ShiftmaxKernel *k = (ShiftmaxKernel *)malloc(sizeof(ShiftmaxKernel));
    k->ort = api;
    int64_t x0_attr = 0;
    api->KernelInfoGetAttribute_int64(info, "x0", &x0_attr);
    k->x0 = (int32_t)x0_attr;
    return k;
}

static void ORT_API_CALL Shiftmax_Destroy(void *op_kernel)
{
    free(op_kernel);
}

static void ORT_API_CALL Shiftmax_Compute(void *op_kernel, OrtKernelContext *ctx)
{
    ShiftmaxKernel *k = (ShiftmaxKernel *)op_kernel;
    const OrtApi *ort = k->ort;
    const int32_t x0 = k->x0;
    const int32_t n  = 16;  /* TVM quantized_softmax uses n=16 */

    const OrtValue *in_val = NULL;
    ort->KernelContext_GetInput(ctx, 0, &in_val);

    OrtTensorTypeAndShapeInfo *shape_info = NULL;
    ort->GetTensorTypeAndShape(in_val, &shape_info);
    size_t ndim = 0;
    ort->GetDimensionsCount(shape_info, &ndim);
    int64_t dims[8] = {0};
    ort->GetDimensions(shape_info, dims, ndim);
    ort->ReleaseTensorTypeAndShapeInfo(shape_info);

    const int8_t *x = NULL;
    ort->GetTensorMutableData((OrtValue *)(uintptr_t)in_val, (void **)&x);

    /* last dimension = softmax axis */
    int64_t last = dims[ndim - 1];
    int64_t outer = 1;
    for (size_t i = 0; i < ndim - 1; i++) outer *= dims[i];

    OrtValue *out_val = NULL;
    ort->KernelContext_GetOutput(ctx, 0, dims, ndim, &out_val);
    int8_t *y = NULL;
    ort->GetTensorMutableData(out_val, (void **)&y);

    for (int64_t b = 0; b < outer; b++) {
        const int8_t *row_in  = x + b * last;
        int8_t       *row_out = y + b * last;

        /* find max */
        int32_t mx = (int32_t)row_in[0];
        for (int64_t i = 1; i < last; i++)
            if ((int32_t)row_in[i] > mx) mx = (int32_t)row_in[i];

        /* compute exp_int for each element + sum */
        int64_t sum = 0;
        int32_t exp_buf[1024];  /* max seq dimension we expect */
        int64_t cap = last < 1024 ? last : 1024;
        for (int64_t i = 0; i < cap; i++) {
            int32_t c = (int32_t)row_in[i] - mx;
            exp_buf[i] = shift_exp(c, x0, n);
            sum += (int64_t)exp_buf[i];
        }
        if (sum == 0) sum = 1;

        int64_t factor = (int64_t)0x7FFFFFFF;  /* 2^31 - 1 */
        int64_t scale  = factor / sum;

        for (int64_t i = 0; i < cap; i++) {
            int64_t v = ((scale * (int64_t)exp_buf[i]) + (((int64_t)1) << 23)) >> 24;
            if (v >  127) v =  127;
            if (v < -128) v = -128;
            row_out[i] = (int8_t)v;
        }
        /* handle overflow beyond buffer */
        for (int64_t i = cap; i < last; i++) row_out[i] = 0;
    }
}

typedef struct {
    const OrtApi *ort;
    int32_t       x0;
} ShiftmaxInt32Kernel;

static void *ORT_API_CALL ShiftmaxInt32_CreateKernel(
        const OrtCustomOp *op, const OrtApi *api, const OrtKernelInfo *info)
{
    (void)op;
    ShiftmaxInt32Kernel *k = (ShiftmaxInt32Kernel *)malloc(sizeof(ShiftmaxInt32Kernel));
    k->ort = api;
    int64_t x0_attr = 0;
    api->KernelInfoGetAttribute_int64(info, "x0", &x0_attr);
    k->x0 = (int32_t)x0_attr;
    return k;
}

static void ORT_API_CALL ShiftmaxInt32_Destroy(void *op_kernel)
{
    free(op_kernel);
}

static void ORT_API_CALL ShiftmaxInt32_Compute(void *op_kernel, OrtKernelContext *ctx)
{
    ShiftmaxInt32Kernel *k = (ShiftmaxInt32Kernel *)op_kernel;
    const OrtApi *ort = k->ort;
    const int32_t x0 = k->x0;
    const int32_t n  = 16;

    const OrtValue *in_val = NULL;
    ort->KernelContext_GetInput(ctx, 0, &in_val);

    OrtTensorTypeAndShapeInfo *shape_info = NULL;
    ort->GetTensorTypeAndShape(in_val, &shape_info);
    size_t ndim = 0;
    ort->GetDimensionsCount(shape_info, &ndim);
    int64_t dims[8] = {0};
    ort->GetDimensions(shape_info, dims, ndim);
    ort->ReleaseTensorTypeAndShapeInfo(shape_info);

    const int32_t *x = NULL;
    ort->GetTensorMutableData((OrtValue *)(uintptr_t)in_val, (void **)&x);

    int64_t last = dims[ndim - 1];
    int64_t outer = 1;
    for (size_t i = 0; i < ndim - 1; i++) outer *= dims[i];

    OrtValue *out_val = NULL;
    ort->KernelContext_GetOutput(ctx, 0, dims, ndim, &out_val);
    int8_t *y = NULL;
    ort->GetTensorMutableData(out_val, (void **)&y);

    for (int64_t b = 0; b < outer; b++) {
        const int32_t *row_in  = x + b * last;
        int8_t        *row_out = y + b * last;

        int32_t mx = row_in[0];
        for (int64_t i = 1; i < last; i++)
            if (row_in[i] > mx) mx = row_in[i];

        int64_t sum = 0;
        int32_t exp_buf[1024];
        int64_t cap = last < 1024 ? last : 1024;
        for (int64_t i = 0; i < cap; i++) {
            int32_t c = row_in[i] - mx;
            exp_buf[i] = shift_exp(c, x0, n);
            sum += (int64_t)exp_buf[i];
        }
        if (sum == 0) sum = 1;

        int64_t factor = (int64_t)0x7FFFFFFF;
        int64_t scale  = factor / sum;

        for (int64_t i = 0; i < cap; i++) {
            int64_t v = ((scale * (int64_t)exp_buf[i]) + (((int64_t)1) << 23)) >> 24;
            if (v >  127) v =  127;
            if (v < -128) v = -128;
            row_out[i] = (int8_t)v;
        }
        for (int64_t i = cap; i < last; i++) row_out[i] = 0;
    }
}

static const char *ORT_API_CALL ShiftmaxInt32_GetName(const OrtCustomOp *op)
{ (void)op; return "ShiftmaxInt32"; }

static const char *ORT_API_CALL ShiftmaxInt32_GetExecutionProviderType(const OrtCustomOp *op)
{ (void)op; return "CPUExecutionProvider"; }

static ONNXTensorElementDataType ORT_API_CALL ShiftmaxInt32_GetInputType(
        const OrtCustomOp *op, size_t idx)
{ (void)op; (void)idx; return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32; }

static size_t ORT_API_CALL ShiftmaxInt32_GetInputTypeCount(const OrtCustomOp *op)
{ (void)op; return 1; }

static ONNXTensorElementDataType ORT_API_CALL ShiftmaxInt32_GetOutputType(
        const OrtCustomOp *op, size_t idx)
{ (void)op; (void)idx; return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8; }

static size_t ORT_API_CALL ShiftmaxInt32_GetOutputTypeCount(const OrtCustomOp *op)
{ (void)op; return 1; }

static OrtCustomOpInputOutputCharacteristic ORT_API_CALL
ShiftmaxInt32_GetInputCharacteristic(const OrtCustomOp *op, size_t idx)
{ (void)op; (void)idx; return INPUT_OUTPUT_REQUIRED; }

static OrtCustomOpInputOutputCharacteristic ORT_API_CALL
ShiftmaxInt32_GetOutputCharacteristic(const OrtCustomOp *op, size_t idx)
{ (void)op; (void)idx; return INPUT_OUTPUT_REQUIRED; }

static OrtCustomOp g_shiftmax_int32_op = {
    .version                    = ORT_API_VERSION,
    .CreateKernel               = ShiftmaxInt32_CreateKernel,
    .GetName                    = ShiftmaxInt32_GetName,
    .GetExecutionProviderType   = ShiftmaxInt32_GetExecutionProviderType,
    .GetInputType               = ShiftmaxInt32_GetInputType,
    .GetInputTypeCount          = ShiftmaxInt32_GetInputTypeCount,
    .GetOutputType              = ShiftmaxInt32_GetOutputType,
    .GetOutputTypeCount         = ShiftmaxInt32_GetOutputTypeCount,
    .KernelCompute              = ShiftmaxInt32_Compute,
    .KernelDestroy              = ShiftmaxInt32_Destroy,
    .GetInputCharacteristic     = ShiftmaxInt32_GetInputCharacteristic,
    .GetOutputCharacteristic    = ShiftmaxInt32_GetOutputCharacteristic,
};

static const char *ORT_API_CALL Shiftmax_GetName(const OrtCustomOp *op)
{ (void)op; return "Shiftmax"; }

static const char *ORT_API_CALL Shiftmax_GetExecutionProviderType(const OrtCustomOp *op)
{ (void)op; return "CPUExecutionProvider"; }

static ONNXTensorElementDataType ORT_API_CALL Shiftmax_GetInputType(
        const OrtCustomOp *op, size_t idx)
{ (void)op; (void)idx; return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8; }

static size_t ORT_API_CALL Shiftmax_GetInputTypeCount(const OrtCustomOp *op)
{ (void)op; return 1; }

static ONNXTensorElementDataType ORT_API_CALL Shiftmax_GetOutputType(
        const OrtCustomOp *op, size_t idx)
{ (void)op; (void)idx; return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8; }

static size_t ORT_API_CALL Shiftmax_GetOutputTypeCount(const OrtCustomOp *op)
{ (void)op; return 1; }

static OrtCustomOpInputOutputCharacteristic ORT_API_CALL
Shiftmax_GetInputCharacteristic(const OrtCustomOp *op, size_t idx)
{ (void)op; (void)idx; return INPUT_OUTPUT_REQUIRED; }

static OrtCustomOpInputOutputCharacteristic ORT_API_CALL
Shiftmax_GetOutputCharacteristic(const OrtCustomOp *op, size_t idx)
{ (void)op; (void)idx; return INPUT_OUTPUT_REQUIRED; }

static OrtCustomOp g_shiftmax_op = {
    .version                    = ORT_API_VERSION,
    .CreateKernel               = Shiftmax_CreateKernel,
    .GetName                    = Shiftmax_GetName,
    .GetExecutionProviderType   = Shiftmax_GetExecutionProviderType,
    .GetInputType               = Shiftmax_GetInputType,
    .GetInputTypeCount          = Shiftmax_GetInputTypeCount,
    .GetOutputType              = Shiftmax_GetOutputType,
    .GetOutputTypeCount         = Shiftmax_GetOutputTypeCount,
    .KernelCompute              = Shiftmax_Compute,
    .KernelDestroy              = Shiftmax_Destroy,
    .GetInputCharacteristic     = Shiftmax_GetInputCharacteristic,
    .GetOutputCharacteristic    = Shiftmax_GetOutputCharacteristic,
};

/* ══════════════════════════════════════════════════════════════════════════
 *  ivit.ShiftGELU  —  integer GELU approximation
 *  Input : int8  [*, C]
 *  Output: int32 [*, C]   (natural scale = input_scale / 128)
 *  Attr  : x0 (int64) = floor(-1 / (scaling_factor * 1.702))
 * ══════════════════════════════════════════════════════════════════════════ */

typedef struct {
    const OrtApi *ort;
    int32_t       x0;
} ShiftGELUKernel;

static void *ORT_API_CALL ShiftGELU_CreateKernel(
        const OrtCustomOp *op, const OrtApi *api, const OrtKernelInfo *info)
{
    (void)op;
    ShiftGELUKernel *k = (ShiftGELUKernel *)malloc(sizeof(ShiftGELUKernel));
    k->ort = api;
    int64_t x0_attr = 0;
    api->KernelInfoGetAttribute_int64(info, "x0", &x0_attr);
    k->x0 = (int32_t)x0_attr;
    return k;
}

static void ORT_API_CALL ShiftGELU_Destroy(void *op_kernel) { free(op_kernel); }

static void ORT_API_CALL ShiftGELU_Compute(void *op_kernel, OrtKernelContext *ctx)
{
    ShiftGELUKernel *k = (ShiftGELUKernel *)op_kernel;
    const OrtApi *ort = k->ort;
    const int32_t x0 = k->x0;
    const int32_t n  = 23;  /* ShiftGELU uses n=23 */

    const OrtValue *in_val = NULL;
    ort->KernelContext_GetInput(ctx, 0, &in_val);

    OrtTensorTypeAndShapeInfo *shape_info = NULL;
    ort->GetTensorTypeAndShape(in_val, &shape_info);
    size_t ndim = 0;
    ort->GetDimensionsCount(shape_info, &ndim);
    int64_t dims[8] = {0};
    ort->GetDimensions(shape_info, dims, ndim);
    ort->ReleaseTensorTypeAndShapeInfo(shape_info);

    const int8_t *x = NULL;
    ort->GetTensorMutableData((OrtValue *)(uintptr_t)in_val, (void **)&x);

    int64_t last  = dims[ndim - 1];
    int64_t outer = 1;
    for (size_t i = 0; i < ndim - 1; i++) outer *= dims[i];

    OrtValue *out_val = NULL;
    ort->KernelContext_GetOutput(ctx, 0, dims, ndim, &out_val);
    int32_t *y = NULL;
    ort->GetTensorMutableData(out_val, (void **)&y);

    for (int64_t b = 0; b < outer; b++) {
        const int8_t *row_in  = x + b * last;
        int32_t      *row_out = y + b * last;

        /* find max for numerical stability */
        int32_t mx = (int32_t)row_in[0];
        for (int64_t i = 1; i < last; i++)
            if ((int32_t)row_in[i] > mx) mx = (int32_t)row_in[i];

        /* exp of (x[i] - mx) for sigmoid numerator */
        int64_t exp_buf[2048];
        int64_t cap = last < 2048 ? last : 2048;
        for (int64_t i = 0; i < cap; i++) {
            int32_t c = (int32_t)row_in[i] - mx;
            exp_buf[i] = (int64_t)shift_exp(c, x0, n);
        }

        /* exp of (-mx) for sigmoid denominator term */
        int64_t exp_max_neg = (int64_t)shift_exp(-mx, x0, n);

        int64_t factor = (int64_t)0x7FFFFFFF;

        for (int64_t i = 0; i < cap; i++) {
            int64_t exp_sum = exp_buf[i] + exp_max_neg;
            if (exp_sum == 0) exp_sum = 1;
            int64_t sig = (factor / exp_sum * exp_buf[i]) >> 24;
            /* sig ∈ [0, 127], represents sigmoid * 128 approximately */
            row_out[i] = (int32_t)((int64_t)(int32_t)row_in[i] * sig);
        }
        for (int64_t i = cap; i < last; i++) row_out[i] = 0;
    }
}

static const char *ORT_API_CALL ShiftGELU_GetName(const OrtCustomOp *op)
{ (void)op; return "ShiftGELU"; }

static const char *ORT_API_CALL ShiftGELU_GetExecutionProviderType(const OrtCustomOp *op)
{ (void)op; return "CPUExecutionProvider"; }

static ONNXTensorElementDataType ORT_API_CALL ShiftGELU_GetInputType(
        const OrtCustomOp *op, size_t idx)
{ (void)op; (void)idx; return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8; }

static size_t ORT_API_CALL ShiftGELU_GetInputTypeCount(const OrtCustomOp *op)
{ (void)op; return 1; }

static ONNXTensorElementDataType ORT_API_CALL ShiftGELU_GetOutputType(
        const OrtCustomOp *op, size_t idx)
{ (void)op; (void)idx; return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32; }

static size_t ORT_API_CALL ShiftGELU_GetOutputTypeCount(const OrtCustomOp *op)
{ (void)op; return 1; }

static OrtCustomOpInputOutputCharacteristic ORT_API_CALL
ShiftGELU_GetInputCharacteristic(const OrtCustomOp *op, size_t idx)
{ (void)op; (void)idx; return INPUT_OUTPUT_REQUIRED; }

static OrtCustomOpInputOutputCharacteristic ORT_API_CALL
ShiftGELU_GetOutputCharacteristic(const OrtCustomOp *op, size_t idx)
{ (void)op; (void)idx; return INPUT_OUTPUT_REQUIRED; }

static OrtCustomOp g_shiftgelu_op = {
    .version                    = ORT_API_VERSION,
    .CreateKernel               = ShiftGELU_CreateKernel,
    .GetName                    = ShiftGELU_GetName,
    .GetExecutionProviderType   = ShiftGELU_GetExecutionProviderType,
    .GetInputType               = ShiftGELU_GetInputType,
    .GetInputTypeCount          = ShiftGELU_GetInputTypeCount,
    .GetOutputType              = ShiftGELU_GetOutputType,
    .GetOutputTypeCount         = ShiftGELU_GetOutputTypeCount,
    .KernelCompute              = ShiftGELU_Compute,
    .KernelDestroy              = ShiftGELU_Destroy,
    .GetInputCharacteristic     = ShiftGELU_GetInputCharacteristic,
    .GetOutputCharacteristic    = ShiftGELU_GetOutputCharacteristic,
};

/* ══════════════════════════════════════════════════════════════════════════
 *  ivit.QLayernorm  —  integer-only Layer Normalization
 *  Input 0: int8  [*, C]   activations
 *  Input 1: int32 [C]      bias_integer (per-channel)
 *  Output : int32 [*, C]   (scale = norm_scaling_factor)
 *
 *  Algorithm (matches TVM quantized_layernorm):
 *    x32 = cast(x, int32)
 *    mean = sum(x32) / C
 *    xc[c] = x32[c] - mean
 *    var = sum(xc * xc)             // scalar per token
 *    std = newton(var, init=2^16, 10 iters)
 *    scale = (2^31 - 1) / std
 *    out[c] = (scale * xc[c]) / 2 + bias[c]
 * ══════════════════════════════════════════════════════════════════════════ */

static int32_t tvm_layernorm_std(uint64_t val)
{
    int64_t s = ((int64_t)1) << 16;
    for (int i = 0; i < 10; i++) {
        s = (s + ((int64_t)val / s)) / 2;
    }
    return (int32_t)s;
}

static int64_t round_divide_i64(int64_t num, int64_t den)
{
    return (int64_t)llrint(((double)num) / ((double)den));
}

typedef struct {
    const OrtApi *ort;
} QLayernormKernel;

static void *ORT_API_CALL QLayernorm_CreateKernel(
        const OrtCustomOp *op, const OrtApi *api, const OrtKernelInfo *info)
{
    (void)op; (void)info;
    QLayernormKernel *k = (QLayernormKernel *)malloc(sizeof(QLayernormKernel));
    k->ort = api;
    return k;
}

static void ORT_API_CALL QLayernorm_Destroy(void *op_kernel) { free(op_kernel); }

static void ORT_API_CALL QLayernorm_Compute(void *op_kernel, OrtKernelContext *ctx)
{
    QLayernormKernel *k = (QLayernormKernel *)op_kernel;
    const OrtApi *ort = k->ort;

    /* input 0: int8 activations */
    const OrtValue *x_val = NULL;
    ort->KernelContext_GetInput(ctx, 0, &x_val);

    OrtTensorTypeAndShapeInfo *shape_info = NULL;
    ort->GetTensorTypeAndShape(x_val, &shape_info);
    size_t ndim = 0;
    ort->GetDimensionsCount(shape_info, &ndim);
    int64_t dims[8] = {0};
    ort->GetDimensions(shape_info, dims, ndim);
    ort->ReleaseTensorTypeAndShapeInfo(shape_info);

    const int8_t *x = NULL;
    ort->GetTensorMutableData((OrtValue *)(uintptr_t)x_val, (void **)&x);

    /* input 1: int32 bias */
    const OrtValue *bias_val = NULL;
    ort->KernelContext_GetInput(ctx, 1, &bias_val);
    const int32_t *bias = NULL;
    ort->GetTensorMutableData((OrtValue *)(uintptr_t)bias_val, (void **)&bias);

    int64_t C     = dims[ndim - 1];  /* feature dimension */
    int64_t outer = 1;
    for (size_t i = 0; i < ndim - 1; i++) outer *= dims[i];

    OrtValue *out_val = NULL;
    ort->KernelContext_GetOutput(ctx, 0, dims, ndim, &out_val);
    int32_t *y = NULL;
    ort->GetTensorMutableData(out_val, (void **)&y);

    const int64_t INTMAX = (int64_t)0x7FFFFFFF;

    for (int64_t b = 0; b < outer; b++) {
        const int8_t *row_in  = x + b * C;
        int32_t      *row_out = y + b * C;

        int64_t sum = 0;
        for (int64_t c = 0; c < C; c++) {
            sum += (int64_t)(int32_t)row_in[c];
        }
        int64_t mean_int64 = round_divide_i64(sum, C);

        uint64_t var = 0;
        for (int64_t c = 0; c < C; c++) {
            int64_t centered = (int64_t)(int32_t)row_in[c] - mean_int64;
            int64_t centered64 = (int64_t)centered;
            var += (uint64_t)(centered64 * centered64);
        }

        int32_t std_int = tvm_layernorm_std(var);
        int64_t norm_scale = INTMAX / (int64_t)std_int;

        for (int64_t c = 0; c < C; c++) {
            int64_t xc = (int64_t)(int32_t)row_in[c] - mean_int64;
            int64_t v = (norm_scale * xc) >> 1;
            row_out[c] = (int32_t)(v + (int64_t)bias[c]);
        }
    }
}

typedef struct {
    const OrtApi *ort;
} QLayernormInt16Kernel;

static void *ORT_API_CALL QLayernormInt16_CreateKernel(
        const OrtCustomOp *op, const OrtApi *api, const OrtKernelInfo *info)
{
    (void)op; (void)info;
    QLayernormInt16Kernel *k = (QLayernormInt16Kernel *)malloc(sizeof(QLayernormInt16Kernel));
    k->ort = api;
    return k;
}

typedef struct {
    const OrtApi *ort;
} QLayernormInt32Kernel;

static void *ORT_API_CALL QLayernormInt32_CreateKernel(
        const OrtCustomOp *op, const OrtApi *api, const OrtKernelInfo *info)
{
    (void)op; (void)info;
    QLayernormInt32Kernel *k = (QLayernormInt32Kernel *)malloc(sizeof(QLayernormInt32Kernel));
    k->ort = api;
    return k;
}

static void ORT_API_CALL QLayernormInt32_Destroy(void *op_kernel) { free(op_kernel); }

static void ORT_API_CALL QLayernormInt32_Compute(void *op_kernel, OrtKernelContext *ctx)
{
    QLayernormInt32Kernel *k = (QLayernormInt32Kernel *)op_kernel;
    const OrtApi *ort = k->ort;

    const OrtValue *x_val = NULL;
    ort->KernelContext_GetInput(ctx, 0, &x_val);

    OrtTensorTypeAndShapeInfo *shape_info = NULL;
    ort->GetTensorTypeAndShape(x_val, &shape_info);
    size_t ndim = 0;
    ort->GetDimensionsCount(shape_info, &ndim);
    int64_t dims[8] = {0};
    ort->GetDimensions(shape_info, dims, ndim);
    ort->ReleaseTensorTypeAndShapeInfo(shape_info);

    const int32_t *x = NULL;
    ort->GetTensorMutableData((OrtValue *)(uintptr_t)x_val, (void **)&x);

    const OrtValue *bias_val = NULL;
    ort->KernelContext_GetInput(ctx, 1, &bias_val);
    const int32_t *bias = NULL;
    ort->GetTensorMutableData((OrtValue *)(uintptr_t)bias_val, (void **)&bias);

    int64_t C     = dims[ndim - 1];
    int64_t outer = 1;
    for (size_t i = 0; i < ndim - 1; i++) outer *= dims[i];

    OrtValue *out_val = NULL;
    ort->KernelContext_GetOutput(ctx, 0, dims, ndim, &out_val);
    int32_t *y = NULL;
    ort->GetTensorMutableData(out_val, (void **)&y);

    const int64_t INTMAX = (int64_t)0x7FFFFFFF;

    for (int64_t b = 0; b < outer; b++) {
        const int32_t *row_in  = x + b * C;
        int32_t       *row_out = y + b * C;

        int64_t sum = 0;
        for (int64_t c = 0; c < C; c++) {
            sum += (int64_t)row_in[c];
        }
        int64_t mean_int64 = round_divide_i64(sum, C);

        uint64_t var = 0;
        for (int64_t c = 0; c < C; c++) {
            int64_t centered = (int64_t)row_in[c] - mean_int64;
            var += (uint64_t)(centered * centered);
        }

        int32_t std_int = tvm_layernorm_std(var);
        int64_t norm_scale = INTMAX / (int64_t)std_int;

        for (int64_t c = 0; c < C; c++) {
            int64_t xc = (int64_t)row_in[c] - mean_int64;
            int64_t v = (norm_scale * xc) >> 1;
            row_out[c] = (int32_t)(v + (int64_t)bias[c]);
        }
    }
}

static void ORT_API_CALL QLayernormInt16_Destroy(void *op_kernel) { free(op_kernel); }

static void ORT_API_CALL QLayernormInt16_Compute(void *op_kernel, OrtKernelContext *ctx)
{
    QLayernormInt16Kernel *k = (QLayernormInt16Kernel *)op_kernel;
    const OrtApi *ort = k->ort;

    const OrtValue *x_val = NULL;
    ort->KernelContext_GetInput(ctx, 0, &x_val);

    OrtTensorTypeAndShapeInfo *shape_info = NULL;
    ort->GetTensorTypeAndShape(x_val, &shape_info);
    size_t ndim = 0;
    ort->GetDimensionsCount(shape_info, &ndim);
    int64_t dims[8] = {0};
    ort->GetDimensions(shape_info, dims, ndim);
    ort->ReleaseTensorTypeAndShapeInfo(shape_info);

    const int16_t *x = NULL;
    ort->GetTensorMutableData((OrtValue *)(uintptr_t)x_val, (void **)&x);

    const OrtValue *bias_val = NULL;
    ort->KernelContext_GetInput(ctx, 1, &bias_val);
    const int32_t *bias = NULL;
    ort->GetTensorMutableData((OrtValue *)(uintptr_t)bias_val, (void **)&bias);

    int64_t C     = dims[ndim - 1];
    int64_t outer = 1;
    for (size_t i = 0; i < ndim - 1; i++) outer *= dims[i];

    OrtValue *out_val = NULL;
    ort->KernelContext_GetOutput(ctx, 0, dims, ndim, &out_val);
    int32_t *y = NULL;
    ort->GetTensorMutableData(out_val, (void **)&y);

    const int64_t INTMAX = (int64_t)0x7FFFFFFF;

    for (int64_t b = 0; b < outer; b++) {
        const int16_t *row_in  = x + b * C;
        int32_t       *row_out = y + b * C;

        int64_t sum = 0;
        for (int64_t c = 0; c < C; c++) {
            sum += (int64_t)(int32_t)row_in[c];
        }
        int64_t mean_int64 = round_divide_i64(sum, C);

        uint64_t var = 0;
        for (int64_t c = 0; c < C; c++) {
            int64_t centered = (int64_t)(int32_t)row_in[c] - mean_int64;
            int64_t centered64 = (int64_t)centered;
            var += (uint64_t)(centered64 * centered64);
        }

        int32_t std_int = tvm_layernorm_std(var);
        int64_t norm_scale = INTMAX / (int64_t)std_int;

        for (int64_t c = 0; c < C; c++) {
            int64_t xc = (int64_t)(int32_t)row_in[c] - mean_int64;
            int64_t v = (norm_scale * xc) >> 1;
            row_out[c] = (int32_t)(v + (int64_t)bias[c]);
        }
    }
}

typedef struct {
    const OrtApi *ort;
} QLayernormI64Kernel;

static void *ORT_API_CALL QLayernormI64_CreateKernel(
        const OrtCustomOp *op, const OrtApi *api, const OrtKernelInfo *info)
{
    (void)op; (void)info;
    QLayernormI64Kernel *k = (QLayernormI64Kernel *)malloc(sizeof(QLayernormI64Kernel));
    k->ort = api;
    return k;
}

static void ORT_API_CALL QLayernormI64_Destroy(void *op_kernel) { free(op_kernel); }

static void ORT_API_CALL QLayernormI64_Compute(void *op_kernel, OrtKernelContext *ctx)
{
    QLayernormI64Kernel *k = (QLayernormI64Kernel *)op_kernel;
    const OrtApi *ort = k->ort;

    const OrtValue *x_val = NULL;
    ort->KernelContext_GetInput(ctx, 0, &x_val);

    OrtTensorTypeAndShapeInfo *shape_info = NULL;
    ort->GetTensorTypeAndShape(x_val, &shape_info);
    size_t ndim = 0;
    ort->GetDimensionsCount(shape_info, &ndim);
    int64_t dims[8] = {0};
    ort->GetDimensions(shape_info, dims, ndim);
    ort->ReleaseTensorTypeAndShapeInfo(shape_info);

    const int8_t *x = NULL;
    ort->GetTensorMutableData((OrtValue *)(uintptr_t)x_val, (void **)&x);

    const OrtValue *bias_val = NULL;
    ort->KernelContext_GetInput(ctx, 1, &bias_val);
    const int64_t *bias = NULL;
    ort->GetTensorMutableData((OrtValue *)(uintptr_t)bias_val, (void **)&bias);

    int64_t C     = dims[ndim - 1];
    int64_t outer = 1;
    for (size_t i = 0; i < ndim - 1; i++) outer *= dims[i];

    OrtValue *out_val = NULL;
    ort->KernelContext_GetOutput(ctx, 0, dims, ndim, &out_val);
    int64_t *y = NULL;
    ort->GetTensorMutableData(out_val, (void **)&y);

    const int64_t INTMAX = (int64_t)0x7FFFFFFF;

    for (int64_t b = 0; b < outer; b++) {
        const int8_t *row_in  = x + b * C;
        int64_t      *row_out = y + b * C;

        int64_t sum = 0;
        for (int64_t c = 0; c < C; c++) {
            sum += (int64_t)(int32_t)row_in[c];
        }
        int64_t mean_int64 = round_divide_i64(sum, C);

        uint64_t var = 0;
        for (int64_t c = 0; c < C; c++) {
            int64_t centered = (int64_t)(int32_t)row_in[c] - mean_int64;
            var += (uint64_t)(centered * centered);
        }

        int32_t std_int = tvm_layernorm_std(var);
        int64_t norm_scale = INTMAX / (int64_t)std_int;

        for (int64_t c = 0; c < C; c++) {
            int64_t xc = (int64_t)(int32_t)row_in[c] - mean_int64;
            int64_t v = (norm_scale * xc) >> 1;
            row_out[c] = v + bias[c];
        }
    }
}

typedef struct {
    const OrtApi *ort;
} QLayernormInt16I64Kernel;

static void *ORT_API_CALL QLayernormInt16I64_CreateKernel(
        const OrtCustomOp *op, const OrtApi *api, const OrtKernelInfo *info)
{
    (void)op; (void)info;
    QLayernormInt16I64Kernel *k = (QLayernormInt16I64Kernel *)malloc(sizeof(QLayernormInt16I64Kernel));
    k->ort = api;
    return k;
}

static void ORT_API_CALL QLayernormInt16I64_Destroy(void *op_kernel) { free(op_kernel); }

static void ORT_API_CALL QLayernormInt16I64_Compute(void *op_kernel, OrtKernelContext *ctx)
{
    QLayernormInt16I64Kernel *k = (QLayernormInt16I64Kernel *)op_kernel;
    const OrtApi *ort = k->ort;

    const OrtValue *x_val = NULL;
    ort->KernelContext_GetInput(ctx, 0, &x_val);

    OrtTensorTypeAndShapeInfo *shape_info = NULL;
    ort->GetTensorTypeAndShape(x_val, &shape_info);
    size_t ndim = 0;
    ort->GetDimensionsCount(shape_info, &ndim);
    int64_t dims[8] = {0};
    ort->GetDimensions(shape_info, dims, ndim);
    ort->ReleaseTensorTypeAndShapeInfo(shape_info);

    const int16_t *x = NULL;
    ort->GetTensorMutableData((OrtValue *)(uintptr_t)x_val, (void **)&x);

    const OrtValue *bias_val = NULL;
    ort->KernelContext_GetInput(ctx, 1, &bias_val);
    const int64_t *bias = NULL;
    ort->GetTensorMutableData((OrtValue *)(uintptr_t)bias_val, (void **)&bias);

    int64_t C     = dims[ndim - 1];
    int64_t outer = 1;
    for (size_t i = 0; i < ndim - 1; i++) outer *= dims[i];

    OrtValue *out_val = NULL;
    ort->KernelContext_GetOutput(ctx, 0, dims, ndim, &out_val);
    int64_t *y = NULL;
    ort->GetTensorMutableData(out_val, (void **)&y);

    const int64_t INTMAX = (int64_t)0x7FFFFFFF;

    for (int64_t b = 0; b < outer; b++) {
        const int16_t *row_in  = x + b * C;
        int64_t       *row_out = y + b * C;

        int64_t sum = 0;
        for (int64_t c = 0; c < C; c++) {
            sum += (int64_t)(int32_t)row_in[c];
        }
        int64_t mean_int64 = round_divide_i64(sum, C);

        uint64_t var = 0;
        for (int64_t c = 0; c < C; c++) {
            int64_t centered = (int64_t)(int32_t)row_in[c] - mean_int64;
            var += (uint64_t)(centered * centered);
        }

        int32_t std_int = tvm_layernorm_std(var);
        int64_t norm_scale = INTMAX / (int64_t)std_int;

        for (int64_t c = 0; c < C; c++) {
            int64_t xc = (int64_t)(int32_t)row_in[c] - mean_int64;
            int64_t v = (norm_scale * xc) >> 1;
            row_out[c] = v + bias[c];
        }
    }
}

typedef struct {
    const OrtApi *ort;
} QLayernormInt32I64Kernel;

static void *ORT_API_CALL QLayernormInt32I64_CreateKernel(
        const OrtCustomOp *op, const OrtApi *api, const OrtKernelInfo *info)
{
    (void)op; (void)info;
    QLayernormInt32I64Kernel *k = (QLayernormInt32I64Kernel *)malloc(sizeof(QLayernormInt32I64Kernel));
    k->ort = api;
    return k;
}

static void ORT_API_CALL QLayernormInt32I64_Destroy(void *op_kernel) { free(op_kernel); }

static void ORT_API_CALL QLayernormInt32I64_Compute(void *op_kernel, OrtKernelContext *ctx)
{
    QLayernormInt32I64Kernel *k = (QLayernormInt32I64Kernel *)op_kernel;
    const OrtApi *ort = k->ort;

    const OrtValue *x_val = NULL;
    ort->KernelContext_GetInput(ctx, 0, &x_val);

    OrtTensorTypeAndShapeInfo *shape_info = NULL;
    ort->GetTensorTypeAndShape(x_val, &shape_info);
    size_t ndim = 0;
    ort->GetDimensionsCount(shape_info, &ndim);
    int64_t dims[8] = {0};
    ort->GetDimensions(shape_info, dims, ndim);
    ort->ReleaseTensorTypeAndShapeInfo(shape_info);

    const int32_t *x = NULL;
    ort->GetTensorMutableData((OrtValue *)(uintptr_t)x_val, (void **)&x);

    const OrtValue *bias_val = NULL;
    ort->KernelContext_GetInput(ctx, 1, &bias_val);
    const int64_t *bias = NULL;
    ort->GetTensorMutableData((OrtValue *)(uintptr_t)bias_val, (void **)&bias);

    int64_t C     = dims[ndim - 1];
    int64_t outer = 1;
    for (size_t i = 0; i < ndim - 1; i++) outer *= dims[i];

    OrtValue *out_val = NULL;
    ort->KernelContext_GetOutput(ctx, 0, dims, ndim, &out_val);
    int64_t *y = NULL;
    ort->GetTensorMutableData(out_val, (void **)&y);

    const int64_t INTMAX = (int64_t)0x7FFFFFFF;

    for (int64_t b = 0; b < outer; b++) {
        const int32_t *row_in  = x + b * C;
        int64_t       *row_out = y + b * C;

        int64_t sum = 0;
        for (int64_t c = 0; c < C; c++) {
            sum += (int64_t)row_in[c];
        }
        int64_t mean_int64 = round_divide_i64(sum, C);

        uint64_t var = 0;
        for (int64_t c = 0; c < C; c++) {
            int64_t centered = (int64_t)row_in[c] - mean_int64;
            var += (uint64_t)(centered * centered);
        }

        int32_t std_int = tvm_layernorm_std(var);
        int64_t norm_scale = INTMAX / (int64_t)std_int;

        for (int64_t c = 0; c < C; c++) {
            int64_t xc = (int64_t)row_in[c] - mean_int64;
            int64_t v = (norm_scale * xc) >> 1;
            row_out[c] = v + bias[c];
        }
    }
}

/* QLayernorm: input 0 = int8, input 1 = int32 */
static ONNXTensorElementDataType ORT_API_CALL QLayernorm_GetInputType(
        const OrtCustomOp *op, size_t idx)
{
    (void)op;
    return idx == 0 ? ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8
                    : ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32;
}

static const char *ORT_API_CALL QLayernorm_GetName(const OrtCustomOp *op)
{ (void)op; return "QLayernorm"; }

static const char *ORT_API_CALL QLayernorm_GetExecutionProviderType(const OrtCustomOp *op)
{ (void)op; return "CPUExecutionProvider"; }

static size_t ORT_API_CALL QLayernorm_GetInputTypeCount(const OrtCustomOp *op)
{ (void)op; return 2; }

static ONNXTensorElementDataType ORT_API_CALL QLayernorm_GetOutputType(
        const OrtCustomOp *op, size_t idx)
{ (void)op; (void)idx; return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32; }

static size_t ORT_API_CALL QLayernorm_GetOutputTypeCount(const OrtCustomOp *op)
{ (void)op; return 1; }

static OrtCustomOpInputOutputCharacteristic ORT_API_CALL
QLayernorm_GetInputCharacteristic(const OrtCustomOp *op, size_t idx)
{ (void)op; (void)idx; return INPUT_OUTPUT_REQUIRED; }

static OrtCustomOpInputOutputCharacteristic ORT_API_CALL
QLayernorm_GetOutputCharacteristic(const OrtCustomOp *op, size_t idx)
{ (void)op; (void)idx; return INPUT_OUTPUT_REQUIRED; }

static OrtCustomOp g_qlayernorm_op = {
    .version                    = ORT_API_VERSION,
    .CreateKernel               = QLayernorm_CreateKernel,
    .GetName                    = QLayernorm_GetName,
    .GetExecutionProviderType   = QLayernorm_GetExecutionProviderType,
    .GetInputType               = QLayernorm_GetInputType,
    .GetInputTypeCount          = QLayernorm_GetInputTypeCount,
    .GetOutputType              = QLayernorm_GetOutputType,
    .GetOutputTypeCount         = QLayernorm_GetOutputTypeCount,
    .KernelCompute              = QLayernorm_Compute,
    .KernelDestroy              = QLayernorm_Destroy,
    .GetInputCharacteristic     = QLayernorm_GetInputCharacteristic,
    .GetOutputCharacteristic    = QLayernorm_GetOutputCharacteristic,
};

static ONNXTensorElementDataType ORT_API_CALL QLayernormInt16_GetInputType(
        const OrtCustomOp *op, size_t idx)
{
    (void)op;
    return idx == 0 ? ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16
                    : ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32;
}

static const char *ORT_API_CALL QLayernormInt16_GetName(const OrtCustomOp *op)
{ (void)op; return "QLayernormInt16"; }

static const char *ORT_API_CALL QLayernormInt16_GetExecutionProviderType(const OrtCustomOp *op)
{ (void)op; return "CPUExecutionProvider"; }

static size_t ORT_API_CALL QLayernormInt16_GetInputTypeCount(const OrtCustomOp *op)
{ (void)op; return 2; }

static ONNXTensorElementDataType ORT_API_CALL QLayernormInt16_GetOutputType(
        const OrtCustomOp *op, size_t idx)
{ (void)op; (void)idx; return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32; }

static size_t ORT_API_CALL QLayernormInt16_GetOutputTypeCount(const OrtCustomOp *op)
{ (void)op; return 1; }

static OrtCustomOpInputOutputCharacteristic ORT_API_CALL
QLayernormInt16_GetInputCharacteristic(const OrtCustomOp *op, size_t idx)
{ (void)op; (void)idx; return INPUT_OUTPUT_REQUIRED; }

static OrtCustomOpInputOutputCharacteristic ORT_API_CALL
QLayernormInt16_GetOutputCharacteristic(const OrtCustomOp *op, size_t idx)
{ (void)op; (void)idx; return INPUT_OUTPUT_REQUIRED; }

static OrtCustomOp g_qlayernorm_int16_op = {
    .version                    = ORT_API_VERSION,
    .CreateKernel               = QLayernormInt16_CreateKernel,
    .GetName                    = QLayernormInt16_GetName,
    .GetExecutionProviderType   = QLayernormInt16_GetExecutionProviderType,
    .GetInputType               = QLayernormInt16_GetInputType,
    .GetInputTypeCount          = QLayernormInt16_GetInputTypeCount,
    .GetOutputType              = QLayernormInt16_GetOutputType,
    .GetOutputTypeCount         = QLayernormInt16_GetOutputTypeCount,
    .KernelCompute              = QLayernormInt16_Compute,
    .KernelDestroy              = QLayernormInt16_Destroy,
    .GetInputCharacteristic     = QLayernormInt16_GetInputCharacteristic,
    .GetOutputCharacteristic    = QLayernormInt16_GetOutputCharacteristic,
};

static ONNXTensorElementDataType ORT_API_CALL QLayernormInt32_GetInputType(
        const OrtCustomOp *op, size_t idx)
{
    (void)op;
    return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32;
}

static const char *ORT_API_CALL QLayernormInt32_GetName(const OrtCustomOp *op)
{ (void)op; return "QLayernormInt32"; }

static const char *ORT_API_CALL QLayernormInt32_GetExecutionProviderType(const OrtCustomOp *op)
{ (void)op; return "CPUExecutionProvider"; }

static size_t ORT_API_CALL QLayernormInt32_GetInputTypeCount(const OrtCustomOp *op)
{ (void)op; return 2; }

static ONNXTensorElementDataType ORT_API_CALL QLayernormInt32_GetOutputType(
        const OrtCustomOp *op, size_t idx)
{ (void)op; (void)idx; return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32; }

static size_t ORT_API_CALL QLayernormInt32_GetOutputTypeCount(const OrtCustomOp *op)
{ (void)op; return 1; }

static OrtCustomOpInputOutputCharacteristic ORT_API_CALL
QLayernormInt32_GetInputCharacteristic(const OrtCustomOp *op, size_t idx)
{ (void)op; (void)idx; return INPUT_OUTPUT_REQUIRED; }

static OrtCustomOpInputOutputCharacteristic ORT_API_CALL
QLayernormInt32_GetOutputCharacteristic(const OrtCustomOp *op, size_t idx)
{ (void)op; (void)idx; return INPUT_OUTPUT_REQUIRED; }

static OrtCustomOp g_qlayernorm_int32_op = {
    .version                    = ORT_API_VERSION,
    .CreateKernel               = QLayernormInt32_CreateKernel,
    .GetName                    = QLayernormInt32_GetName,
    .GetExecutionProviderType   = QLayernormInt32_GetExecutionProviderType,
    .GetInputType               = QLayernormInt32_GetInputType,
    .GetInputTypeCount          = QLayernormInt32_GetInputTypeCount,
    .GetOutputType              = QLayernormInt32_GetOutputType,
    .GetOutputTypeCount         = QLayernormInt32_GetOutputTypeCount,
    .KernelCompute              = QLayernormInt32_Compute,
    .KernelDestroy              = QLayernormInt32_Destroy,
    .GetInputCharacteristic     = QLayernormInt32_GetInputCharacteristic,
    .GetOutputCharacteristic    = QLayernormInt32_GetOutputCharacteristic,
};

static ONNXTensorElementDataType ORT_API_CALL QLayernormI64_GetInputType(
        const OrtCustomOp *op, size_t idx)
{
    (void)op;
    return idx == 0 ? ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8
                    : ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64;
}

static const char *ORT_API_CALL QLayernormI64_GetName(const OrtCustomOp *op)
{ (void)op; return "QLayernormI64"; }

static const char *ORT_API_CALL QLayernormI64_GetExecutionProviderType(const OrtCustomOp *op)
{ (void)op; return "CPUExecutionProvider"; }

static size_t ORT_API_CALL QLayernormI64_GetInputTypeCount(const OrtCustomOp *op)
{ (void)op; return 2; }

static ONNXTensorElementDataType ORT_API_CALL QLayernormI64_GetOutputType(
        const OrtCustomOp *op, size_t idx)
{ (void)op; (void)idx; return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64; }

static size_t ORT_API_CALL QLayernormI64_GetOutputTypeCount(const OrtCustomOp *op)
{ (void)op; return 1; }

static OrtCustomOpInputOutputCharacteristic ORT_API_CALL
QLayernormI64_GetInputCharacteristic(const OrtCustomOp *op, size_t idx)
{ (void)op; (void)idx; return INPUT_OUTPUT_REQUIRED; }

static OrtCustomOpInputOutputCharacteristic ORT_API_CALL
QLayernormI64_GetOutputCharacteristic(const OrtCustomOp *op, size_t idx)
{ (void)op; (void)idx; return INPUT_OUTPUT_REQUIRED; }

static OrtCustomOp g_qlayernorm_i64_op = {
    .version                    = ORT_API_VERSION,
    .CreateKernel               = QLayernormI64_CreateKernel,
    .GetName                    = QLayernormI64_GetName,
    .GetExecutionProviderType   = QLayernormI64_GetExecutionProviderType,
    .GetInputType               = QLayernormI64_GetInputType,
    .GetInputTypeCount          = QLayernormI64_GetInputTypeCount,
    .GetOutputType              = QLayernormI64_GetOutputType,
    .GetOutputTypeCount         = QLayernormI64_GetOutputTypeCount,
    .KernelCompute              = QLayernormI64_Compute,
    .KernelDestroy              = QLayernormI64_Destroy,
    .GetInputCharacteristic     = QLayernormI64_GetInputCharacteristic,
    .GetOutputCharacteristic    = QLayernormI64_GetOutputCharacteristic,
};

static ONNXTensorElementDataType ORT_API_CALL QLayernormInt16I64_GetInputType(
        const OrtCustomOp *op, size_t idx)
{
    (void)op;
    return idx == 0 ? ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16
                    : ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64;
}

static const char *ORT_API_CALL QLayernormInt16I64_GetName(const OrtCustomOp *op)
{ (void)op; return "QLayernormInt16I64"; }

static const char *ORT_API_CALL QLayernormInt16I64_GetExecutionProviderType(const OrtCustomOp *op)
{ (void)op; return "CPUExecutionProvider"; }

static size_t ORT_API_CALL QLayernormInt16I64_GetInputTypeCount(const OrtCustomOp *op)
{ (void)op; return 2; }

static ONNXTensorElementDataType ORT_API_CALL QLayernormInt16I64_GetOutputType(
        const OrtCustomOp *op, size_t idx)
{ (void)op; (void)idx; return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64; }

static size_t ORT_API_CALL QLayernormInt16I64_GetOutputTypeCount(const OrtCustomOp *op)
{ (void)op; return 1; }

static OrtCustomOpInputOutputCharacteristic ORT_API_CALL
QLayernormInt16I64_GetInputCharacteristic(const OrtCustomOp *op, size_t idx)
{ (void)op; (void)idx; return INPUT_OUTPUT_REQUIRED; }

static OrtCustomOpInputOutputCharacteristic ORT_API_CALL
QLayernormInt16I64_GetOutputCharacteristic(const OrtCustomOp *op, size_t idx)
{ (void)op; (void)idx; return INPUT_OUTPUT_REQUIRED; }

static OrtCustomOp g_qlayernorm_int16_i64_op = {
    .version                    = ORT_API_VERSION,
    .CreateKernel               = QLayernormInt16I64_CreateKernel,
    .GetName                    = QLayernormInt16I64_GetName,
    .GetExecutionProviderType   = QLayernormInt16I64_GetExecutionProviderType,
    .GetInputType               = QLayernormInt16I64_GetInputType,
    .GetInputTypeCount          = QLayernormInt16I64_GetInputTypeCount,
    .GetOutputType              = QLayernormInt16I64_GetOutputType,
    .GetOutputTypeCount         = QLayernormInt16I64_GetOutputTypeCount,
    .KernelCompute              = QLayernormInt16I64_Compute,
    .KernelDestroy              = QLayernormInt16I64_Destroy,
    .GetInputCharacteristic     = QLayernormInt16I64_GetInputCharacteristic,
    .GetOutputCharacteristic    = QLayernormInt16I64_GetOutputCharacteristic,
};

static ONNXTensorElementDataType ORT_API_CALL QLayernormInt32I64_GetInputType(
        const OrtCustomOp *op, size_t idx)
{
    (void)op;
    return idx == 0 ? ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32
                    : ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64;
}

static const char *ORT_API_CALL QLayernormInt32I64_GetName(const OrtCustomOp *op)
{ (void)op; return "QLayernormInt32I64"; }

static const char *ORT_API_CALL QLayernormInt32I64_GetExecutionProviderType(const OrtCustomOp *op)
{ (void)op; return "CPUExecutionProvider"; }

static size_t ORT_API_CALL QLayernormInt32I64_GetInputTypeCount(const OrtCustomOp *op)
{ (void)op; return 2; }

static ONNXTensorElementDataType ORT_API_CALL QLayernormInt32I64_GetOutputType(
        const OrtCustomOp *op, size_t idx)
{ (void)op; (void)idx; return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64; }

static size_t ORT_API_CALL QLayernormInt32I64_GetOutputTypeCount(const OrtCustomOp *op)
{ (void)op; return 1; }

static OrtCustomOpInputOutputCharacteristic ORT_API_CALL
QLayernormInt32I64_GetInputCharacteristic(const OrtCustomOp *op, size_t idx)
{ (void)op; (void)idx; return INPUT_OUTPUT_REQUIRED; }

static OrtCustomOpInputOutputCharacteristic ORT_API_CALL
QLayernormInt32I64_GetOutputCharacteristic(const OrtCustomOp *op, size_t idx)
{ (void)op; (void)idx; return INPUT_OUTPUT_REQUIRED; }

static OrtCustomOp g_qlayernorm_int32_i64_op = {
    .version                    = ORT_API_VERSION,
    .CreateKernel               = QLayernormInt32I64_CreateKernel,
    .GetName                    = QLayernormInt32I64_GetName,
    .GetExecutionProviderType   = QLayernormInt32I64_GetExecutionProviderType,
    .GetInputType               = QLayernormInt32I64_GetInputType,
    .GetInputTypeCount          = QLayernormInt32I64_GetInputTypeCount,
    .GetOutputType              = QLayernormInt32I64_GetOutputType,
    .GetOutputTypeCount         = QLayernormInt32I64_GetOutputTypeCount,
    .KernelCompute              = QLayernormInt32I64_Compute,
    .KernelDestroy              = QLayernormInt32I64_Destroy,
    .GetInputCharacteristic     = QLayernormInt32I64_GetInputCharacteristic,
    .GetOutputCharacteristic    = QLayernormInt32I64_GetOutputCharacteristic,
};

/* ══════════════════════════════════════════════════════════════════════════
 *  ivit.FQQKMatMul — flexi fq_deit.QKMatMul.forward_i8
 *  Inputs: int8 q, int8 k, scale, zp, qk_scale
 *  Output: float32 scores [B,H,N,N]
 * ══════════════════════════════════════════════════════════════════════════ */

static void fq_qk_matmul_core(
    const int8_t *q,
    const int8_t *k,
    float *out,
    int64_t B,
    int64_t H,
    int64_t N,
    int64_t D,
    float scale,
    float zp,
    float qk_scale)
{
    const float c = 128.0f - zp;
    const float mul = scale * scale * qk_scale;
    const float bias = c * c * (float)D;

    for (int64_t b = 0; b < B; b++) {
        for (int64_t h = 0; h < H; h++) {
            const int64_t bh = b * H + h;
            for (int64_t i = 0; i < N; i++) {
                int32_t qsum = 0;
                for (int64_t d = 0; d < D; d++) {
                    qsum += (int32_t)q[((bh * N) + i) * D + d];
                }
                for (int64_t j = 0; j < N; j++) {
                    int32_t ksum = 0;
                    int32_t acc = 0;
                    for (int64_t d = 0; d < D; d++) {
                        const int32_t qv = (int32_t)q[((bh * N) + i) * D + d];
                        const int32_t kv = (int32_t)k[((bh * N) + j) * D + d];
                        acc += qv * kv;
                        ksum += kv;
                    }
                    float acc_f = (float)acc + c * (float)ksum + c * (float)qsum + bias;
                    out[((bh * N) + i) * N + j] = acc_f * mul;
                }
            }
        }
    }
}

typedef struct {
    const OrtApi *ort;
} FQQKMatMulKernel;

static void *ORT_API_CALL FQQKMatMul_CreateKernel(
    const OrtCustomOp *op, const OrtApi *api, const OrtKernelInfo *info)
{
    (void)op;
    (void)info;
    FQQKMatMulKernel *k = (FQQKMatMulKernel *)malloc(sizeof(FQQKMatMulKernel));
    k->ort = api;
    return k;
}

static void ORT_API_CALL FQQKMatMul_Destroy(void *op_kernel) { free(op_kernel); }

static void ORT_API_CALL FQQKMatMul_Compute(void *op_kernel, OrtKernelContext *ctx)
{
    FQQKMatMulKernel *k = (FQQKMatMulKernel *)op_kernel;
    const OrtApi *ort = k->ort;

    const OrtValue *q_val = NULL;
    ort->KernelContext_GetInput(ctx, 0, &q_val);
    OrtTensorTypeAndShapeInfo *shape_info = NULL;
    ort->GetTensorTypeAndShape(q_val, &shape_info);
    size_t ndim = 0;
    ort->GetDimensionsCount(shape_info, &ndim);
    int64_t dims[8] = {0};
    ort->GetDimensions(shape_info, dims, ndim);
    ort->ReleaseTensorTypeAndShapeInfo(shape_info);

    const int8_t *q = NULL;
    ort->GetTensorMutableData((OrtValue *)(uintptr_t)q_val, (void **)&q);

    const OrtValue *k_val = NULL;
    ort->KernelContext_GetInput(ctx, 1, &k_val);
    const int8_t *kptr = NULL;
    ort->GetTensorMutableData((OrtValue *)(uintptr_t)k_val, (void **)&kptr);

    const OrtValue *scale_val = NULL;
    ort->KernelContext_GetInput(ctx, 2, &scale_val);
    const float *scale_ptr = NULL;
    ort->GetTensorMutableData((OrtValue *)(uintptr_t)scale_val, (void **)&scale_ptr);

    const OrtValue *zp_val = NULL;
    ort->KernelContext_GetInput(ctx, 3, &zp_val);
    const float *zp_ptr = NULL;
    ort->GetTensorMutableData((OrtValue *)(uintptr_t)zp_val, (void **)&zp_ptr);

    const OrtValue *qk_scale_val = NULL;
    ort->KernelContext_GetInput(ctx, 4, &qk_scale_val);
    const float *qk_scale_ptr = NULL;
    ort->GetTensorMutableData((OrtValue *)(uintptr_t)qk_scale_val, (void **)&qk_scale_ptr);

    int64_t B = dims[0];
    int64_t H = dims[1];
    int64_t N = dims[2];
    int64_t D = dims[3];
    int64_t out_dims[4] = {B, H, N, N};

    OrtValue *out_val = NULL;
    ort->KernelContext_GetOutput(ctx, 0, out_dims, 4, &out_val);
    float *out = NULL;
    ort->GetTensorMutableData(out_val, (void **)&out);

    fq_qk_matmul_core(q, kptr, out, B, H, N, D, scale_ptr[0], zp_ptr[0], qk_scale_ptr[0]);
}

static ONNXTensorElementDataType ORT_API_CALL FQQKMatMul_GetInputType(
    const OrtCustomOp *op, size_t idx)
{
    (void)op;
    if (idx == 0 || idx == 1) return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8;
    return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
}

static const char *ORT_API_CALL FQQKMatMul_GetName(const OrtCustomOp *op)
{
    (void)op;
    return "FQQKMatMul";
}

static const char *ORT_API_CALL FQQKMatMul_GetExecutionProviderType(const OrtCustomOp *op)
{
    (void)op;
    return "CPUExecutionProvider";
}

static size_t ORT_API_CALL FQQKMatMul_GetInputTypeCount(const OrtCustomOp *op)
{
    (void)op;
    return 5;
}

static ONNXTensorElementDataType ORT_API_CALL FQQKMatMul_GetOutputType(
    const OrtCustomOp *op, size_t idx)
{
    (void)op;
    (void)idx;
    return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
}

static size_t ORT_API_CALL FQQKMatMul_GetOutputTypeCount(const OrtCustomOp *op)
{
    (void)op;
    return 1;
}

static OrtCustomOpInputOutputCharacteristic ORT_API_CALL FQQKMatMul_GetInputCharacteristic(
    const OrtCustomOp *op, size_t idx)
{
    (void)op;
    (void)idx;
    return INPUT_OUTPUT_REQUIRED;
}

static OrtCustomOpInputOutputCharacteristic ORT_API_CALL FQQKMatMul_GetOutputCharacteristic(
    const OrtCustomOp *op, size_t idx)
{
    (void)op;
    (void)idx;
    return INPUT_OUTPUT_REQUIRED;
}

static OrtCustomOp g_fq_qk_matmul_op = {
    .version                    = ORT_API_VERSION,
    .CreateKernel               = FQQKMatMul_CreateKernel,
    .GetName                    = FQQKMatMul_GetName,
    .GetExecutionProviderType   = FQQKMatMul_GetExecutionProviderType,
    .GetInputType               = FQQKMatMul_GetInputType,
    .GetInputTypeCount          = FQQKMatMul_GetInputTypeCount,
    .GetOutputType              = FQQKMatMul_GetOutputType,
    .GetOutputTypeCount         = FQQKMatMul_GetOutputTypeCount,
    .KernelCompute              = FQQKMatMul_Compute,
    .KernelDestroy              = FQQKMatMul_Destroy,
    .GetInputCharacteristic     = FQQKMatMul_GetInputCharacteristic,
    .GetOutputCharacteristic    = FQQKMatMul_GetOutputCharacteristic,
};

/* ══════════════════════════════════════════════════════════════════════════
 *  ivit.FQLISSoftmax — flexi fq_deit.LISSoftmax
 *  Inputs: float32 x, attn_scale, attn_zp
 *  Output: float32 log-domain softmax (same shape as x)
 * ══════════════════════════════════════════════════════════════════════════ */

static int64_t fq_safe_left_shift_i64(int64_t value, int32_t shift)
{
    if (shift < 0) return 0;
    if (shift >= 62) return 0;
    return value << shift;
}

static void fq_lis_softmax_row(const float *x, float *out, int64_t N, float s, float attn_zp)
{
    const int32_t n = 30;
    int32_t x0_int = (int32_t)floor(-0.6931 / (double)s);
    int32_t b_int = (int32_t)floor((0.96963238 / 0.35815147) / (double)s);
    int32_t c_int = (int32_t)floor((1.0 / 0.35815147) / ((double)s * (double)s));
    int32_t nx0 = n * x0_int;

    int32_t *x_int = (int32_t *)malloc((size_t)N * sizeof(int32_t));
    int64_t *exp_int = (int64_t *)malloc((size_t)N * sizeof(int64_t));
    if (!x_int || !exp_int) {
        free(x_int);
        free(exp_int);
        return;
    }

    int32_t max_v = INT32_MIN;
    for (int64_t i = 0; i < N; i++) {
        double rounded = round((double)x[i] / (double)s);
        double cap = 255.0 - (double)attn_zp;
        if (rounded > cap) rounded = cap;
        x_int[i] = (int32_t)rounded;
        if (x_int[i] > max_v) max_v = x_int[i];
    }
    for (int64_t i = 0; i < N; i++) {
        x_int[i] -= max_v;
    }

    for (int64_t i = 0; i < N; i++) {
        int32_t xi = x_int[i];
        if (xi < nx0) xi = nx0;
        int32_t qd = (int32_t)floor((double)xi / (double)x0_int);
        int32_t r = xi - x0_int * qd;
        int64_t z = (int64_t)r * (int64_t)(r + b_int) + (int64_t)c_int;
        exp_int[i] = fq_safe_left_shift_i64(z, n - qd);
    }

    int64_t exp_sum = 0;
    for (int64_t j = 0; j < N; j++) {
        exp_sum += exp_int[j];
    }

    for (int64_t i = 0; i < N; i++) {
        double smx = round((double)exp_sum / (double)exp_int[i]);
        double lf = floor(log2(smx));
        double pow_lf = 0.0;
        if (lf >= 0.0 && lf < 62.0) {
            pow_lf = (double)(1LL << (int64_t)lf);
        }
        double indicator = ((smx - pow_lf) >= pow_lf * 0.5) ? 1.0 : 0.0;
        out[i] = (float)(lf + indicator);
    }

    free(x_int);
    free(exp_int);
}

typedef struct {
    const OrtApi *ort;
} FQLISSoftmaxKernel;

static void *ORT_API_CALL FQLISSoftmax_CreateKernel(
    const OrtCustomOp *op, const OrtApi *api, const OrtKernelInfo *info)
{
    (void)op;
    (void)info;
    FQLISSoftmaxKernel *k = (FQLISSoftmaxKernel *)malloc(sizeof(FQLISSoftmaxKernel));
    k->ort = api;
    return k;
}

static void ORT_API_CALL FQLISSoftmax_Destroy(void *op_kernel) { free(op_kernel); }

static void ORT_API_CALL FQLISSoftmax_Compute(void *op_kernel, OrtKernelContext *ctx)
{
    FQLISSoftmaxKernel *k = (FQLISSoftmaxKernel *)op_kernel;
    const OrtApi *ort = k->ort;

    const OrtValue *x_val = NULL;
    ort->KernelContext_GetInput(ctx, 0, &x_val);
    OrtTensorTypeAndShapeInfo *shape_info = NULL;
    ort->GetTensorTypeAndShape(x_val, &shape_info);
    size_t ndim = 0;
    ort->GetDimensionsCount(shape_info, &ndim);
    int64_t dims[8] = {0};
    ort->GetDimensions(shape_info, dims, ndim);
    ort->ReleaseTensorTypeAndShapeInfo(shape_info);

    const float *x = NULL;
    ort->GetTensorMutableData((OrtValue *)(uintptr_t)x_val, (void **)&x);

    const OrtValue *scale_val = NULL;
    ort->KernelContext_GetInput(ctx, 1, &scale_val);
    const float *scale_ptr = NULL;
    ort->GetTensorMutableData((OrtValue *)(uintptr_t)scale_val, (void **)&scale_ptr);

    const OrtValue *zp_val = NULL;
    ort->KernelContext_GetInput(ctx, 2, &zp_val);
    const float *zp_ptr = NULL;
    ort->GetTensorMutableData((OrtValue *)(uintptr_t)zp_val, (void **)&zp_ptr);

    int64_t N = dims[ndim - 1];
    int64_t outer = 1;
    for (size_t i = 0; i < ndim - 1; i++) outer *= dims[i];

    OrtValue *y_val = NULL;
    ort->KernelContext_GetOutput(ctx, 0, dims, ndim, &y_val);
    float *y = NULL;
    ort->GetTensorMutableData(y_val, (void **)&y);

    for (int64_t b = 0; b < outer; b++) {
        fq_lis_softmax_row(x + b * N, y + b * N, N, scale_ptr[0], zp_ptr[0]);
    }
}

static ONNXTensorElementDataType ORT_API_CALL FQLISSoftmax_GetInputType(
    const OrtCustomOp *op, size_t idx)
{
    (void)op;
    (void)idx;
    return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
}

static const char *ORT_API_CALL FQLISSoftmax_GetName(const OrtCustomOp *op)
{
    (void)op;
    return "FQLISSoftmax";
}

static const char *ORT_API_CALL FQLISSoftmax_GetExecutionProviderType(const OrtCustomOp *op)
{
    (void)op;
    return "CPUExecutionProvider";
}

static size_t ORT_API_CALL FQLISSoftmax_GetInputTypeCount(const OrtCustomOp *op)
{
    (void)op;
    return 3;
}

static ONNXTensorElementDataType ORT_API_CALL FQLISSoftmax_GetOutputType(
    const OrtCustomOp *op, size_t idx)
{
    (void)op;
    (void)idx;
    return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
}

static size_t ORT_API_CALL FQLISSoftmax_GetOutputTypeCount(const OrtCustomOp *op)
{
    (void)op;
    return 1;
}

static OrtCustomOpInputOutputCharacteristic ORT_API_CALL FQLISSoftmax_GetInputCharacteristic(
    const OrtCustomOp *op, size_t idx)
{
    (void)op;
    (void)idx;
    return INPUT_OUTPUT_REQUIRED;
}

static OrtCustomOpInputOutputCharacteristic ORT_API_CALL FQLISSoftmax_GetOutputCharacteristic(
    const OrtCustomOp *op, size_t idx)
{
    (void)op;
    (void)idx;
    return INPUT_OUTPUT_REQUIRED;
}

static OrtCustomOp g_fq_lis_softmax_op = {
    .version                    = ORT_API_VERSION,
    .CreateKernel               = FQLISSoftmax_CreateKernel,
    .GetName                    = FQLISSoftmax_GetName,
    .GetExecutionProviderType   = FQLISSoftmax_GetExecutionProviderType,
    .GetInputType               = FQLISSoftmax_GetInputType,
    .GetInputTypeCount          = FQLISSoftmax_GetInputTypeCount,
    .GetOutputType              = FQLISSoftmax_GetOutputType,
    .GetOutputTypeCount         = FQLISSoftmax_GetOutputTypeCount,
    .KernelCompute              = FQLISSoftmax_Compute,
    .KernelDestroy              = FQLISSoftmax_Destroy,
    .GetInputCharacteristic     = FQLISSoftmax_GetInputCharacteristic,
    .GetOutputCharacteristic    = FQLISSoftmax_GetOutputCharacteristic,
};

/* ══════════════════════════════════════════════════════════════════════════
 *  ivit.FQAttnVMatMul — flexi fq_deit.AttnVMatMul.forward_i8
 *  Inputs: float32 qlog, int8 v, v_scale, v_zp
 *  Output: float32 context
 * ══════════════════════════════════════════════════════════════════════════ */

static void fq_attn_v_matmul_core(
    const float *qlog,
    const int8_t *v,
    float *out,
    int64_t B,
    int64_t H,
    int64_t M,
    int64_t K,
    int64_t D,
    float v_scale,
    float v_zp)
{
    const int32_t Hval = 15;
    const int32_t v_off = (int32_t)lrintf(128.0f - v_zp);
    const float scale = v_scale / (float)(1LL << Hval);

    for (int64_t b = 0; b < B; b++) {
        for (int64_t h = 0; h < H; h++) {
            const int64_t bh = b * H + h;
            for (int64_t m = 0; m < M; m++) {
                for (int64_t d = 0; d < D; d++) {
                    int64_t acc = 0;
                    for (int64_t k = 0; k < K; k++) {
                        const int32_t qi = (int32_t)qlog[(bh * M + m) * K + k];
                        if (qi >= 16) continue;
                        int32_t lsh = Hval - qi;
                        if (lsh < 0) lsh = 0;
                        const int32_t vi = (int32_t)v[(bh * K + k) * D + d] + v_off;
                        acc += (int64_t)vi << lsh;
                    }
                    out[(bh * M + m) * D + d] = (float)acc * scale;
                }
            }
        }
    }
}

typedef struct {
    const OrtApi *ort;
} FQAttnVMatMulKernel;

static void *ORT_API_CALL FQAttnVMatMul_CreateKernel(
    const OrtCustomOp *op, const OrtApi *api, const OrtKernelInfo *info)
{
    (void)op;
    (void)info;
    FQAttnVMatMulKernel *k = (FQAttnVMatMulKernel *)malloc(sizeof(FQAttnVMatMulKernel));
    k->ort = api;
    return k;
}

static void ORT_API_CALL FQAttnVMatMul_Destroy(void *op_kernel) { free(op_kernel); }

static void ORT_API_CALL FQAttnVMatMul_Compute(void *op_kernel, OrtKernelContext *ctx)
{
    FQAttnVMatMulKernel *k = (FQAttnVMatMulKernel *)op_kernel;
    const OrtApi *ort = k->ort;

    const OrtValue *qlog_val = NULL;
    ort->KernelContext_GetInput(ctx, 0, &qlog_val);
    OrtTensorTypeAndShapeInfo *q_shape = NULL;
    ort->GetTensorTypeAndShape(qlog_val, &q_shape);
    size_t ndim = 0;
    ort->GetDimensionsCount(q_shape, &ndim);
    int64_t qdims[8] = {0};
    ort->GetDimensions(q_shape, qdims, ndim);
    ort->ReleaseTensorTypeAndShapeInfo(q_shape);

    const float *qlog = NULL;
    ort->GetTensorMutableData((OrtValue *)(uintptr_t)qlog_val, (void **)&qlog);

    const OrtValue *v_val = NULL;
    ort->KernelContext_GetInput(ctx, 1, &v_val);
    const int8_t *v = NULL;
    ort->GetTensorMutableData((OrtValue *)(uintptr_t)v_val, (void **)&v);

    const OrtValue *v_scale_val = NULL;
    ort->KernelContext_GetInput(ctx, 2, &v_scale_val);
    const float *v_scale_ptr = NULL;
    ort->GetTensorMutableData((OrtValue *)(uintptr_t)v_scale_val, (void **)&v_scale_ptr);

    const OrtValue *v_zp_val = NULL;
    ort->KernelContext_GetInput(ctx, 3, &v_zp_val);
    const float *v_zp_ptr = NULL;
    ort->GetTensorMutableData((OrtValue *)(uintptr_t)v_zp_val, (void **)&v_zp_ptr);

    int64_t B = 1;
    int64_t H = 1;
    int64_t M = 1;
    int64_t K = 1;
    int64_t D = 1;
    if (ndim == 4) {
        B = qdims[0];
        H = qdims[1];
        M = qdims[2];
        K = qdims[3];
        OrtTensorTypeAndShapeInfo *v_shape = NULL;
        ort->GetTensorTypeAndShape(v_val, &v_shape);
        size_t v_ndim = 0;
        ort->GetDimensionsCount(v_shape, &v_ndim);
        int64_t vdims[8] = {0};
        ort->GetDimensions(v_shape, vdims, v_ndim);
        ort->ReleaseTensorTypeAndShapeInfo(v_shape);
        D = vdims[v_ndim - 1];
    } else if (ndim == 5) {
        B = qdims[0];
        H = qdims[1];
        M = qdims[2];
        K = qdims[3];
        D = qdims[4];
    }

    int64_t out_dims[4] = {B, H, M, D};
    OrtValue *out_val = NULL;
    ort->KernelContext_GetOutput(ctx, 0, out_dims, 4, &out_val);
    float *out = NULL;
    ort->GetTensorMutableData(out_val, (void **)&out);

    fq_attn_v_matmul_core(qlog, v, out, B, H, M, K, D, v_scale_ptr[0], v_zp_ptr[0]);
}

static ONNXTensorElementDataType ORT_API_CALL FQAttnVMatMul_GetInputType(
    const OrtCustomOp *op, size_t idx)
{
    (void)op;
    if (idx == 0) return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
    if (idx == 1) return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8;
    return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
}

static const char *ORT_API_CALL FQAttnVMatMul_GetName(const OrtCustomOp *op)
{
    (void)op;
    return "FQAttnVMatMul";
}

static const char *ORT_API_CALL FQAttnVMatMul_GetExecutionProviderType(const OrtCustomOp *op)
{
    (void)op;
    return "CPUExecutionProvider";
}

static size_t ORT_API_CALL FQAttnVMatMul_GetInputTypeCount(const OrtCustomOp *op)
{
    (void)op;
    return 4;
}

static ONNXTensorElementDataType ORT_API_CALL FQAttnVMatMul_GetOutputType(
    const OrtCustomOp *op, size_t idx)
{
    (void)op;
    (void)idx;
    return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
}

static size_t ORT_API_CALL FQAttnVMatMul_GetOutputTypeCount(const OrtCustomOp *op)
{
    (void)op;
    return 1;
}

static OrtCustomOpInputOutputCharacteristic ORT_API_CALL FQAttnVMatMul_GetInputCharacteristic(
    const OrtCustomOp *op, size_t idx)
{
    (void)op;
    (void)idx;
    return INPUT_OUTPUT_REQUIRED;
}

static OrtCustomOpInputOutputCharacteristic ORT_API_CALL FQAttnVMatMul_GetOutputCharacteristic(
    const OrtCustomOp *op, size_t idx)
{
    (void)op;
    (void)idx;
    return INPUT_OUTPUT_REQUIRED;
}

static OrtCustomOp g_fq_attn_v_matmul_op = {
    .version                    = ORT_API_VERSION,
    .CreateKernel               = FQAttnVMatMul_CreateKernel,
    .GetName                    = FQAttnVMatMul_GetName,
    .GetExecutionProviderType   = FQAttnVMatMul_GetExecutionProviderType,
    .GetInputType               = FQAttnVMatMul_GetInputType,
    .GetInputTypeCount          = FQAttnVMatMul_GetInputTypeCount,
    .GetOutputType              = FQAttnVMatMul_GetOutputType,
    .GetOutputTypeCount         = FQAttnVMatMul_GetOutputTypeCount,
    .KernelCompute              = FQAttnVMatMul_Compute,
    .KernelDestroy              = FQAttnVMatMul_Destroy,
    .GetInputCharacteristic     = FQAttnVMatMul_GetInputCharacteristic,
    .GetOutputCharacteristic    = FQAttnVMatMul_GetOutputCharacteristic,
};

/* ══════════════════════════════════════════════════════════════════════════
 *  ivit.FQPTFLayerNorm — flexi fq_deit.PTFLayerNorm (float in/out)
 *  Inputs: float32 x, inv_in[C], ratio[C], out_scale, weight[C], bias[C]
 *  Output: float32 (same shape as x)
 * ══════════════════════════════════════════════════════════════════════════ */

static int32_t float_bits_as_int32(float value)
{
    uint32_t bits;
    memcpy(&bits, &value, sizeof(bits));
    return (int32_t)bits;
}

static float int32_bits_as_float(int32_t bits)
{
    uint32_t u = (uint32_t)bits;
    float value;
    memcpy(&value, &u, sizeof(value));
    return value;
}

static void fq_ptf_layernorm_row(
    const float *x,
    float *out,
    int64_t C,
    const float *inv_in,
    const int32_t *ratio,
    float out_scale,
    const float *weight,
    const float *bias)
{
    int32_t *x_q = (int32_t *)malloc((size_t)C * sizeof(int32_t));
    if (!x_q) return;

    int64_t s = 0;
    int64_t ss = 0;
    for (int64_t c = 0; c < C; c++) {
        int32_t xq = (int32_t)lrint((double)x[c] * (double)inv_in[c]) * ratio[c];
        x_q[c] = xq;
        s += xq;
        ss += (int64_t)xq * (int64_t)xq;
    }

    float var = (float)C * (float)ss - (float)s * (float)s;
    float rstd = 1.0f / sqrtf(var);
    float s_f = (float)s;

    for (int64_t c = 0; c < C; c++) {
        float A = fabsf(((float)C / out_scale) * rstd * weight[c]);
        int32_t bits = float_bits_as_int32(A);
        int32_t eb = (bits >> 23) & 0xFF;
        int32_t Nn = 134 - eb;
        int32_t M = ((bits >> 16) & 0x7F) + 128;
        float pow_nn = int32_bits_as_float((261 - eb) << 23);

        float mean_term = s_f * rstd * weight[c];
        int32_t Bt = (int32_t)lrint((double)((bias[c] - mean_term) / out_scale * pow_nn));

        int32_t wsign = (weight[c] > 0.0f) ? 1 : ((weight[c] < 0.0f) ? -1 : 0);
        int64_t acc = (int64_t)wsign * (int64_t)M * (int64_t)x_q[c] + (int64_t)Bt;
        int64_t half = 0;
        if (Nn > 0) {
            int64_t shift_amt = (int64_t)Nn - 1;
            if (shift_amt >= 0 && shift_amt < 62) {
                half = (int64_t)1 << shift_amt;
            }
        }
        int64_t shifted = 0;
        if (Nn >= 0 && Nn < 62) {
            shifted = (acc + half) >> Nn;
        }
        out[c] = (float)shifted * out_scale;
    }

    free(x_q);
}

typedef struct {
    const OrtApi *ort;
} FQPTFLayerNormKernel;

static void *ORT_API_CALL FQPTFLayerNorm_CreateKernel(
    const OrtCustomOp *op, const OrtApi *api, const OrtKernelInfo *info)
{
    (void)op;
    (void)info;
    FQPTFLayerNormKernel *k = (FQPTFLayerNormKernel *)malloc(sizeof(FQPTFLayerNormKernel));
    k->ort = api;
    return k;
}

static void ORT_API_CALL FQPTFLayerNorm_Destroy(void *op_kernel)
{
    free(op_kernel);
}

static void ORT_API_CALL FQPTFLayerNorm_Compute(void *op_kernel, OrtKernelContext *ctx)
{
    FQPTFLayerNormKernel *k = (FQPTFLayerNormKernel *)op_kernel;
    const OrtApi *ort = k->ort;

    const OrtValue *x_val = NULL;
    ort->KernelContext_GetInput(ctx, 0, &x_val);
    OrtTensorTypeAndShapeInfo *shape_info = NULL;
    ort->GetTensorTypeAndShape(x_val, &shape_info);
    size_t ndim = 0;
    ort->GetDimensionsCount(shape_info, &ndim);
    int64_t dims[8] = {0};
    ort->GetDimensions(shape_info, dims, ndim);
    ort->ReleaseTensorTypeAndShapeInfo(shape_info);

    const float *x = NULL;
    ort->GetTensorMutableData((OrtValue *)(uintptr_t)x_val, (void **)&x);

    const OrtValue *inv_in_val = NULL;
    ort->KernelContext_GetInput(ctx, 1, &inv_in_val);
    const float *inv_in = NULL;
    ort->GetTensorMutableData((OrtValue *)(uintptr_t)inv_in_val, (void **)&inv_in);

    const OrtValue *ratio_val = NULL;
    ort->KernelContext_GetInput(ctx, 2, &ratio_val);
    const int32_t *ratio = NULL;
    ort->GetTensorMutableData((OrtValue *)(uintptr_t)ratio_val, (void **)&ratio);

    const OrtValue *out_scale_val = NULL;
    ort->KernelContext_GetInput(ctx, 3, &out_scale_val);
    const float *out_scale_ptr = NULL;
    ort->GetTensorMutableData((OrtValue *)(uintptr_t)out_scale_val, (void **)&out_scale_ptr);
    float out_scale = out_scale_ptr[0];

    const OrtValue *weight_val = NULL;
    ort->KernelContext_GetInput(ctx, 4, &weight_val);
    const float *weight = NULL;
    ort->GetTensorMutableData((OrtValue *)(uintptr_t)weight_val, (void **)&weight);

    const OrtValue *bias_val = NULL;
    ort->KernelContext_GetInput(ctx, 5, &bias_val);
    const float *bias = NULL;
    ort->GetTensorMutableData((OrtValue *)(uintptr_t)bias_val, (void **)&bias);

    int64_t C = dims[ndim - 1];
    int64_t outer = 1;
    for (size_t i = 0; i < ndim - 1; i++) outer *= dims[i];

    OrtValue *y_val = NULL;
    ort->KernelContext_GetOutput(ctx, 0, dims, ndim, &y_val);
    float *y = NULL;
    ort->GetTensorMutableData(y_val, (void **)&y);

    for (int64_t b = 0; b < outer; b++) {
        fq_ptf_layernorm_row(
            x + b * C,
            y + b * C,
            C,
            inv_in,
            ratio,
            out_scale,
            weight,
            bias);
    }
}

static ONNXTensorElementDataType ORT_API_CALL FQPTFLayerNorm_GetInputType(
    const OrtCustomOp *op, size_t idx)
{
    (void)op;
    if (idx == 0 || idx == 1 || idx == 3 || idx == 4 || idx == 5) return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
    if (idx == 2) return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32;
    return ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
}

static const char *ORT_API_CALL FQPTFLayerNorm_GetName(const OrtCustomOp *op)
{
    (void)op;
    return "FQPTFLayerNorm";
}

static const char *ORT_API_CALL FQPTFLayerNorm_GetExecutionProviderType(const OrtCustomOp *op)
{
    (void)op;
    return "CPUExecutionProvider";
}

static size_t ORT_API_CALL FQPTFLayerNorm_GetInputTypeCount(const OrtCustomOp *op)
{
    (void)op;
    return 6;
}

static ONNXTensorElementDataType ORT_API_CALL FQPTFLayerNorm_GetOutputType(
    const OrtCustomOp *op, size_t idx)
{
    (void)op;
    (void)idx;
    return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
}

static size_t ORT_API_CALL FQPTFLayerNorm_GetOutputTypeCount(const OrtCustomOp *op)
{
    (void)op;
    return 1;
}

static OrtCustomOpInputOutputCharacteristic ORT_API_CALL FQPTFLayerNorm_GetInputCharacteristic(
    const OrtCustomOp *op, size_t idx)
{
    (void)op;
    (void)idx;
    return INPUT_OUTPUT_REQUIRED;
}

static OrtCustomOpInputOutputCharacteristic ORT_API_CALL FQPTFLayerNorm_GetOutputCharacteristic(
    const OrtCustomOp *op, size_t idx)
{
    (void)op;
    (void)idx;
    return INPUT_OUTPUT_REQUIRED;
}

static OrtCustomOp g_fq_ptf_layernorm_op = {
    .version                    = ORT_API_VERSION,
    .CreateKernel               = FQPTFLayerNorm_CreateKernel,
    .GetName                    = FQPTFLayerNorm_GetName,
    .GetExecutionProviderType   = FQPTFLayerNorm_GetExecutionProviderType,
    .GetInputType               = FQPTFLayerNorm_GetInputType,
    .GetInputTypeCount          = FQPTFLayerNorm_GetInputTypeCount,
    .GetOutputType              = FQPTFLayerNorm_GetOutputType,
    .GetOutputTypeCount         = FQPTFLayerNorm_GetOutputTypeCount,
    .KernelCompute              = FQPTFLayerNorm_Compute,
    .KernelDestroy              = FQPTFLayerNorm_Destroy,
    .GetInputCharacteristic     = FQPTFLayerNorm_GetInputCharacteristic,
    .GetOutputCharacteristic    = FQPTFLayerNorm_GetOutputCharacteristic,
};

/* ══════════════════════════════════════════════════════════════════════════
 *  ivit.FQIntMmLinear / ivit.FQIntMmLinearRequant
 * ══════════════════════════════════════════════════════════════════════════ */
typedef struct {
    const OrtApi *ort;
} FQIntMmLinearKernel;

static void *ORT_API_CALL FQIntMmLinear_CreateKernel(
    const OrtCustomOp *op, const OrtApi *api, const OrtKernelInfo *info)
{
    (void)op;
    (void)info;
    FQIntMmLinearKernel *k = (FQIntMmLinearKernel *)malloc(sizeof(FQIntMmLinearKernel));
    k->ort = api;
    return k;
}

static void ORT_API_CALL FQIntMmLinear_Destroy(void *op_kernel) { free(op_kernel); }

static void fq_int_mm_linear_core(
    const float *x,
    const int8_t *w,
    float *out,
    int8_t *out_i8,
    int64_t rows,
    int64_t in_features,
    int64_t out_features,
    float in_scale,
    float in_zp,
    const float *out_m,
    const float *out_b,
    float requant_zp,
    int do_requant)
{
    int8_t *x_i8 = (int8_t *)malloc((size_t)rows * (size_t)in_features);
    if (!x_i8) return;
    for (int64_t r = 0; r < rows; r++) {
        for (int64_t c = 0; c < in_features; c++) {
            float q = roundf(x[r * in_features + c] / in_scale) + in_zp;
            if (q < 0.0f) q = 0.0f;
            if (q > 255.0f) q = 255.0f;
            int32_t centered = (int32_t)q - 128;
            x_i8[r * in_features + c] = (int8_t)centered;
        }
    }

    for (int64_t r = 0; r < rows; r++) {
        for (int64_t oc = 0; oc < out_features; oc++) {
            int32_t acc = 0;
            for (int64_t ic = 0; ic < in_features; ic++) {
                acc += (int32_t)x_i8[r * in_features + ic] * (int32_t)w[ic * out_features + oc];
            }
            float y = (float)acc * out_m[oc] + out_b[oc];
            if (do_requant) {
                float q = roundf(y) + requant_zp;
                if (q < 0.0f) q = 0.0f;
                if (q > 255.0f) q = 255.0f;
                int32_t centered = (int32_t)q - 128;
                out_i8[r * out_features + oc] = (int8_t)centered;
            } else {
                out[r * out_features + oc] = y;
            }
        }
    }
    free(x_i8);
}

static void ORT_API_CALL FQIntMmLinear_Compute(void *op_kernel, OrtKernelContext *ctx)
{
    FQIntMmLinearKernel *k = (FQIntMmLinearKernel *)op_kernel;
    const OrtApi *ort = k->ort;

    const OrtValue *x_val = NULL;
    ort->KernelContext_GetInput(ctx, 0, &x_val);
    OrtTensorTypeAndShapeInfo *shape_info = NULL;
    ort->GetTensorTypeAndShape(x_val, &shape_info);
    size_t ndim = 0;
    ort->GetDimensionsCount(shape_info, &ndim);
    int64_t dims[8] = {0};
    ort->GetDimensions(shape_info, dims, ndim);
    ort->ReleaseTensorTypeAndShapeInfo(shape_info);
    const float *x = NULL;
    ort->GetTensorMutableData((OrtValue *)(uintptr_t)x_val, (void **)&x);

    const OrtValue *w_val = NULL;
    ort->KernelContext_GetInput(ctx, 1, &w_val);
    const int8_t *w = NULL;
    ort->GetTensorMutableData((OrtValue *)(uintptr_t)w_val, (void **)&w);

    const OrtValue *in_scale_val = NULL;
    ort->KernelContext_GetInput(ctx, 2, &in_scale_val);
    const float *in_scale_ptr = NULL;
    ort->GetTensorMutableData((OrtValue *)(uintptr_t)in_scale_val, (void **)&in_scale_ptr);

    const OrtValue *in_zp_val = NULL;
    ort->KernelContext_GetInput(ctx, 3, &in_zp_val);
    const float *in_zp_ptr = NULL;
    ort->GetTensorMutableData((OrtValue *)(uintptr_t)in_zp_val, (void **)&in_zp_ptr);

    const OrtValue *out_m_val = NULL;
    ort->KernelContext_GetInput(ctx, 4, &out_m_val);
    const float *out_m = NULL;
    ort->GetTensorMutableData((OrtValue *)(uintptr_t)out_m_val, (void **)&out_m);

    const OrtValue *out_b_val = NULL;
    ort->KernelContext_GetInput(ctx, 5, &out_b_val);
    const float *out_b = NULL;
    ort->GetTensorMutableData((OrtValue *)(uintptr_t)out_b_val, (void **)&out_b);

    int64_t rows = 1;
    for (size_t i = 0; i + 1 < ndim; i++) rows *= dims[i];
    int64_t in_features = dims[ndim - 1];
    OrtTensorTypeAndShapeInfo *w_shape = NULL;
    ort->GetTensorTypeAndShape(w_val, &w_shape);
    int64_t wdims[2] = {0};
    ort->GetDimensions(w_shape, wdims, 2);
    ort->ReleaseTensorTypeAndShapeInfo(w_shape);
    int64_t out_features = wdims[1];
    int64_t out_dims[8] = {0};
    for (size_t i = 0; i < ndim - 1; i++) out_dims[i] = dims[i];
    out_dims[ndim - 1] = out_features;
    OrtValue *out_val = NULL;
    ort->KernelContext_GetOutput(ctx, 0, out_dims, ndim, &out_val);
    float *out = NULL;
    ort->GetTensorMutableData(out_val, (void **)&out);
    fq_int_mm_linear_core(
        x, w, out, NULL, rows, in_features, out_features,
        in_scale_ptr[0], in_zp_ptr[0], out_m, out_b, 0.0f, 0);
}

static void ORT_API_CALL FQIntMmLinearRequant_Compute(void *op_kernel, OrtKernelContext *ctx)
{
    FQIntMmLinearKernel *k = (FQIntMmLinearKernel *)op_kernel;
    const OrtApi *ort = k->ort;
    const OrtValue *x_val = NULL;
    ort->KernelContext_GetInput(ctx, 0, &x_val);
    OrtTensorTypeAndShapeInfo *shape_info = NULL;
    ort->GetTensorTypeAndShape(x_val, &shape_info);
    size_t ndim = 0;
    ort->GetDimensionsCount(shape_info, &ndim);
    int64_t dims[8] = {0};
    ort->GetDimensions(shape_info, dims, ndim);
    ort->ReleaseTensorTypeAndShapeInfo(shape_info);
    const float *x = NULL;
    ort->GetTensorMutableData((OrtValue *)(uintptr_t)x_val, (void **)&x);

    const OrtValue *w_val = NULL; ort->KernelContext_GetInput(ctx, 1, &w_val);
    const int8_t *w = NULL; ort->GetTensorMutableData((OrtValue *)(uintptr_t)w_val, (void **)&w);
    const OrtValue *in_scale_val = NULL; ort->KernelContext_GetInput(ctx, 2, &in_scale_val);
    const float *in_scale_ptr = NULL; ort->GetTensorMutableData((OrtValue *)(uintptr_t)in_scale_val, (void **)&in_scale_ptr);
    const OrtValue *in_zp_val = NULL; ort->KernelContext_GetInput(ctx, 3, &in_zp_val);
    const float *in_zp_ptr = NULL; ort->GetTensorMutableData((OrtValue *)(uintptr_t)in_zp_val, (void **)&in_zp_ptr);
    const OrtValue *out_m_val = NULL; ort->KernelContext_GetInput(ctx, 4, &out_m_val);
    const float *out_m = NULL; ort->GetTensorMutableData((OrtValue *)(uintptr_t)out_m_val, (void **)&out_m);
    const OrtValue *out_b_val = NULL; ort->KernelContext_GetInput(ctx, 5, &out_b_val);
    const float *out_b = NULL; ort->GetTensorMutableData((OrtValue *)(uintptr_t)out_b_val, (void **)&out_b);
    const OrtValue *rzp_val = NULL; ort->KernelContext_GetInput(ctx, 6, &rzp_val);
    const float *rzp_ptr = NULL; ort->GetTensorMutableData((OrtValue *)(uintptr_t)rzp_val, (void **)&rzp_ptr);

    int64_t rows = 1;
    for (size_t i = 0; i + 1 < ndim; i++) rows *= dims[i];
    int64_t in_features = dims[ndim - 1];
    OrtTensorTypeAndShapeInfo *w_shape = NULL;
    ort->GetTensorTypeAndShape(w_val, &w_shape);
    int64_t wdims[2] = {0};
    ort->GetDimensions(w_shape, wdims, 2);
    ort->ReleaseTensorTypeAndShapeInfo(w_shape);
    int64_t out_features = wdims[1];
    int64_t out_dims[8] = {0};
    for (size_t i = 0; i < ndim - 1; i++) out_dims[i] = dims[i];
    out_dims[ndim - 1] = out_features;
    OrtValue *out_val = NULL;
    ort->KernelContext_GetOutput(ctx, 0, out_dims, ndim, &out_val);
    int8_t *out = NULL;
    ort->GetTensorMutableData(out_val, (void **)&out);
    fq_int_mm_linear_core(
        x, w, NULL, out, rows, in_features, out_features,
        in_scale_ptr[0], in_zp_ptr[0], out_m, out_b, rzp_ptr[0], 1);
}

static ONNXTensorElementDataType ORT_API_CALL FQIntMmLinear_GetInputType(const OrtCustomOp *op, size_t idx)
{
    (void)op;
    if (idx == 1) return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8;
    return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
}
static const char *ORT_API_CALL FQIntMmLinear_GetExecutionProviderType(const OrtCustomOp *op) { (void)op; return "CPUExecutionProvider"; }
static size_t ORT_API_CALL FQIntMmLinear_GetInputTypeCount(const OrtCustomOp *op) { (void)op; return 6; }
static ONNXTensorElementDataType ORT_API_CALL FQIntMmLinear_GetOutputType(const OrtCustomOp *op, size_t idx) { (void)op; (void)idx; return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT; }
static size_t ORT_API_CALL FQIntMmLinear_GetOutputTypeCount(const OrtCustomOp *op) { (void)op; return 1; }
static OrtCustomOpInputOutputCharacteristic ORT_API_CALL FQIntMmLinear_GetInputCharacteristic(const OrtCustomOp *op, size_t idx) { (void)op; (void)idx; return INPUT_OUTPUT_REQUIRED; }
static OrtCustomOpInputOutputCharacteristic ORT_API_CALL FQIntMmLinear_GetOutputCharacteristic(const OrtCustomOp *op, size_t idx) { (void)op; (void)idx; return INPUT_OUTPUT_REQUIRED; }
static const char *ORT_API_CALL FQIntMmLinear_GetName(const OrtCustomOp *op) { (void)op; return "FQIntMmLinear"; }

static OrtCustomOp g_fq_int_mm_linear_op = {
    .version = ORT_API_VERSION,
    .CreateKernel = FQIntMmLinear_CreateKernel,
    .GetName = FQIntMmLinear_GetName,
    .GetExecutionProviderType = FQIntMmLinear_GetExecutionProviderType,
    .GetInputType = FQIntMmLinear_GetInputType,
    .GetInputTypeCount = FQIntMmLinear_GetInputTypeCount,
    .GetOutputType = FQIntMmLinear_GetOutputType,
    .GetOutputTypeCount = FQIntMmLinear_GetOutputTypeCount,
    .KernelCompute = FQIntMmLinear_Compute,
    .KernelDestroy = FQIntMmLinear_Destroy,
    .GetInputCharacteristic = FQIntMmLinear_GetInputCharacteristic,
    .GetOutputCharacteristic = FQIntMmLinear_GetOutputCharacteristic,
};

static size_t ORT_API_CALL FQIntMmLinearRequant_GetInputTypeCount(const OrtCustomOp *op) { (void)op; return 7; }
static ONNXTensorElementDataType ORT_API_CALL FQIntMmLinearRequant_GetOutputType(const OrtCustomOp *op, size_t idx) { (void)op; (void)idx; return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8; }
static const char *ORT_API_CALL FQIntMmLinearRequant_GetName(const OrtCustomOp *op) { (void)op; return "FQIntMmLinearRequant"; }

static OrtCustomOp g_fq_int_mm_linear_requant_op = {
    .version = ORT_API_VERSION,
    .CreateKernel = FQIntMmLinear_CreateKernel,
    .GetName = FQIntMmLinearRequant_GetName,
    .GetExecutionProviderType = FQIntMmLinear_GetExecutionProviderType,
    .GetInputType = FQIntMmLinear_GetInputType,
    .GetInputTypeCount = FQIntMmLinearRequant_GetInputTypeCount,
    .GetOutputType = FQIntMmLinearRequant_GetOutputType,
    .GetOutputTypeCount = FQIntMmLinear_GetOutputTypeCount,
    .KernelCompute = FQIntMmLinearRequant_Compute,
    .KernelDestroy = FQIntMmLinear_Destroy,
    .GetInputCharacteristic = FQIntMmLinear_GetInputCharacteristic,
    .GetOutputCharacteristic = FQIntMmLinear_GetOutputCharacteristic,
};


/* ══════════════════════════════════════════════════════════════════════════
 *  RegisterCustomOps — called by ort_test via -DUSE_CUSTOM_OP_LIBRARY
 *  TwinSoftmaxMatMul / TwinGeluLinear live in ivit_gemmini_ops.cc (Gemmini MM).
 * ══════════════════════════════════════════════════════════════════════════ */
OrtStatus *ORT_API_CALL RegisterIvitCustomOps(OrtSessionOptions *options,
                                              const OrtApiBase *api_base)
{
    g_ort = api_base->GetApi(ORT_API_VERSION);
    if (!g_ort) return NULL;

    OrtCustomOpDomain *domain = NULL;
    OrtStatus *status;

    status = g_ort->CreateCustomOpDomain("ivit", &domain);
    if (status) return status;

    status = g_ort->CustomOpDomain_Add(domain, &g_shiftmax_op);
    if (status) return status;

    status = g_ort->CustomOpDomain_Add(domain, &g_shiftmax_int32_op);
    if (status) return status;

    status = g_ort->CustomOpDomain_Add(domain, &g_shiftgelu_op);
    if (status) return status;

    status = g_ort->CustomOpDomain_Add(domain, &g_qlayernorm_op);
    if (status) return status;

    status = g_ort->CustomOpDomain_Add(domain, &g_qlayernorm_int16_op);
    if (status) return status;

    status = g_ort->CustomOpDomain_Add(domain, &g_qlayernorm_int32_op);
    if (status) return status;

    status = g_ort->CustomOpDomain_Add(domain, &g_qlayernorm_i64_op);
    if (status) return status;

    status = g_ort->CustomOpDomain_Add(domain, &g_qlayernorm_int16_i64_op);
    if (status) return status;

    status = g_ort->CustomOpDomain_Add(domain, &g_qlayernorm_int32_i64_op);
    if (status) return status;

    status = g_ort->CustomOpDomain_Add(domain, &g_fq_ptf_layernorm_op);
    if (status) return status;

    status = g_ort->CustomOpDomain_Add(domain, &g_fq_qk_matmul_op);
    if (status) return status;

    status = g_ort->CustomOpDomain_Add(domain, &g_fq_lis_softmax_op);
    if (status) return status;

    status = g_ort->CustomOpDomain_Add(domain, &g_fq_attn_v_matmul_op);
    if (status) return status;

    status = g_ort->CustomOpDomain_Add(domain, &g_fq_int_mm_linear_op);
    if (status) return status;

    status = g_ort->CustomOpDomain_Add(domain, &g_fq_int_mm_linear_requant_op);
    if (status) return status;

    status = AddIvitGemminiOps(domain, g_ort);
    if (status) return status;

    status = g_ort->AddCustomOpDomain(options, domain);
    return status;
}

OrtStatus *ORT_API_CALL RegisterCustomOps(OrtSessionOptions *options,
                                          const OrtApiBase *api_base)
{
    return RegisterIvitCustomOps(options, api_base);
}
