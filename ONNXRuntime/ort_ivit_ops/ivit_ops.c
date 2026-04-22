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
 *  RegisterCustomOps — called by ort_test via -DUSE_CUSTOM_OP_LIBRARY
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
