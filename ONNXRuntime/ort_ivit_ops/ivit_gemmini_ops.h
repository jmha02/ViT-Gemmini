#pragma once

#include "onnxruntime_c_api.h"

#ifdef __cplusplus
extern "C" {
#endif

OrtStatus *ORT_API_CALL AddIvitGemminiOps(OrtCustomOpDomain *domain, const OrtApi *ort);
OrtStatus *ORT_API_CALL RegisterIvitCustomOps(OrtSessionOptions *options, const OrtApiBase *api_base);
void ORT_API_CALL SetIvitExecutionMode(int mode);

#ifdef __cplusplus
}
#endif
