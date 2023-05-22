#pragma once

#include "ggml.h"

#ifdef __cplusplus
extern "C" {
#endif

// TODO: ???
typedef void starcoder_c_ctx;

// C ABI version (starcoder_c)
bool starcoder_c_model_load(starcoder_c_ctx **ctx, const char *model_file_path);

#ifdef __cplusplus
}
#endif