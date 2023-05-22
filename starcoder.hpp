#pragma once

#include "ggml.h"

#include "common.h"
// #include "common-ggml.h"

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// default hparams (GPT-2 117M)
// https://huggingface.co/bigcode/gpt_bigcode-santacoder/blob/main/config.json
struct starcoder_hparams {
  int32_t n_vocab = 49280;
  int32_t n_ctx = 2048;
  int32_t n_embd = 2048;
  int32_t n_head = 16;
  int32_t n_layer = 24;
  int32_t ftype = 1;
};

struct starcoder_layer {
  // normalization
  struct ggml_tensor *ln_1_g;
  struct ggml_tensor *ln_1_b;

  struct ggml_tensor *ln_2_g;
  struct ggml_tensor *ln_2_b;

  // attention
  struct ggml_tensor *c_attn_attn_w;
  struct ggml_tensor *c_attn_attn_b;

  struct ggml_tensor *c_attn_proj_w;
  struct ggml_tensor *c_attn_proj_b;

  // mlp
  struct ggml_tensor *c_mlp_fc_w;
  struct ggml_tensor *c_mlp_fc_b;

  struct ggml_tensor *c_mlp_proj_w;
  struct ggml_tensor *c_mlp_proj_b;
};

struct starcoder_model {
  starcoder_hparams hparams;

  // normalization
  struct ggml_tensor *ln_f_g;
  struct ggml_tensor *ln_f_b;

  struct ggml_tensor *wte;     // position embedding
  struct ggml_tensor *wpe;     //    token embedding
  struct ggml_tensor *lm_head; // language model head

  std::vector<starcoder_layer> layers;

  // key + value memory
  struct ggml_tensor *memory_k;
  struct ggml_tensor *memory_v;

  //
  struct ggml_context *ctx;
  std::map<std::string, struct ggml_tensor *> tensors;
};

bool starcoder_model_load(const std::string &fname, starcoder_model &model,
                          gpt_vocab &vocab);
bool starcoder_eval(const starcoder_model &model, const int n_threads,
                    const int n_past,
                    const std::vector<gpt_vocab::id> &embd_inp,
                    std::vector<float> &embd_w, size_t &mem_per_token);

#ifdef __cplusplus
}
#endif
