#include "ggml.h"

#include "common.h"

#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <map>
#include <stdexcept>
#include <string>
#include <unistd.h>
#include <vector>

#include "util/httplib.hpp"
#include "util/json.hpp"

#include "starcoder.hpp"

std::vector<std::string>
starcoder_demo_generate(const starcoder_model &model, const gpt_vocab &vocab,
                        std::vector<gpt_vocab::id> input_ids, int n_predict,
                        int top_k, float top_p, float temp, int n_threads,
                        int n_batch, std::mt19937 rng) {

  int n_past = 0;

  int64_t t_sample_us = 0;
  int64_t t_predict_us = 0;

  std::vector<float> logits;

  n_predict = std::min(n_predict, model.hparams.n_ctx - (int)input_ids.size());
  // printf("%s: corrected n_predict = %d\n", __func__, n_predict);

  //   std::string model_output_text = "";
  std::vector<std::string> model_output_tokens;

  // submit the input prompt token-by-token
  // this reduces the memory usage during inference, at the cost of a bit of
  // speed at the beginning
  std::vector<gpt_vocab::id> embd;

  // determine the required inference memory per token:
  size_t mem_per_token = 0;
  starcoder_eval(model, n_threads, 0, {0, 1, 2, 3}, logits, mem_per_token);

  for (int i = embd.size(); i < input_ids.size() + n_predict; i++) {
    // predict
    if (embd.size() > 0) {
      const int64_t t_start_us = ggml_time_us();

      if (!starcoder_eval(model, n_threads, n_past, embd, logits,
                          mem_per_token)) {
        // printf("Failed to predict\n");
        // return 1;
        // throw error
        throw std::runtime_error("Failed to predict");
      }

      t_predict_us += ggml_time_us() - t_start_us;
    }

    n_past += embd.size();
    embd.clear();

    if (i >= input_ids.size()) {
      // sample next token
      // printf("%s: sampling token #%d (top_k = %d, top_p = %f, temp = %f)\n",
      //       __func__, i, top_k, top_p, temp);

      const int n_vocab = model.hparams.n_vocab;

      gpt_vocab::id id = 0;

      {
        const int64_t t_start_sample_us = ggml_time_us();

        id = gpt_sample_top_k_top_p(vocab,
                                    logits.data() + (logits.size() - n_vocab),
                                    top_k, top_p, temp, rng);

        t_sample_us += ggml_time_us() - t_start_sample_us;
      }

      // add it to the context
      embd.push_back(id);
    } else {
      // if here, it means we are still processing the input prompt
      // printf("%s: processing input token #%d\n", __func__, i);
      for (int k = i; k < input_ids.size(); k++) {
        embd.push_back(input_ids[k]);
        if (embd.size() >= n_batch) {
          break;
        }
      }
      i += embd.size() - 1;
    }

    // display text
    for (auto id : embd) {
      // printf("%s", vocab.id_to_token[id].c_str());
      // use find instead of []
      auto it = vocab.id_to_token.find(id);
      if (it != vocab.id_to_token.end()) {
        printf("%s", it->second.c_str());
        // model_output_text += it->second;
        model_output_tokens.push_back(it->second);
      } else {
        // throw an error (wtf token?)
        throw std::runtime_error("failed to decode token: " +
                                 std::to_string(id));
      }
    }
    fflush(stdout);

    // check if model is santacoder
    if (model.hparams.n_layer <= 30 && embd.back() == 49152) {
      break;
    }
    // check if model is starcoder
    else if (embd.back() == 0) { // TODO: this is only for starcoder
      // printf("stopping generation due to end token\n");
      break;
    }
  }

  //   return model_output_text;
  return model_output_tokens;
}

int main(int argc, char **argv) {
  ggml_time_init();

  const int64_t t_main_start_us = ggml_time_us();

  gpt_params params;
  params.model = "models/bigcode/gpt_bigcode-santacoder-ggml.bin";

  if (gpt_params_parse(argc, argv, params) == false) {
    return 1;
  }

  if (params.seed < 0) {
    params.seed = time(NULL);
  }

  printf("%s: seed = %d\n", __func__, params.seed);

  std::mt19937 rng(params.seed);
  //   if (params.prompt.empty()) {
  //     if (!isatty(STDIN_FILENO)) {
  //       std::string line;
  //       while (std::getline(std::cin, line)) {
  //         params.prompt = params.prompt + "\n" + line;
  //       }
  //     } else {
  //       params.prompt = gpt_random_prompt(rng);
  //     }
  //   }

  int64_t t_load_us = 0;

  gpt_vocab vocab;
  starcoder_model model;

  // load the model
  {
    const int64_t t_start_us = ggml_time_us();

    if (!starcoder_model_load(params.model, model, vocab)) {
      fprintf(stderr, "%s: failed to load model from '%s'\n", __func__,
              params.model.c_str());
      return 1;
    }

    t_load_us = ggml_time_us() - t_start_us;
  }

  // start http server
  fprintf(stderr, "%s: starting http server on port %d\n", __func__,
          params.http_server_port);

  httplib::Server svr;

  // POST /v1/starcoder/generate
  //    Input: { "prompt": "...", "n_predict": 200, "top_k": 40, "top_p": 0.9,
  //    "temp": 0.9 } Output: { "text": "..." }

  svr.Post("/v1/starcoder/generate", [model, vocab, params,
                                      rng](const httplib::Request &req,
                                           httplib::Response &res) {
    try {
      // read the request body as JSON
      nlohmann::json req_json = nlohmann::json::parse(req.body);

      // ensure prompt field is present
      if (!req_json.contains("prompt")) {
        throw std::runtime_error("missing prompt field");
      }
      std::string req_prompt = req_json["prompt"];
      // all other fields are optional
      int req_n_predict = req_json.value("n_predict", 200);
      int req_top_k = req_json.value("top_k", 40);
      float req_top_p = req_json.value("top_p", 0.9);
      float req_temp = req_json.value("temp", 0.9);

      // tokenize the prompt
      std::vector<gpt_vocab::id> input_ids = ::gpt_tokenize(vocab, req_prompt);

      printf("%s: prompt: '%s'\n", __func__, req_prompt.c_str());
      printf("%s: prompt_n = %zu\n", __func__, input_ids.size());
      printf("%s: n_predict = %d\n", __func__, req_n_predict);
      printf("%s: top_k = %d\n", __func__, req_top_k);
      printf("%s: top_p = %f\n", __func__, req_top_p);
      printf("%s: temp = %f\n", __func__, req_temp);

      // call model
      std::vector<std::string> output_tokens = starcoder_demo_generate(
          model, vocab, input_ids, req_n_predict, req_top_k, req_top_p,
          req_temp, params.n_threads, params.n_batch, rng);

      printf("%s: output_n = %zu\n", __func__, output_tokens.size());
      // dump output tokens
      printf("%s: output: [", __func__);
      for (auto &token : output_tokens) {
        printf("%s, ", token.c_str());
      }
      printf("]\n");

      std::string model_output;
      // join tokens into a string
      for (auto &token : output_tokens) {
        model_output += token;
      }

      // create response json
      nlohmann::json res_json;
      res_json["text"] = model_output;

      // set response content type
      res.set_content(res_json.dump(), "application/json");
      res.status = 200;
    } catch (const std::exception &e) {
      fprintf(stderr, "%s: exception: %s\n", __func__, e.what());
      // res.set_content("{}", "application/json");
      res.set_content(e.what(), "text/plain");
      res.status = 400;
    }
  });

  // listen on port
  svr.listen("0.0.0.0", params.http_server_port);

  ggml_free(model.ctx);

  return 0;
}
