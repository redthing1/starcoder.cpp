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

std::string starcoder_demo_generate(const starcoder_model &model,
                                    const gpt_vocab &vocab,
                                    std::string prompt_text, int n_predict,
                                    int top_k, float top_p, float temp,
                                    int n_threads, int n_batch,
                                    std::mt19937 rng) {

  int n_past = 0;

  int64_t t_sample_us = 0;
  int64_t t_predict_us = 0;

  std::vector<float> logits;

  // tokenize the prompt
  std::vector<gpt_vocab::id> embd_inp = ::gpt_tokenize(vocab, prompt_text);

  n_predict = std::min(n_predict, model.hparams.n_ctx - (int)embd_inp.size());

  printf("%s: prompt: '%s'\n", __func__, prompt_text.c_str());
  printf("%s: number of tokens in prompt = %zu, first 8 tokens: ", __func__,
         embd_inp.size());
  for (int i = 0; i < std::min(8, (int)embd_inp.size()); i++) {
    printf("%d ", embd_inp[i]);
  }
  printf("\n\n");

  // submit the input prompt token-by-token
  // this reduces the memory usage during inference, at the cost of a bit of
  // speed at the beginning
  std::vector<gpt_vocab::id> embd;

  // determine the required inference memory per token:
  size_t mem_per_token = 0;
  starcoder_eval(model, n_threads, 0, {0, 1, 2, 3}, logits, mem_per_token);

  for (int i = embd.size(); i < embd_inp.size() + n_predict; i++) {
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

    if (i >= embd_inp.size()) {
      // sample next token
      const int top_k = top_k;
      const float top_p = top_p;
      const float temp = temp;

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
      for (int k = i; k < embd_inp.size(); k++) {
        embd.push_back(embd_inp[k]);
        if (embd.size() >= n_batch) {
          break;
        }
      }
      i += embd.size() - 1;
    }

    // // display text
    // for (auto id : embd) {
    //   printf("%s", vocab.id_to_token[id].c_str());
    // }
    // fflush(stdout);

    // // check if model is santacoder
    // if (model.hparams.n_layer <= 30 && embd.back() == 49152) {
    //   break;
    // }
    // // check if model is starcoder
    // else if (embd.back() == 0) { // TODO: this is only for starcoder
    //   break;
    // }

    // return text as a string
    std::string ret_text = "";
    for (auto id : embd) {
      // ret_text += vocab.id_to_token[id];
      // use find instead of []
      auto it = vocab.id_to_token.find(id);
      if (it != vocab.id_to_token.end()) {
        ret_text += it->second;
      } else {
        // throw an error (wtf token?)
        throw std::runtime_error("failed to decode token: " +
                                 std::to_string(id));
      }
    }

    return ret_text;
  }

  // ???
  return "";
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

  svr.Post("/v1/starcoder/generate",
           [model, vocab, params, rng](const httplib::Request &req,
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

               // call model
               std::string model_output = starcoder_demo_generate(
                   model, vocab, req_prompt, req_n_predict, req_top_k,
                   req_top_p, req_temp, params.n_threads, params.n_batch, rng);

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
