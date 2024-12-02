#include <torch/script.h>
#include <sentencepiece_processor.h>

#include <iostream>
#include <memory>

int main(int argc, const char* argv[]) {
  if (argc != 2) {
    std::cerr << "usage: " << std::string(argv[0]) << " <path-to-exported-script-module>\n";
    return -1;
  }


  torch::jit::script::Module module;
  try {
    module = torch::jit::load("/Users/grzegorz.michalak/repos/ai/ts/my_llama.pt");
    module.eval();
    auto sp = std::make_shared<sentencepiece::SentencePieceProcessor>();
  } catch (const c10::Error& e) {
    std::cerr << "error loading the model\n";
    return -1;
  }

  std::cout << "ok\n";
}