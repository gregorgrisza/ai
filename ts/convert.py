import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM
import os

hf_token=os.environ['HF_TOKEN']
# base_model_id = "meta-llama/Llama-3.2-1B-Instruct"
# base_model_id = "facebook/opt-1.3b"
base_model_id = "/Users/grzegorz.michalak/repos/ai/models/meta-llama_Llama-3.2-1B-Instruct/checkpoint"

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()

        mymodel = AutoModelForCausalLM

        self.model = mymodel.from_pretrained(
                        base_model_id,
                        return_dict=True,
                        low_cpu_mem_usage=True,
                        device_map= "auto",
                        torch_dtype=torch.float16,
                        token=hf_token,
                        torchscript=True,
                        )

        self.model.output_hidden_states = False

    def forward(self, tokens_tensor):
        self.model.eval()
        o = self.model(tokens_tensor, output_hidden_states=False)
        return o[0]


question = "What is the capital of France?"
tokenizer = AutoTokenizer.from_pretrained(base_model_id, token=hf_token)
tokens_tensor = tokenizer.encode(question, return_tensors="pt").to("mps")

model = MyModel()
model.eval()

with torch.no_grad():
    outputs = model(tokens_tensor)
    decoded_output = tokenizer.decode(outputs[0].argmax(dim=-1))
    print("\v\tGenerated output: " + decoded_output)


    trace_model = torch.jit.trace(model, [tokens_tensor])

    torch.jit.save(trace_model, "my_llama.pt")

loaded_model = torch.jit.load("my_llama.pt")
loaded_model.eval()

with torch.no_grad():
    outputs = loaded_model(tokens_tensor)
    print(type(outputs[0]))
    v=outputs[0].argmax(dim=-1)
    print("v:" + str(v))
    print("v:type" + str(type(v)))
    decoded_output = tokenizer.decode(v)
    print("\v\tdecoded_output from loaded model: " + decoded_output)
