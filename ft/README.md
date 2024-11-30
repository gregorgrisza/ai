# Fine-tuning

**Notes**

> Code examples variables are encapsulated by `<` and `>`

## Setup

- Configure access tokens `export HF_TOKEN="<huggingface_token>"`

- Install libraries PyTorch tools: https://pytorch.org/torchtune/stable/install.html#install-label

(Optionally): `pip install bitsandbytes` most likely will be needed in fine-tuning steps

- Login to HuggingFace: `huggingface-cli login`

- Download model: `tune download meta-llama/Llama-3.2-1B-Instruct --output-dir ~/repos/ai/models/Meta-Llama-3-8B-Instruct/checkpoint --hf-token $HF_TOKEN`

Note: you must accept license agreement on HuggingFace if the model is gated. Search for desired model on HF (HuggingFace) and request access by submitting acceptance.

## Fine-tuning

### Using Torchtune

#### Configuration

Ref.: https://pytorch.org/torchtune/stable/deep_dives/configs.html#config-tutorial-label


Available configurations `tune ls`

Copy recipe for later modification: `tune cp full_finetune_single_device .`

Copy configuration for later modification `tune cp llama3_2/1B_full_single_device <llama3_2_1B_full_single_device.yaml>`

Validate config: `tune validate llama3_2_1B_full_single_device.yaml`

#### Run fine-tuning

Run locally modified configuration and recipe to start fine-tuning: `tune run <full_finetune_single_device.py> --config <llama3_2_1B_full_single_device.yaml>`
