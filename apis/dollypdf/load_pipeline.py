import torch
from transformers import pipeline

# use dolly-v2-12b if you're using Colab Pro+, using pythia-2.8b for Free Colab

generate_text = pipeline('alpaca-lora-dolly-2.0', model="../../trained_models/alpaca-lora-dolly-2.0", 
                         torch_dtype=torch.bfloat16, 
                         trust_remote_code=True,
                         device_map="auto")