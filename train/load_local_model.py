import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("./models/dolly-v2-3b", padding_side="left")

model = AutoModelForCausalLM.from_pretrained("./models/dolly-v2-3b", 
                                             device_map="auto",
                                             torch_dtype=torch.bfloat16)