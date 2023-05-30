import torch
from pipeline import InstructionTextGenerationPipeline

# use dolly-v2-12b if you're using Colab Pro+, using pythia-2.8b for Free Colab

model = AutoModelForCausalLM.from_pretrained("../../local_models/dolly-v2-3b", 
                                             device_map="auto",
                                             torch_dtype=torch.bfloat16)

tokenizer = AutoTokenizer.from_pretrained("../../local_models/dolly-v2-3b", padding_side="left")

generate_text = InstructionTextGenerationPipeline(model=model, tokenizer=tokenizer)