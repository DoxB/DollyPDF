from datasets import load_dataset
import load_local_model
data = load_dataset("json", 
                    data_files="./train_dataset/ko_alpaca_data.json")

def generate_prompt(data_point):
    # taken from https://github.com/tloen/alpaca-lora
    if data_point["instruction"]:
        return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{data_point["instruction"]}

### Input:
{data_point["input"]}

### Response:
{data_point["output"]}"""
    else:
        return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{data_point["instruction"]}

### Response:
{data_point["output"]}"""


data = data.map(lambda data_point: {"prompt": load_local_model.tokenizer(generate_prompt(data_point))})