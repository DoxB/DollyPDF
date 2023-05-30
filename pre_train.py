import os

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
import torch.nn as nn
import bitsandbytes as bnb
from datasets import load_dataset
import transformers
from transformers import AutoTokenizer, AutoModel, AutoConfig, GPTJForCausalLM

from peft import prepare_model_for_int8_training, LoraConfig, get_peft_model

from train import load_local_dataset
from train import load_local_model

model = load_local_model.model
tokenizer = load_local_model.tokenizer

############################# setting ############################# 

# Settings for A100 - For 3090 
MICRO_BATCH_SIZE = 4  # change to 4 for 3090
BATCH_SIZE = 128
GRADIENT_ACCUMULATION_STEPS = BATCH_SIZE // MICRO_BATCH_SIZE
EPOCHS = 2  # paper uses 3
LEARNING_RATE = 2e-5  
CUTOFF_LEN = 256  
LORA_R = 4
LORA_ALPHA = 16
LORA_DROPOUT = 0.05
     
###################################################################

model = prepare_model_for_int8_training(model, 
                                        use_gradient_checkpointing=True)

config = LoraConfig(
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    lora_dropout=LORA_DROPOUT,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, config)
tokenizer.pad_token_id = 0  # unk. we want this to be different from the eos token

data = load_local_dataset.data

data = data.shuffle().map(
    lambda data_point: tokenizer(
        load_local_dataset.generate_prompt(data_point),
        truncation=True,
        max_length=CUTOFF_LEN,
        padding="max_length",
    )
)

trainer = transformers.Trainer(
    model=model,
    train_dataset=data["train"],
    args=transformers.TrainingArguments(
        per_device_train_batch_size=MICRO_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        warmup_steps=100,
        num_train_epochs=EPOCHS,
        learning_rate=LEARNING_RATE,
        fp16=True,
        logging_steps=1,
        output_dir="lora-dolly",
        save_total_limit=3,
    ),
    data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
)
model.config.use_cache = False
trainer.train(resume_from_checkpoint=False)

model.save_pretrained("./models/ko-alpaca-lora-dolly-2.0")