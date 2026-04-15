import os
import torch
import json
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments
from peft import LoraConfig
from trl import SFTTrainer

def train():
    model_id = os.getenv("BASE_MODEL_ID", "TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    print(f"Starting training for: {model_id}")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    
    model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb_config, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    
    dataset = load_dataset("json", data_files={"train": "data/processed/train.json"})
    
    peft_config = LoraConfig(
        r=16, 
        lora_alpha=32, 
        target_modules=["q_proj", "v_proj"], 
        task_type="CAUSAL_LM"
    )
    
    args = TrainingArguments(
        output_dir="./results",
        per_device_train_batch_size=4,
        max_steps=50,
        logging_steps=10,
        report_to="none"
    )
    
    # We remove 'dataset_text_field' and use 'formatting_func' instead
    # This is the most compatible way for modern SFTTrainer versions
    def formatting_prompts_func(example):
        return example['text']

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset["train"],
        peft_config=peft_config,
        args=args,
        formatting_func=formatting_prompts_func,
    )
    
    trainer.train()
    trainer.model.save_pretrained("./models/fine_tuned_adapter")
    print("✅ Success: Adapter saved!")

if __name__ == "__main__":
    train()
