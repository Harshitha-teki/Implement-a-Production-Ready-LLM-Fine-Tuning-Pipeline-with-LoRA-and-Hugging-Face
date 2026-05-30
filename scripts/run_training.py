import os
import torch
import json
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments
import os
import json
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments
from peft import LoraConfig
from trl import SFTTrainer


def load_lora_config(path="config/lora_config.json"):
    if not os.path.isfile(path):
        raise FileNotFoundError(f"LoRA config not found at: {path}")
    with open(path, "r") as f:
        return json.load(f)


def train():
    model_id = os.getenv("BASE_MODEL_ID", "TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    print(f"Starting training for: {model_id}")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb_config, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = load_dataset("json", data_files={"train": "data/processed/train.json"})

    # Load LoRA parameters from config file
    lora_cfg = load_lora_config()
    peft_config = LoraConfig(
        r=int(lora_cfg.get("r", 8)),
        lora_alpha=int(lora_cfg.get("lora_alpha", 16)),
        lora_dropout=float(lora_cfg.get("lora_dropout", 0.0)),
        target_modules=lora_cfg.get("target_modules", ["q_proj", "v_proj"]),
        bias=lora_cfg.get("bias", "none"),
        task_type=lora_cfg.get("task_type", "CAUSAL_LM"),
    )

    args = TrainingArguments(
        output_dir="./results",
        per_device_train_batch_size=int(os.getenv("BATCH_SIZE", 4)),
        max_steps=int(os.getenv("MAX_STEPS", 50)),
        logging_steps=10,
        report_to="wandb" if os.getenv("WANDB_API_KEY") else "none",
    )

    def formatting_prompts_func(example):
        return example["text"]

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset["train"],
        peft_config=peft_config,
        args=args,
        formatting_func=formatting_prompts_func,
    )

    trainer.train()
    trainer.model.save_pretrained("./models/fine_tuned_adapter")
    print("✅ Success: Adapter saved to ./models/fine_tuned_adapter")


if __name__ == "__main__":
    train()
