import os
import json
from datasets import load_dataset

def main():
    print("Step 1: Downloading dataset...")
    dataset = load_dataset("yahma/alpaca-cleaned", split="train[:1000]")
    dataset = dataset.train_test_split(test_size=0.1)
    
    def format_instruction(sample):
        return {"text": f"### Instruction:\n{sample['instruction']}\n\n### Response:\n{sample['output']}"}
    
    train_data = dataset['train'].map(format_instruction)
    val_data = dataset['test'].map(format_instruction)
    
    os.makedirs("data/processed", exist_ok=True)
    with open("data/processed/train.json", "w") as f:
        json.dump(list(train_data), f)
    with open("data/processed/validation.json", "w") as f:
        json.dump(list(val_data), f)
    print("✅ Success: Data saved to data/processed/")

if __name__ == "__main__":
    main()
