# Implement a Production-Ready LLM Fine-Tuning Pipeline with LoRA and Hugging Face

This repository contains a complete, end-to-end pipeline for fine-tuning Large Language Models (LLMs) using **PEFT (Parameter-Efficient Fine-Tuning)** and **LoRA (Low-Rank Adaptation)**. The project demonstrates how to take a base model, adapt it to specific instructions, and deploy the resulting adapter to the Hugging Face Hub.

## 🚀 Project Overview

The goal of this project was to fine-tune the **TinyLlama-1.1B-Chat** model to follow instructions more effectively using the Alpaca dataset. By using 4-bit quantization and LoRA, the training was made efficient enough to run on a single commodity GPU (NVIDIA T4).

### Key Features:
* **Quantization:** Utilized `bitsandbytes` for 4-bit NormalFloat (NF4) quantization to reduce VRAM usage.
* **LoRA Adaptation:** Trained only a small fraction of the model's parameters, making it faster and lightweight.
* **Data Pipeline:** Custom scripts for processing and tokenizing the Alpaca-style instruction dataset.
* **Deployment:** Integrated with Hugging Face Hub for automated model versioning and hosting.

---

## 📦 Model & Weights

The fine-tuned LoRA adapter has been published to Hugging Face. You can find the weights and the model card here:

👉 **[Harshitha2407/TinyLlama-Alpaca-FineTuned](https://huggingface.co/Harshitha2407/TinyLlama-Alpaca-FineTuned)**

---

## 🛠️ Tech Stack

* **Model:** [TinyLlama/TinyLlama-1.1B-Chat-v1.0](https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0)
* **Library:** `transformers`, `peft`, `trl`, `bitsandbytes`, `accelerate`
* **Environment:** Google Colab / Linux
* **Storage:** Google Drive / GitHub / Hugging Face

---

## 📂 Repository Structure


├── data/
│   └── processed/          # Tokenized and formatted datasets
├── models/
│   └── fine_tuned_adapter/ # Local copy of the LoRA weights
├── scripts/
│   ├── run_training.py     # Main training execution script
│   └── evaluate_model.py   # Script to test inference
├── README.md               # Project documentation
└── .gitignore              # Ensures large model files are not pushed to Git 


🚀 How to Use
1. Installation
Bash
pip install -q torch transformers peft datasets bitsandbytes accelerate
2. Inference with the Adapter
You can use the following snippet to run the model directly from the Hugging Face Hub:

Python
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
adapter_id = "Harshitha2407/TinyLlama-Alpaca-FineTuned"

# Load Base Model
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16, device_map="auto")

# Load your custom Adapter
model = PeftModel.from_pretrained(model, adapter_id)

# Tokenize and Generate
tokenizer = AutoTokenizer.from_pretrained(model_id)
prompt = "### Instruction:\nExplain fine-tuning to a student.\n\n### Response:\n"
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
outputs = model.generate(**inputs, max_new_tokens=100)

print(tokenizer.decode(outputs[0], skip_special_tokens=True))

Training Results
The model was trained for 50 steps on the Alpaca dataset.

Final Loss: ~1.365

Mean Token Accuracy: ~65%

Training Time: ~2.5 minutes on a Tesla T4 GPU.

🤝 Contributing
Feel free to fork this repository or open an issue if you have suggestions for improving the pipeline!


---

### ## How to update it on GitHub:
1.  Go to your [GitHub Repository](https://github.com/Harshitha-teki/Implement-a-Production-Ready-LLM-Fine-Tuning-Pipeline-with-LoRA-and-Hugging-Face).
2.  Click the **README.md** file.
3.  Click the **Pencil icon** (Edit this file).
4.  Delete the existing text, paste the content above, and click **Commit changes**.
