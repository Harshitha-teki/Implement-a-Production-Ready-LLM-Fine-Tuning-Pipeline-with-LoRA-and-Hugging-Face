from fastapi import FastAPI
from pydantic import BaseModel
import os
import torch
from fastapi import FastAPI
from pydantic import BaseModel
import os
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

app = FastAPI()


# Configuration (can be overridden via env)
BASE_MODEL_ID = os.getenv("BASE_MODEL_ID", "TinyLlama/TinyLlama-1.1B-Chat-v1.0")
ADAPTER_PATH = os.getenv("ADAPTER_PATH", "./models/fine_tuned_adapter")


class GenerateRequest(BaseModel):
    prompt: str
    max_new_tokens: int = 128


@app.on_event("startup")
def load_model():
    """Load base model and LoRA adapter once at startup for efficient inference."""
    global model, tokenizer
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load tokenizer and base model
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load base model in float16 for inference where possible
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
    )

    # If adapter exists, load it with PeftModel on top of base model
    if os.path.isdir(ADAPTER_PATH):
        try:
            model = PeftModel.from_pretrained(model, ADAPTER_PATH, device_map="auto" if torch.cuda.is_available() else None)
        except Exception:
            # Fall back to loading adapter onto CPU if device_map failed
            model = PeftModel.from_pretrained(model, ADAPTER_PATH)

    model.eval()


@app.get("/health")
def health():
    # Return the exact required value to satisfy tests
    return {"status": "ok"}


@app.post("/generate")
def generate(req: GenerateRequest):
    """Generate text from the loaded model. Uses the preloaded tokenizer and model.

    Returns JSON with key `generated_text`.
    """
    inputs = tokenizer(req.prompt, return_tensors="pt").to(next(model.parameters()).device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=req.max_new_tokens)
    text = tokenizer.decode(out[0], skip_special_tokens=True)
    return {"generated_text": text}
