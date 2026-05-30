# run_gpu_training.ps1
# Helper PowerShell script to build the docker image, run training with GPU, run evaluation,
# and copy artifacts back to the host. Edit the environment variables below as needed.

param(
    [string]$BASE_MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    [string]$WANDB_API_KEY = "",
    [string]$ADAPTER_PATH = "./models/fine_tuned_adapter",
    [int]$BATCH_SIZE = 4,
    [int]$MAX_STEPS = 1000
)

Write-Host "Checking Docker GPU availability..."
try {
    docker run --rm --gpus all nvidia/cuda:11.0-base nvidia-smi
} catch {
    Write-Warning "Docker GPU check failed. Ensure Docker Desktop + WSL2 + NVIDIA drivers + NVIDIA Container Toolkit are installed."
}

Write-Host "Building Docker image (no-cache)..."
docker build -t llm-pipeline .

Write-Host "Running training container with GPU..."
$envArgs = @(
    "-e BASE_MODEL_ID=$BASE_MODEL_ID",
    "-e ADAPTER_PATH=$ADAPTER_PATH",
    "-e BATCH_SIZE=$BATCH_SIZE",
    "-e MAX_STEPS=$MAX_STEPS"
)
if ($WANDB_API_KEY -ne "") { $envArgs += "-e WANDB_API_KEY=$WANDB_API_KEY" }

$envString = $envArgs -join ' '

Write-Host "Starting docker run... this may take a while as models are downloaded"
docker run --rm --gpus all $envString -v ${PWD}:/app -w /app llm-pipeline python scripts/run_training.py

Write-Host "Training finished (if successful). Running evaluation inside container..."
docker run --rm --gpus all -v ${PWD}:/app -w /app llm-pipeline python scripts/evaluate_model.py

Write-Host "Evaluation complete. Check results/evaluation_metrics.json and results/comparison.md"
