```bash
#!/usr/bin/env bash

set -e

echo "🔧 Setting up environment (CUDA 13 / cu130)..."

# --- Optional system deps ---
if command -v apt-get &> /dev/null; then
    sudo apt-get update || true
    sudo apt-get install -y python3-pip python3-venv git || true
fi

# --- Git config (global) ---
git config --global user.email "sathya.pranav.deepak@gmail.com"
git config --global user.name "PranavDeepakSathya"

# --- Create venv ---
python3 -m venv .venv
source .venv/bin/activate

# --- Upgrade pip ---
pip install --upgrade pip

# --- Core deps ---
pip install numpy matplotlib networkx pyvis

# --- PyTorch (CUDA 13 / cu130) ---
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130

# --- Triton ---
pip install triton

# --- Sanity check ---
python - <<EOF
import torch
print("Torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
EOF

echo "✅ Setup complete"
echo "👉 Activate with: source .venv/bin/activate"
```
