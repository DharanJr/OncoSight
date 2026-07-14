"""
Checks whether PyTorch can see your GPU and reports why, if not.

Run this BEFORE starting training — it takes 30 seconds and saves you from
discovering an hour into training that you've been running on CPU.

Usage:
    python -m src.utils.check_environment
"""

import subprocess
import sys


def check_nvidia_driver():
    """nvidia-smi comes with the NVIDIA driver, not with PyTorch — if this
    fails, PyTorch cannot use the GPU no matter what you pip install."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0 and result.stdout.strip():
            print("[OK] NVIDIA driver found. GPU(s) reported by nvidia-smi:")
            for line in result.stdout.strip().splitlines():
                print(f"       {line}")
            return True
        else:
            print("[FAIL] nvidia-smi ran but returned no GPU info.")
            return False
    except FileNotFoundError:
        print(
            "[FAIL] 'nvidia-smi' not found — either you don't have an NVIDIA "
            "GPU, or the driver isn't installed.\n"
            "       If you DO have an NVIDIA GPU: install the driver from "
            "https://www.nvidia.com/Download/index.aspx and reboot.\n"
            "       If you don't have an NVIDIA GPU (e.g. laptop with only "
            "integrated/AMD/Apple graphics): training on CPU is your only "
            "local option — see the note at the bottom of this output."
        )
        return False
    except subprocess.TimeoutExpired:
        print("[FAIL] nvidia-smi timed out.")
        return False


def check_pytorch_cuda():
    try:
        import torch
    except ImportError:
        print("[FAIL] PyTorch is not installed. Run: pip install -r requirements.txt")
        return

    print(f"\n[INFO] PyTorch version: {torch.__version__}")
    print(f"[INFO] PyTorch built with CUDA support: {torch.backends.cuda.is_built()}")

    if torch.cuda.is_available():
        print(f"[OK] torch.cuda.is_available() = True")
        print(f"[OK] CUDA version PyTorch is using: {torch.version.cuda}")
        print(f"[OK] Number of visible GPUs: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            name = torch.cuda.get_device_name(i)
            mem_gb = torch.cuda.get_device_properties(i).total_memory / (1024 ** 3)
            print(f"       GPU {i}: {name} ({mem_gb:.1f} GB)")
        print("\nYou're good — src/config.py has DEVICE='cuda' and train.py will use it automatically.")
    else:
        print("[FAIL] torch.cuda.is_available() = False")
        print(
            "\nMost common cause: you installed the CPU-only build of PyTorch.\n"
            "`pip install torch` from requirements.txt grabs whatever build pip\n"
            "resolves to by default, which is often CPU-only on Windows.\n\n"
            "Fix — uninstall and reinstall a CUDA-enabled build:\n\n"
            "    pip uninstall torch torchvision\n"
            "    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121\n\n"
            "(cu121 = CUDA 12.1 build; check https://pytorch.org/get-started/locally/\n"
            " for the exact command matching your installed CUDA driver version,\n"
            " shown by nvidia-smi above under 'CUDA Version'.)"
        )


def main():
    print("=" * 60)
    print("GPU / environment check")
    print("=" * 60)
    driver_ok = check_nvidia_driver()
    check_pytorch_cuda()

    print("\n" + "=" * 60)
    if not driver_ok:
        print(
            "No NVIDIA GPU detected on this machine. train.py will still run\n"
            "on CPU — it falls back automatically — but ResNet50/EfficientNet\n"
            "training will be considerably slower (hours instead of minutes\n"
            "per epoch, depending on dataset size). Options:\n"
            "  1. Train anyway on CPU (fine for the CNN baseline; slow for the others)\n"
            "  2. Use a free GPU notebook (Google Colab, Kaggle Notebooks)\n"
            "  3. Reduce EPOCHS / BATCH_SIZE in src/config.py to make CPU runs shorter"
        )
    print("=" * 60)


if __name__ == "__main__":
    main()