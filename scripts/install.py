#!/usr/bin/env python3
"""
Cross-platform installer helper.
Usage:
  python scripts/install.py --venv .venv [--force-gpu]
"""
import argparse
import subprocess
import sys
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--venv", default=".venv")
parser.add_argument("--force-gpu", action="store_true")
args = parser.parse_args()

venv = Path(args.venv)
subprocess.check_call([sys.executable, "-m", "venv", str(venv)])
activate = venv / ("Scripts" if sys.platform == "win32" else "bin") / ("Activate.ps1" if sys.platform == "win32" else "activate")
print(f"Created venv at {venv}. Activate it before running commands: {activate}")

def detect_gpu():
    try:
        subprocess.check_call(["nvidia-smi"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    except Exception:
        return False

print("Upgrading pip and build-tools...")
subprocess.check_call([str(venv / ("Scripts" if sys.platform == "win32" else "bin") / "python"), "-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel"])

if args.force_gpu or detect_gpu():
    print("GPU detected or forced. Installing GPU requirements...")
    python_exe = str(venv / ("Scripts" if sys.platform == "win32" else "bin") / "python")
    subprocess.check_call([python_exe, "-m", "pip", "install", "--find-links", "https://storage.googleapis.com/jax-releases/jax_cuda_releases.html", "-r", "requirements-gpu.txt"])
    subprocess.check_call([python_exe, "-m", "pip", "install", "-e", "."])
else:
    print("No GPU detected. Installing CPU/dev requirements...")
    python_exe = str(venv / ("Scripts" if sys.platform == "win32" else "bin") / "python")
    subprocess.check_call([python_exe, "-m", "pip", "install", "-r", "requirements.txt"])
    subprocess.check_call([python_exe, "-m", "pip", "install", "-e", "."])

print("Done. Activate the venv and run the examples/ or CLI.")

