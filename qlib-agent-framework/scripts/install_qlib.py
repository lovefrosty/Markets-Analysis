"""Utility script to install PyQLib with required dependencies."""

import subprocess
import sys
from pathlib import Path

REQUIREMENTS_FILE = Path(__file__).resolve().parents[1] / "requirements.txt"


def run(cmd):
    print(f"[install_qlib] Running: {' '.join(cmd)}")
    subprocess.check_call(cmd)


def main():
    if not REQUIREMENTS_FILE.exists():
        raise FileNotFoundError(f"Missing requirements file: {REQUIREMENTS_FILE}")

    run([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])
    run([sys.executable, "-m", "pip", "install", "-r", str(REQUIREMENTS_FILE)])


if __name__ == "__main__":
    main()
