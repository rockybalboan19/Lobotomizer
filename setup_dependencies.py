#!/usr/bin/env python3
"""Install package dependencies for the Brain Surgeon project."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REQ = ROOT / "requirements.txt"


def run(cmd):
    print(f"$ {' '.join(cmd)}")
    subprocess.check_call(cmd)


def main():
    if not REQ.exists():
        raise FileNotFoundError(f"requirements.txt not found at {REQ}")

    run([sys.executable, "-m", "pip", "install", "-r", str(REQ)])

    # Torch-Pruning is not used by the final published version of this project.
    # If a future branch wants it, clone it into a local, gitignored directory.
    local_pruning_dir = ROOT / ".local_deps" / "Torch-Pruning"
    if os.environ.get("INSTALL_TORCH_PRUNING", "0").lower() in {"1", "true", "yes"}:
        local_pruning_dir.parent.mkdir(parents=True, exist_ok=True)
        if not local_pruning_dir.exists():
            run([
                "git",
                "clone",
                "https://github.com/VainF/Torch-Pruning.git",
                str(local_pruning_dir),
            ])

    print("\nDependency installation complete.")


if __name__ == "__main__":
    main()
