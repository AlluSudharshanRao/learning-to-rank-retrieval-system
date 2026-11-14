"""Source-checkout entry point; supports Python isolated mode on HPC."""

import sys
from pathlib import Path

# -I removes automatic script-directory imports. Add only this checkout's source tree.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ltr_system.commands import faiss_main  # noqa: E402

if __name__ == "__main__":
    faiss_main()
