"""Source-checkout entry point; supports Python isolated mode."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ltr_system.benchmark_plots import main  # noqa: E402

if __name__ == "__main__":
    main()
