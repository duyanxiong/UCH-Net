"""Run from a source checkout without installing the package first."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from mid_prior_yolo.cli import main


if __name__ == "__main__":
    main()

