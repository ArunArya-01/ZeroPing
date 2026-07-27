#!/usr/bin/env python3
"""Build the featured dataset from scratch using the AeroTwin pipeline."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aerotwin.engine.build_featured_dataset import main

if __name__ == "__main__":
    main()
