"""Relocatable paths shared by the development tree and submission package.

In development this package is ``work/agent``.  In the submitted archive it is
``agent`` at the package root.  The parent of this module's package is therefore
the stable runtime root in both layouts.
"""
from pathlib import Path


WORK_DIR = Path(__file__).resolve().parents[1]
REPO_DIR = WORK_DIR.parent
PROCESSED_DIR = WORK_DIR / "processed_data"
OUTPUT_DIR = WORK_DIR / "output"
ENV_FILE = WORK_DIR / ".env"

