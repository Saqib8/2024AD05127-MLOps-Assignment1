"""Ensure the project root is importable during test collection (CI safety net)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
