"""Locations in the portable repository checkout, independent of the current directory."""
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = SRC_ROOT.parent
CONF_ROOT = REPO_ROOT / "conf"
