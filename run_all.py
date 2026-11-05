"""Workspace-level entry point for the edge-migration project."""
from pathlib import Path
import runpy
import sys

project = Path(__file__).parent / "edge_migration"
sys.path.insert(0, str(project))
runpy.run_path(str(project / "run_all.py"), run_name="__main__")
