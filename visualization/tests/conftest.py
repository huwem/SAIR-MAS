import sys
from pathlib import Path

# Ensure the repository root is first on sys.path so that top-level packages
# (e.g. `scripts`, `mas_topo`) are importable from tests/ regardless of pytest's
# default import path ordering.
ROOT = Path(__file__).parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
