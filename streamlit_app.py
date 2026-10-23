"""Entry point for local runs and Streamlit Community Cloud.

Community Cloud installs third-party packages from requirements.txt and then
runs this file. It does not install this repository as a package, so ``src``
has to be importable before ``smart_clearing`` is loaded.
"""

from __future__ import annotations

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from smart_clearing.app import main  # noqa: E402

main()
