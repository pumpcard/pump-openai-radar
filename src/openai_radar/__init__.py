"""Importable ``openai_radar`` package for this checkout.

Scanner code lives at the repository root (``cli.py``, ``client.py``, and the
rest) and under ``src/open_radar``. This module is what ``pip install -e .``
exposes, and it points ``__path__`` at both places.
"""

from pathlib import Path

_root = Path(__file__).resolve().parents[2]
_init = _root / "__init__.py"
exec(compile(_init.read_text(encoding="utf-8"), str(_init), "exec"), globals())

__path__ = [str(_root), str(_root / "src" / "open_radar")]
