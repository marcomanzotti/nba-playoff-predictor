"""Avvia la demo in locale con un doppio click / `python3 run_demo.py`.

app.py e' un'app Streamlit: NON va lanciata con `python3 app.py` (Streamlit non
parte da solo, serve il comando `streamlit run`). Questo script lo fa per te,
usando sempre l'interprete del venv del progetto (.venv), a prescindere dalla
cartella da cui lo esegui.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
VENV_PY = ROOT / ".venv" / "bin" / "python3"
APP = ROOT / "app.py"


def main() -> None:
    python = VENV_PY if VENV_PY.exists() else sys.executable
    if not VENV_PY.exists():
        print(f"[avviso] {VENV_PY} non trovato, uso {sys.executable}")
    print(f"Avvio la demo con: {python} -m streamlit run {APP}")
    subprocess.run([str(python), "-m", "streamlit", "run", str(APP)], cwd=ROOT)


if __name__ == "__main__":
    main()
