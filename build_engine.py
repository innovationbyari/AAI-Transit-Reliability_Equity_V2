"""Regenerate transit_ai/engine.py from the notebook.

The notebook is the source of truth. Run this after editing it:

    python tools/build_engine.py            # folder layout
    python build_engine.py                  # everything in one folder
    python build_engine.py My_notebook.ipynb   # a specific notebook

Every code cell from SETUP 3 (LLM client) through E4 (outputs) is copied in order. Colab-only cells are left out:
package install, the local Ollama server, the Gradio dashboard and launch, Google Drive, and the Results summary.
IPython display calls are routed to a no-op shim so the module runs outside Jupyter.
"""
import json
import pathlib
import re
import sys

HERE = pathlib.Path(__file__).resolve().parent
# Works with the folder layout (tools/, notebooks/, transit_ai/) and with every file in one folder.
ROOT = HERE.parent if HERE.name == "tools" else HERE


def _find_notebook():
    if len(sys.argv) > 1:
        return pathlib.Path(sys.argv[1]).resolve()
    found = sorted([*ROOT.glob("notebooks/*.ipynb"), *ROOT.glob("*.ipynb")], key=lambda p: p.stat().st_mtime)
    if not found:
        raise SystemExit("No notebook found: pass its path, e.g. python build_engine.py Transit_v6.ipynb")
    return found[-1]                                   # the most recently changed notebook


NOTEBOOK = _find_notebook()
OUT = (ROOT / "transit_ai" / "engine.py") if (ROOT / "transit_ai").is_dir() else (ROOT / "engine.py")

# Colab-only or display-only cells: package install, local Ollama, the architecture picture, the Gradio dashboard and
# launch, Google Drive, and the Results summary.
SKIP_TITLES = ("SETUP 1", "SETUP 2", "System architecture diagram", "E5 ·", "Results summary")
# Plain helper functions that live in the Gradio cell but are also used by export_outputs() and the Streamlit app.
KEEP_FROM_SKIPPED = [("Sections C + D helpers", "def h_load_roster(")]

HEADER = '''# ruff: noqa
# fmt: off
"""GENERATED FILE: do not edit by hand. Edit the notebook, then run `python tools/build_engine.py`.

Engine for the Streamlit app: every non-UI code cell of
notebooks/Transit_Agentic_AI_Submission_v5_with_evaluation.ipynb, in notebook order.
"""
import os as _os
import matplotlib as _matplotlib
_matplotlib.use("Agg")                     # no screen on a server: figures render off-screen
_os.environ.setdefault("MPLBACKEND", "Agg")
from transit_ai._ipython_shim import display, HTML, Markdown, clear_output  # noqa: F401
'''

FOOTER = '''

# ---- end of notebook cells ----
import matplotlib.pyplot as _plt
_plt.close("all")                          # free figures drawn during setup
'''


def main():
    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    parts, used = [HEADER], []
    for i, cell in enumerate(nb["cells"]):
        if cell["cell_type"] != "code":
            continue
        src = "".join(cell["source"])
        title = src.split("\n", 1)[0].replace("#@title", "").strip()
        if any(t in title for t in SKIP_TITLES):
            for start_marker, end_marker in KEEP_FROM_SKIPPED:
                if start_marker in src and end_marker in src:
                    a = src.rfind("\n", 0, src.index(start_marker)) + 1
                    kept = src[a:src.index(end_marker)].rstrip()
                    parts.append(f"\n\n# {'=' * 100}\n# helpers kept from notebook cell {i}: {title}\n# {'=' * 100}\n" + kept)
            continue
        lines = []
        for line in src.split("\n"):
            if re.match(r"\s*[!%]", line):            # shell or magic line
                line = re.sub(r"^(\s*).*$", r"\1pass  # (notebook-only line removed)", line)
            line = line.replace("from IPython.display import", "from transit_ai._ipython_shim import")
            lines.append(line)
        parts.append(f"\n\n# {'=' * 100}\n# notebook cell {i}: {title}\n# {'=' * 100}\n" + "\n".join(lines))
        used.append(i)
    parts.append(FOOTER)
    text = "".join(parts)
    if (ROOT / "transit_ai").is_dir():
        shim_import = "from transit_ai._ipython_shim import"
    else:
        shim_import = "from _ipython_shim import"
        text = text.replace("from transit_ai._ipython_shim import", shim_import)
    OUT.write_text(text, encoding="utf-8")
    print(f"Wrote {OUT.relative_to(ROOT)} from {NOTEBOOK.name}: {len(used)} notebook cells ({used[0]}..{used[-1]}).")


if __name__ == "__main__":
    main()
