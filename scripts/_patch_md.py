"""Patch one markdown cell of an executed .ipynb, in place, preserving outputs.

Why this exists: the notebooks here are NOT jupytext-paired (no `jupytext`
key in notebook metadata), so `jupytext --sync` refuses to push a prose edit
from the .py into the .ipynb. Re-running the notebook just to fix a paragraph
would also throw away hours of executed output. This edits the one cell.

    python scripts/_patch_md.py <notebook> <match-substring> <new-md-file>
"""
from __future__ import annotations

import json
import sys
from pathlib import Path


def main() -> int:
    nb_path, needle, new_md = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3])
    nb = json.loads(nb_path.read_text(encoding="utf-8"))

    body = new_md.read_text(encoding="utf-8").rstrip("\n")
    # nbformat stores source as a list of lines, each keeping its trailing \n
    # except the last -- match that shape so the diff stays minimal.
    lines = body.split("\n")
    source = [f"{ln}\n" for ln in lines[:-1]] + [lines[-1]]

    hits = [
        c for c in nb["cells"]
        if c["cell_type"] == "markdown" and needle in "".join(c["source"])
    ]
    if len(hits) != 1:
        print(f"ERROR: needle matched {len(hits)} markdown cells, need exactly 1")
        return 1

    hits[0]["source"] = source
    nb_path.write_text(json.dumps(nb, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"patched 1 markdown cell in {nb_path.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
