"""The Colab notebook must stay a thin wrapper: setup + stage invocation only.

Any Python logic (function/class definitions, torch imports, loops) belongs in
the package where it is reviewed and tested — never in notebook JSON.
"""

import json
from pathlib import Path

NOTEBOOK = Path(__file__).parent.parent / "notebooks" / "pretrain_colab.ipynb"


def test_notebook_is_thin():
    cells = json.loads(NOTEBOOK.read_text(encoding="utf-8"))["cells"]
    code_cells = [c for c in cells if c["cell_type"] == "code"]
    assert 1 <= len(code_cells) <= 4
    source = "\n".join("".join(c["source"]) for c in code_cells)
    for forbidden in ("def ", "class ", "import torch", "for ", "while "):
        assert forbidden not in source, forbidden
    assert "ts2-pretrain" in source
    assert "--resume" in source


def test_notebook_has_no_secrets_or_outputs():
    text = NOTEBOOK.read_text(encoding="utf-8")
    assert "hf_" not in text  # no literal HF token prefixes
    cells = json.loads(text)["cells"]
    assert all(not c.get("outputs") for c in cells)  # committed clean
