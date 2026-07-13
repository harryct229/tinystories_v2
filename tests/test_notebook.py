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


SFT_NOTEBOOK = Path(__file__).parent.parent / "notebooks" / "sft_colab.ipynb"


def test_sft_notebook_is_thin():
    cells = json.loads(SFT_NOTEBOOK.read_text(encoding="utf-8"))["cells"]
    code_cells = [c for c in cells if c["cell_type"] == "code"]
    assert 1 <= len(code_cells) <= 4
    source = "\n".join("".join(c["source"]) for c in code_cells)
    for forbidden in ("def ", "class ", "import torch", "for ", "while "):
        assert forbidden not in source, forbidden
    # Turnkey: the notebook invokes the one-command bootstrap (which itself runs
    # ts2-sft-data then ts2-sft --resume) rather than the stage directly.
    assert "scripts/sft_colab.py" in source


def test_sft_notebook_has_no_secrets_or_outputs():
    text = SFT_NOTEBOOK.read_text(encoding="utf-8")
    assert "hf_" not in text
    cells = json.loads(text)["cells"]
    assert all(not c.get("outputs") for c in cells)


PREF_NOTEBOOK = Path(__file__).parent.parent / "notebooks" / "pref_data_colab.ipynb"


def test_pref_data_notebook_is_thin():
    cells = json.loads(PREF_NOTEBOOK.read_text(encoding="utf-8"))["cells"]
    code_cells = [c for c in cells if c["cell_type"] == "code"]
    assert 1 <= len(code_cells) <= 4
    source = "\n".join("".join(c["source"]) for c in code_cells)
    for forbidden in ("def ", "class ", "import torch", "for ", "while "):
        assert forbidden not in source, forbidden
    assert "ts2-pref-data" in source
    assert "--resume" in source
    assert "[judge]" in source  # real Judge needs the transformers extra


def test_pref_data_notebook_has_no_secrets_or_outputs():
    text = PREF_NOTEBOOK.read_text(encoding="utf-8")
    assert "hf_" not in text  # no literal HF token prefixes
    cells = json.loads(text)["cells"]
    assert all(not c.get("outputs") for c in cells)


REWARD_NOTEBOOK = Path(__file__).parent.parent / "notebooks" / "reward_colab.ipynb"


def test_reward_notebook_is_thin():
    cells = json.loads(REWARD_NOTEBOOK.read_text(encoding="utf-8"))["cells"]
    code_cells = [c for c in cells if c["cell_type"] == "code"]
    assert 1 <= len(code_cells) <= 4
    source = "\n".join("".join(c["source"]) for c in code_cells)
    for forbidden in ("def ", "class ", "import torch", "for ", "while "):
        assert forbidden not in source, forbidden
    # Turnkey: the notebook invokes the one-command bootstrap (which itself
    # downloads the tokenizer + pairs, then runs ts2-reward --resume) rather
    # than the stage directly.
    assert "scripts/reward_colab.py" in source


def test_reward_notebook_has_no_secrets_or_outputs():
    text = REWARD_NOTEBOOK.read_text(encoding="utf-8")
    assert "hf_" not in text
    cells = json.loads(text)["cells"]
    assert all(not c.get("outputs") for c in cells)


EVAL_NOTEBOOK = Path(__file__).parent.parent / "notebooks" / "eval_colab.ipynb"


def test_eval_notebook_is_thin():
    cells = json.loads(EVAL_NOTEBOOK.read_text(encoding="utf-8"))["cells"]
    code_cells = [c for c in cells if c["cell_type"] == "code"]
    assert 1 <= len(code_cells) <= 4
    source = "\n".join("".join(c["source"]) for c in code_cells)
    for forbidden in ("def ", "class ", "import torch", "for ", "while "):
        assert forbidden not in source, forbidden
    # Turnkey: the notebook invokes the one-command bootstrap, not the stage.
    assert "scripts/eval_colab.py" in source


def test_eval_notebook_has_no_secrets_or_outputs():
    text = EVAL_NOTEBOOK.read_text(encoding="utf-8")
    assert "hf_" not in text
    cells = json.loads(text)["cells"]
    assert all(not c.get("outputs") for c in cells)


DPO_NOTEBOOK = Path(__file__).parent.parent / "notebooks" / "dpo_colab.ipynb"


def test_dpo_notebook_is_thin():
    cells = json.loads(DPO_NOTEBOOK.read_text(encoding="utf-8"))["cells"]
    code_cells = [c for c in cells if c["cell_type"] == "code"]
    assert 1 <= len(code_cells) <= 4
    source = "\n".join("".join(c["source"]) for c in code_cells)
    for forbidden in ("def ", "class ", "import torch", "for ", "while "):
        assert forbidden not in source, forbidden
    # Turnkey: the notebook invokes the one-command bootstrap (which itself
    # downloads the tokenizer + pairs, then runs ts2-dpo --resume) rather than
    # the stage directly.
    assert "scripts/dpo_colab.py" in source


def test_dpo_notebook_has_no_secrets_or_outputs():
    text = DPO_NOTEBOOK.read_text(encoding="utf-8")
    assert "hf_" not in text
    cells = json.loads(text)["cells"]
    assert all(not c.get("outputs") for c in cells)


GRPO_NOTEBOOK = Path(__file__).parent.parent / "notebooks" / "grpo_colab.ipynb"


def test_grpo_notebook_is_thin():
    cells = json.loads(GRPO_NOTEBOOK.read_text(encoding="utf-8"))["cells"]
    code_cells = [c for c in cells if c["cell_type"] == "code"]
    assert 1 <= len(code_cells) <= 4
    source = "\n".join("".join(c["source"]) for c in code_cells)
    for forbidden in ("def ", "class ", "import torch", "for ", "while "):
        assert forbidden not in source, forbidden
    # Turnkey: the notebook invokes the one-command bootstrap (which downloads the
    # tokenizer + pref split, then runs ts2-grpo --resume) rather than the stage.
    assert "scripts/grpo_colab.py" in source


def test_grpo_notebook_has_no_secrets_or_outputs():
    text = GRPO_NOTEBOOK.read_text(encoding="utf-8")
    assert "hf_" not in text
    cells = json.loads(text)["cells"]
    assert all(not c.get("outputs") for c in cells)
