import os

from tinystories_v2.config import load_config, load_env


def test_load_config_parses_toml(tmp_path):
    cfg_file = tmp_path / "stage.toml"
    cfg_file.write_text('out_dir = "artifacts/x"\n\n[source]\nkind = "jsonl"\n', encoding="utf-8")
    assert load_config(cfg_file) == {"out_dir": "artifacts/x", "source": {"kind": "jsonl"}}


def test_load_env_sets_without_override_or_output(tmp_path, capsys, monkeypatch):
    env_file = tmp_path / ".env"
    env_file.write_text("NEW_KEY=fresh\nEXISTING_KEY=changed\n# comment\n", encoding="utf-8")
    monkeypatch.setenv("EXISTING_KEY", "original")
    monkeypatch.delenv("NEW_KEY", raising=False)
    load_env(env_file)
    assert os.environ["NEW_KEY"] == "fresh"
    assert os.environ["EXISTING_KEY"] == "original"
    assert capsys.readouterr() == ("", "")  # secret values must never be printed


def test_load_env_missing_file_is_noop(tmp_path):
    load_env(tmp_path / "absent.env")  # must not raise
