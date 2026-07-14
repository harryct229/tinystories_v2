"""Eval-stage resumability (issue 07 follow-up): completions are cached per
stage, judgments stream per pairing to append-only files, and an interrupted
eval — resumed in place or on a fresh out_dir via the hub target — finishes
with results identical to an uninterrupted run without re-judging completed
Scaffolds. Motivated by L4 preemptions killing ~50-min eval runs whole."""

import json
import shutil
from pathlib import Path

import pytest

from tinystories_v2 import eval as eval_stage
from tinystories_v2.data import run as data_run
from tinystories_v2.judge import SlotCoverageFakeJudge
from tinystories_v2.tokenizer import run as tokenizer_run

TOY_MODEL = {"vocab_size": 512, "d_model": 64, "n_layers": 2,
             "n_heads": 2, "context": 128, "ffn_hidden": 192}


def _prepare_inputs(base, fixture_path, make_init_checkpoint, stage_names):
    """A tokenizer, an eval split, and one toy checkpoint per stage name
    (mirrors test_eval_stage.py's harness; test files stay self-contained)."""
    data_dir, tok_dir = base / "data", base / "tok"
    data_run({
        "out_dir": str(data_dir), "max_extraction_failures": 0,
        "source": {"kind": "jsonl", "path": str(fixture_path)},
        "splits": {"seed": "fixture-v1", "pretrain": 0.1, "sft": 0.6,
                   "pref": 0.1, "eval": 0.2},
    })
    tokenizer_run({"out_dir": str(tok_dir), "corpus": [str(fixture_path)],
                   "text_field": "fable", "vocab_size": 512})
    tokenizer_path = tok_dir / "tokenizer.json"
    stages = []
    for name in stage_names:
        stage_dir = base / name
        make_init_checkpoint(stage_dir, TOY_MODEL, tokenizer_path)
        stages.append({"name": name, "local_dir": str(stage_dir)})
    return {
        "out_dir": str(base / "out"),
        "max_eval_scaffolds": 16,
        "sample_sheet_k": 3,
        "sync_every": 4,
        "data": {"eval_split": str(data_dir / "splits" / "eval.jsonl"),
                 "tokenizer": str(tokenizer_path)},
        "stages": stages,
        "sampling": {"max_new_tokens": 24, "temperature": 0.8, "top_p": 0.95,
                     "seed": 1337},
        "metrics": {"self_bleu_sample_size": 0, "self_bleu_seed": 0},
        "judge": {"kind": "fake_slot_coverage"},
    }


def make_generate_fn():
    """Deterministic, stage-distinct fables (the two toy checkpoints share a
    seed, so real generation would make every pair degenerate and the judge
    would never run). Stages are generated in config order, so the call index
    identifies the stage reproducibly across runs."""
    calls = {"n": 0}

    def gen(model, tokenizer, scaffolds, seeds, sampling, *, device="cpu"):
        stage_ix = calls["n"]
        calls["n"] += 1
        return [f"stage {stage_ix} fable: the {s.trait} {s.character} in "
                f"{s.setting} learned {s.moral}" for s in scaffolds]

    return gen


class CountingJudge:
    """SlotCoverage verdicts with a compare-call counter; optionally raises
    KeyboardInterrupt after max_calls to simulate a preemption mid-judging."""

    judge_id = SlotCoverageFakeJudge().judge_id

    def __init__(self, max_calls: int | None = None) -> None:
        self.inner = SlotCoverageFakeJudge()
        self.calls = 0
        self.max_calls = max_calls

    def compare(self, scaffold, fable_a, fable_b):
        if self.max_calls is not None and self.calls >= self.max_calls:
            raise KeyboardInterrupt("simulated preemption")
        self.calls += 1
        return self.inner.compare(scaffold, fable_a, fable_b)


def _comparable(results: dict) -> dict:
    return {key: results[key] for key in
            ("eval_judge_id", "n_scaffolds", "stages", "win_rates", "metrics")}


def test_interrupted_eval_resumes_in_place_to_identical_results(
        tmp_path, fixture_path, make_init_checkpoint):
    config = _prepare_inputs(tmp_path, fixture_path, make_init_checkpoint,
                             ["base", "sft"])
    reference_cfg = dict(config, out_dir=str(tmp_path / "ref_out"))
    ref_judge = CountingJudge()
    reference = eval_stage.run(reference_cfg,
                               judge_factory=lambda cfg: ref_judge,
                               generate_fn=make_generate_fn())

    dying = CountingJudge(max_calls=6)
    with pytest.raises(KeyboardInterrupt):
        eval_stage.run(config, judge_factory=lambda cfg: dying,
                       generate_fn=make_generate_fn())
    out = Path(config["out_dir"])
    assert (out / "completions").exists()
    assert (out / "judgments").exists()

    resumed_judge = CountingJudge()
    results = eval_stage.run(config, resume=True,
                             judge_factory=lambda cfg: resumed_judge,
                             generate_fn=make_generate_fn())
    assert _comparable(results) == _comparable(reference)
    # Completed Scaffolds were not re-judged: the resumed run made strictly
    # fewer compare calls than the uninterrupted reference.
    assert resumed_judge.calls < ref_judge.calls
    assert resumed_judge.calls + dying.calls >= ref_judge.calls


def test_fresh_out_dir_resumes_via_hub_mirror(tmp_path, fixture_path,
                                              make_init_checkpoint):
    config = _prepare_inputs(tmp_path, fixture_path, make_init_checkpoint,
                             ["base", "sft"])
    reference_cfg = dict(config, out_dir=str(tmp_path / "ref_out"))
    reference = eval_stage.run(reference_cfg,
                               judge_factory=lambda cfg: CountingJudge(),
                               generate_fn=make_generate_fn())

    mirror = tmp_path / "mirror"
    config["hub"] = {"target": str(mirror)}
    with pytest.raises(KeyboardInterrupt):
        # 10 calls = 5 judged Scaffolds: enough to cross the sync_every=4
        # threshold so partial judgments reach the mirror before "preemption".
        eval_stage.run(config,
                       judge_factory=lambda cfg: CountingJudge(max_calls=10),
                       generate_fn=make_generate_fn())
    assert (mirror / "judgments").exists()  # partial state reached the mirror

    shutil.rmtree(config["out_dir"])  # a fresh Colab VM has no local artifact
    resumed_judge = CountingJudge()
    results = eval_stage.run(config, resume=True,
                             judge_factory=lambda cfg: resumed_judge,
                             generate_fn=make_generate_fn())
    assert _comparable(results) == _comparable(reference)
    assert resumed_judge.calls < 2 * 16  # skipped the mirrored judgments


def test_resume_rejects_a_different_judge_identity(tmp_path, fixture_path,
                                                   make_init_checkpoint):
    config = _prepare_inputs(tmp_path, fixture_path, make_init_checkpoint,
                             ["base", "sft"])
    with pytest.raises(KeyboardInterrupt):
        eval_stage.run(config,
                       judge_factory=lambda cfg: CountingJudge(max_calls=6),
                       generate_fn=make_generate_fn())

    class OtherJudge(CountingJudge):
        judge_id = "fake:other-judge-v1"

    with pytest.raises(ValueError, match="judge"):
        eval_stage.run(config, resume=True,
                       judge_factory=lambda cfg: OtherJudge(),
                       generate_fn=make_generate_fn())


def test_torn_judgment_line_is_dropped_on_resume(tmp_path, fixture_path,
                                                 make_init_checkpoint):
    config = _prepare_inputs(tmp_path, fixture_path, make_init_checkpoint,
                             ["base", "sft"])
    reference_cfg = dict(config, out_dir=str(tmp_path / "ref_out"))
    reference = eval_stage.run(reference_cfg,
                               judge_factory=lambda cfg: CountingJudge(),
                               generate_fn=make_generate_fn())

    with pytest.raises(KeyboardInterrupt):
        eval_stage.run(config,
                       judge_factory=lambda cfg: CountingJudge(max_calls=6),
                       generate_fn=make_generate_fn())
    logs = list((Path(config["out_dir"]) / "judgments").glob("*.jsonl"))
    assert logs
    with logs[0].open("a", encoding="utf-8") as f:
        f.write('{"prompt_hash": "torn-mid-')  # killed mid-append

    results = eval_stage.run(config, resume=True,
                             judge_factory=lambda cfg: CountingJudge(),
                               generate_fn=make_generate_fn())
    assert _comparable(results) == _comparable(reference)


def test_fresh_run_clears_stale_resume_state(tmp_path, fixture_path,
                                             make_init_checkpoint):
    config = _prepare_inputs(tmp_path, fixture_path, make_init_checkpoint,
                             ["base", "sft"])
    with pytest.raises(KeyboardInterrupt):
        eval_stage.run(config,
                       judge_factory=lambda cfg: CountingJudge(max_calls=6),
                       generate_fn=make_generate_fn())

    fresh_judge = CountingJudge()
    eval_stage.run(config, judge_factory=lambda cfg: fresh_judge,
                   generate_fn=make_generate_fn())  # no resume
    # A non-resume run must not silently reuse the stale judgments: it judges
    # everything again from scratch.
    reference_cfg = dict(config, out_dir=str(tmp_path / "ref_out"))
    ref_judge = CountingJudge()
    eval_stage.run(reference_cfg, judge_factory=lambda cfg: ref_judge,
                               generate_fn=make_generate_fn())
    assert fresh_judge.calls == ref_judge.calls
