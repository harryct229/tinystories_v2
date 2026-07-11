import sys
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType

import pytest

from tinystories_v2.config import load_config
from tinystories_v2.judge import (
    PositionBiasedFakeJudge,
    SlotCoverageFakeJudge,
    TransformersJudge,
    Verdict,
    build_judge,
    render_rubric_prompt,
)
from tinystories_v2.slots import extract_slots

CONFIG_DIR = Path(__file__).parents[1] / "configs"
PROMPT_TOKEN_IDS = (11, 12, 13)
COMPLETION_TOKEN_IDS = (21, 22)

PRODUCTION_JUDGE_CASES = [
    pytest.param(
        "judge_l4.toml",
        "Qwen/Qwen3-8B",
        False,
        (
            "transformers:Qwen/Qwen3-8B;precision=fp16;thinking=false;"
            "rubric=fable-pairwise-v1"
        ),
        {"enable_thinking": False},
        id="l4",
    ),
    pytest.param(
        "judge_t4.toml",
        "Qwen/Qwen3-4B-Instruct-2507",
        None,
        (
            "transformers:Qwen/Qwen3-4B-Instruct-2507;precision=fp16;"
            "thinking=default;rubric=fable-pairwise-v1"
        ),
        {},
        id="t4",
    ),
]


@dataclass(frozen=True)
class _FakeDType:
    name: str


@dataclass(frozen=True)
class _FakeTensor:
    token_ids: tuple[int, ...]

    @property
    def shape(self) -> tuple[int, int]:
        return (1, len(self.token_ids))


@dataclass(frozen=True)
class _GeneratedRow:
    token_ids: tuple[int, ...]

    def __getitem__(self, index):
        return self.token_ids[index]


@dataclass
class _BoundaryCalls:
    tokenizer_loads: list[str] = field(default_factory=list)
    model_loads: list[tuple[str, _FakeDType]] = field(
        default_factory=list
    )
    model_devices: list[str] = field(default_factory=list)
    model_eval_count: int = 0
    template_calls: list[
        tuple[list[dict[str, str]], dict[str, object]]
    ] = field(default_factory=list)
    input_devices: list[str] = field(default_factory=list)
    generation_calls: list[dict[str, object]] = field(
        default_factory=list
    )
    decode_calls: list[dict[str, object]] = field(default_factory=list)
    inference_entries: int = 0
    inference_exits: int = 0


class _FakeBatchEncoding(dict[str, _FakeTensor]):
    def __init__(self, calls: _BoundaryCalls) -> None:
        super().__init__(
            input_ids=_FakeTensor(PROMPT_TOKEN_IDS),
            attention_mask=_FakeTensor((1,) * len(PROMPT_TOKEN_IDS)),
        )
        self._calls = calls

    def to(self, device: str) -> "_FakeBatchEncoding":
        self._calls.input_devices.append(device)
        return self


def _install_transformers_boundary(monkeypatch):
    calls = _BoundaryCalls()
    float16 = _FakeDType("float16")
    bfloat16 = _FakeDType("bfloat16")

    class _InferenceMode:
        def __enter__(self):
            calls.inference_entries += 1
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            calls.inference_exits += 1
            return False

    class _Tokenizer:
        eos_token_id = 0

        def apply_chat_template(
            self,
            messages,
            *,
            tokenize,
            add_generation_prompt,
            return_dict,
            return_tensors,
            **template_options,
        ):
            calls.template_calls.append(
                (
                    [dict(message) for message in messages],
                    {
                        "tokenize": tokenize,
                        "add_generation_prompt": add_generation_prompt,
                        "return_dict": return_dict,
                        "return_tensors": return_tensors,
                        **template_options,
                    },
                )
            )
            return _FakeBatchEncoding(calls)

        def decode(self, token_ids, *, skip_special_tokens):
            calls.decode_calls.append(
                {
                    "token_ids": tuple(token_ids),
                    "skip_special_tokens": skip_special_tokens,
                }
            )
            return "\nVerdict: B.\n"

    class _Model:
        def to(self, device):
            calls.model_devices.append(device)
            return self

        def eval(self):
            calls.model_eval_count += 1
            return self

        def generate(
            self,
            *,
            input_ids,
            attention_mask,
            max_new_tokens,
            do_sample,
            pad_token_id,
        ):
            calls.generation_calls.append(
                {
                    "input_ids": input_ids.token_ids,
                    "attention_mask": attention_mask.token_ids,
                    "max_new_tokens": max_new_tokens,
                    "do_sample": do_sample,
                    "pad_token_id": pad_token_id,
                }
            )
            return [
                _GeneratedRow(PROMPT_TOKEN_IDS + COMPLETION_TOKEN_IDS)
            ]

    class _AutoTokenizer:
        @classmethod
        def from_pretrained(cls, model_id):
            calls.tokenizer_loads.append(model_id)
            return _Tokenizer()

    class _AutoModelForCausalLM:
        @classmethod
        def from_pretrained(cls, model_id, *, torch_dtype):
            calls.model_loads.append((model_id, torch_dtype))
            return _Model()

    torch_module = ModuleType("torch")
    torch_module.float16 = float16
    torch_module.bfloat16 = bfloat16
    torch_module.inference_mode = _InferenceMode
    transformers_module = ModuleType("transformers")
    transformers_module.AutoTokenizer = _AutoTokenizer
    transformers_module.AutoModelForCausalLM = _AutoModelForCausalLM
    monkeypatch.setitem(sys.modules, "torch", torch_module)
    monkeypatch.setitem(sys.modules, "transformers", transformers_module)
    return calls, float16


@pytest.mark.parametrize(
    (
        "filename",
        "model_id",
        "enable_thinking",
        "expected_judge_id",
        "_expected_template_options",
    ),
    PRODUCTION_JUDGE_CASES,
)
def test_real_configs_select_one_lazy_transformers_path(
    filename,
    model_id,
    enable_thinking,
    expected_judge_id,
    _expected_template_options,
):
    config = load_config(CONFIG_DIR / filename)["judge"]
    judge = build_judge(config)

    assert type(judge) is TransformersJudge
    assert judge.model_id == model_id
    assert judge.precision == "fp16"
    assert judge.device == "cuda"
    assert judge.enable_thinking is enable_thinking
    assert judge.judge_id == expected_judge_id


@pytest.mark.parametrize("missing_module", ["torch", "transformers"])
def test_transformers_judge_reports_each_missing_optional_dependency(
    missing_module,
    monkeypatch,
    fixture_records,
):
    for module_name in ("torch", "transformers"):
        module = None if module_name == missing_module else ModuleType(
            module_name
        )
        monkeypatch.setitem(sys.modules, module_name, module)
    judge = TransformersJudge(
        model_id="offline/model",
        precision="fp16",
        device="cuda",
    )
    scaffold = extract_slots(fixture_records[0]["prompt"])

    with pytest.raises(RuntimeError) as exc_info:
        judge.compare(scaffold, "Candidate A.", "Candidate B.")

    assert str(exc_info.value) == (
        "real Judge dependencies are missing; "
        "install with: uv pip install -e '.[judge]'"
    )
    assert isinstance(exc_info.value.__cause__, ImportError)


@pytest.mark.parametrize(
    (
        "filename",
        "model_id",
        "_enable_thinking",
        "expected_judge_id",
        "expected_template_options",
    ),
    PRODUCTION_JUDGE_CASES,
)
def test_transformers_judge_uses_lazy_cached_backend_offline(
    filename,
    model_id,
    _enable_thinking,
    expected_judge_id,
    expected_template_options,
    monkeypatch,
    fixture_records,
):
    calls, float16 = _install_transformers_boundary(monkeypatch)
    config = load_config(CONFIG_DIR / filename)["judge"]

    judge = build_judge(config)

    assert type(judge) is TransformersJudge
    assert judge.judge_id == expected_judge_id
    assert calls.tokenizer_loads == []
    assert calls.model_loads == []

    scaffold = extract_slots(fixture_records[0]["prompt"])
    candidate_a = "A patient fox shared a loaf."
    candidate_b = "A swift hare kept every crumb."

    first_verdict = judge.compare(scaffold, candidate_a, candidate_b)
    second_verdict = judge.compare(scaffold, candidate_a, candidate_b)

    assert first_verdict is Verdict.B
    assert second_verdict is Verdict.B
    assert calls.tokenizer_loads == [model_id]
    assert calls.model_loads == [(model_id, float16)]
    assert calls.model_devices == ["cuda"]
    assert calls.model_eval_count == 1

    expected_messages = [
        {
            "role": "system",
            "content": (
                "Follow the pairwise Fable rubric and return only its "
                "requested verdict label."
            ),
        },
        {
            "role": "user",
            "content": render_rubric_prompt(
                scaffold,
                candidate_a,
                candidate_b,
            ),
        },
    ]
    expected_template_call = (
        expected_messages,
        {
            "tokenize": True,
            "add_generation_prompt": True,
            "return_dict": True,
            "return_tensors": "pt",
            **expected_template_options,
        },
    )
    assert calls.template_calls == [
        expected_template_call,
        expected_template_call,
    ]
    assert calls.input_devices == ["cuda", "cuda"]

    expected_generation_call = {
        "input_ids": PROMPT_TOKEN_IDS,
        "attention_mask": (1, 1, 1),
        "max_new_tokens": 4,
        "do_sample": False,
        "pad_token_id": 0,
    }
    assert calls.generation_calls == [
        expected_generation_call,
        expected_generation_call,
    ]
    assert calls.inference_entries == 2
    assert calls.inference_exits == 2
    expected_decode_call = {
        "token_ids": COMPLETION_TOKEN_IDS,
        "skip_special_tokens": True,
    }
    assert calls.decode_calls == [
        expected_decode_call,
        expected_decode_call,
    ]


@pytest.mark.parametrize(
    ("kind", "expected_type"),
    [
        ("fake_slot_coverage", SlotCoverageFakeJudge),
        ("fake_position_biased", PositionBiasedFakeJudge),
    ],
)
def test_factory_selects_offline_fakes(kind, expected_type):
    assert type(build_judge({"kind": kind})) is expected_type


def test_factory_rejects_unknown_kind():
    with pytest.raises(ValueError, match="unknown Judge kind"):
        build_judge({"kind": "remote_api"})


def test_factory_rejects_unknown_precision_without_loading_model():
    with pytest.raises(ValueError, match="precision"):
        build_judge(
            {
                "kind": "transformers",
                "model_id": "Qwen/Qwen3-8B",
                "precision": "int8",
                "device": "cuda",
            }
        )
