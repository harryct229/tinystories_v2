import json

import numpy as np
import pytest
import torch
from tokenizers import Tokenizer

from tinystories_v2.pack import get_batch, load_packed, pack_split
from tinystories_v2.slots import SLOT_SPECIAL_TOKENS
from tinystories_v2.tokenizer import run as run_tokenizer

END_TOKEN = SLOT_SPECIAL_TOKENS[-1]


@pytest.fixture(scope="module")
def tokenizer_path(tmp_path_factory, fixture_path):
    out = tmp_path_factory.mktemp("tok")
    run_tokenizer({
        "out_dir": str(out), "corpus": [str(fixture_path)],
        "text_field": "fable", "vocab_size": 512,
    })
    return out / "tokenizer.json"


@pytest.fixture(scope="module")
def packed(tmp_path_factory, fixture_path, tokenizer_path):
    out = tmp_path_factory.mktemp("packed") / "pretrain.bin"
    manifest = pack_split(fixture_path, tokenizer_path, out)
    return out, manifest


def test_manifest_documents_dtype_and_shape(packed, fixture_records):
    out, manifest = packed
    assert manifest["dtype"] == "uint16"
    assert manifest["n_docs"] == len(fixture_records)
    assert manifest["n_tokens"] == out.stat().st_size // 2  # uint16 = 2 bytes
    on_disk = json.loads((out.parent / "pretrain.bin.json").read_text(encoding="utf-8"))
    assert on_disk == manifest


def test_roundtrip_first_fable_against_tokenizer(packed, fixture_records, tokenizer_path):
    out, manifest = packed
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    data = load_packed(out)
    end_id = manifest["end_id"]
    assert end_id == tokenizer.token_to_id(END_TOKEN)
    first_end = int(np.argmax(data == end_id))
    decoded = tokenizer.decode(data[:first_end].tolist())
    assert decoded == fixture_records[0]["fable"]


def test_every_doc_is_end_separated(packed, fixture_records):
    out, manifest = packed
    data = load_packed(out)
    assert int((data == manifest["end_id"]).sum()) == len(fixture_records)
    assert int(data[-1]) == manifest["end_id"]


def test_get_batch_shapes_and_shift(packed):
    out, _ = packed
    data = load_packed(out)
    x, y = get_batch(data, 4, 32, seed=1, step=0, micro_step=0)
    assert x.shape == y.shape == (4, 32)
    assert x.dtype == y.dtype == torch.int64
    assert torch.equal(x[:, 1:], y[:, :-1])  # y is x shifted left by one


def test_get_batch_is_pure_function_of_seed_step_microstep(packed):
    out, _ = packed
    data = load_packed(out)
    a = get_batch(data, 4, 32, seed=1, step=5, micro_step=2)
    b = get_batch(data, 4, 32, seed=1, step=5, micro_step=2)
    c = get_batch(data, 4, 32, seed=1, step=5, micro_step=3)
    d = get_batch(data, 4, 32, seed=2, step=5, micro_step=2)
    assert torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])
    assert not torch.equal(a[0], c[0])
    assert not torch.equal(a[0], d[0])
