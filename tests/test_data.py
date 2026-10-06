"""Offline tests for asr_robustness.data (synthetic audio; no network, no real dataset)."""

import io
import sys
import types

import numpy as np
import pytest
import soundfile as sf

from asr_robustness import data
from asr_robustness.data import (
    convert_record,
    decode_audio,
    load_librispeech_subset,
    select_indices,
)

SR = 16000


def encode(samples, fmt="FLAC", subtype=None, sr=SR) -> bytes:
    buf = io.BytesIO()
    sf.write(buf, samples, sr, format=fmt, subtype=subtype)
    return buf.getvalue()


def make_samples(i: int, n: int = 160) -> np.ndarray:
    return 0.25 * np.sin(2 * np.pi * (200 + 10 * i) * np.arange(n) / SR)


def make_example(i: int, **overrides) -> dict:
    """Example in the undecoded (Audio(decode=False)) representation."""
    example = {
        "id": f"1-2-{i:04d}",
        "text": f"UTTERANCE NUMBER {i}",
        "audio": {"bytes": encode(make_samples(i)), "path": f"1-2-{i:04d}.flac"},
    }
    example.update(overrides)
    return example


class FakeDataset:
    """Minimal stand-in for a Hugging Face Dataset (len + integer indexing + cast_column)."""

    def __init__(self, examples):
        self.examples = list(examples)
        self.casts = []

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        return self.examples[index]

    def cast_column(self, name, feature):
        self.casts.append((name, feature))
        return self


@pytest.fixture
def patch_dataset(monkeypatch):
    def _patch(examples):
        monkeypatch.setattr(data, "_load_split", lambda: FakeDataset(examples))

    return _patch


# --- audio decoding --------------------------------------------------------
def test_decode_flac_bytes_roundtrip():
    samples = make_samples(1)
    wav, sr = decode_audio({"bytes": encode(samples), "path": "ignored.flac"})
    assert sr == SR and wav.dtype == np.float64 and wav.shape == samples.shape
    np.testing.assert_allclose(wav, samples, atol=1 / 2**14)  # 16-bit quantization


def test_decode_wav_float_bytes_is_exact():
    samples = make_samples(2)
    wav, sr = decode_audio({"bytes": encode(samples, "WAV", "FLOAT"), "path": None})
    assert sr == SR
    np.testing.assert_allclose(wav, samples, atol=1e-6)


def test_decode_bytes_take_precedence_over_path():
    samples = make_samples(3)
    wav, _ = decode_audio({"bytes": encode(samples), "path": "/nonexistent/file.flac"})
    assert wav.shape == samples.shape


def test_decode_local_path(tmp_path):
    samples = make_samples(4)
    path = tmp_path / "clip.wav"
    path.write_bytes(encode(samples, "WAV", "FLOAT"))
    wav, sr = decode_audio({"bytes": None, "path": str(path)})
    assert sr == SR
    np.testing.assert_allclose(wav, samples, atol=1e-6)


def test_decode_preserves_native_sample_rate():
    _, sr = decode_audio({"bytes": encode(make_samples(0), sr=8000), "path": None})
    assert sr == 8000


@pytest.mark.parametrize(
    "bad,match",
    [
        (None, "mapping"),
        ("file.flac", "mapping"),
        (b"raw", "mapping"),
        ({}, "neither"),
        ({"bytes": None, "path": None}, "neither"),
        ({"array": np.ones(10), "sampling_rate": SR}, "already decoded"),
        ({"bytes": "not-bytes", "path": None}, "must be bytes"),
        ({"bytes": None, "path": 123}, "non-empty string"),
        ({"bytes": None, "path": "  "}, "non-empty string"),
        ({"bytes": b"definitely not audio", "path": None}, "could not decode"),
        ({"bytes": b"", "path": None}, "could not decode"),
        ({"bytes": None, "path": "/nonexistent/dir/missing.flac"}, "could not decode"),
    ],
)
def test_decode_rejects_unsupported_or_undecodable_audio(bad, match):
    with pytest.raises(ValueError, match=match):
        decode_audio(bad, "utt-1")


def test_decode_error_names_the_utterance():
    with pytest.raises(ValueError, match="utt-42"):
        decode_audio({"bytes": b"junk", "path": None}, "utt-42")


# --- record conversion -----------------------------------------------------
def test_convert_valid_record():
    rec = convert_record(make_example(3))
    assert set(rec) == {"utterance_id", "audio", "sample_rate", "reference"}
    assert rec["utterance_id"] == "1-2-0003"
    assert rec["reference"] == "UTTERANCE NUMBER 3"
    assert rec["sample_rate"] == SR and type(rec["sample_rate"]) is int
    assert isinstance(rec["audio"], np.ndarray) and rec["audio"].shape == (160,)


def test_convert_ignores_extra_dataset_fields():
    ex = make_example(0, file="x.flac", speaker_id=1, chapter_id=2)
    assert set(convert_record(ex)) == {"utterance_id", "audio", "sample_rate", "reference"}


def test_convert_stereo_is_rejected_not_downmixed():
    stereo = np.stack([make_samples(0), make_samples(1)], axis=1)
    ex = make_example(0, audio={"bytes": encode(stereo, "WAV", "FLOAT"), "path": None})
    with pytest.raises(ValueError, match="one-dimensional"):
        convert_record(ex)


@pytest.mark.parametrize("samples", [np.array([0.1, np.nan, 0.2]), np.array([0.1, np.inf, 0.2])])
def test_convert_rejects_non_finite_audio(samples):
    ex = make_example(0, audio={"bytes": encode(samples, "WAV", "FLOAT"), "path": None})
    with pytest.raises(ValueError, match="non-finite"):
        convert_record(ex)


def test_convert_rejects_empty_audio():
    ex = make_example(0, audio={"bytes": encode(np.zeros(0), "WAV", "FLOAT"), "path": None})
    with pytest.raises(ValueError, match="empty"):
        convert_record(ex)


@pytest.mark.parametrize("bad_rate", [0, -16000, 16000.0, True, None])
def test_convert_rejects_invalid_sample_rate(monkeypatch, bad_rate):
    monkeypatch.setattr(data, "decode_audio", lambda field, uid="": (np.ones(10), bad_rate))
    with pytest.raises(ValueError, match="sample rate"):
        convert_record(make_example(0))


def test_convert_accepts_numpy_integer_sample_rate(monkeypatch):
    monkeypatch.setattr(data, "decode_audio", lambda field, uid="": (np.ones(10), np.int64(16000)))
    rec = convert_record(make_example(0))
    assert rec["sample_rate"] == 16000 and type(rec["sample_rate"]) is int


@pytest.mark.parametrize(
    "overrides",
    [
        {"id": ""},
        {"id": "   "},
        {"id": 7},
        {"text": ""},
        {"text": "   "},
        {"text": None},
        {"text": 5},
        {"audio": None},
        {"audio": {"bytes": None, "path": None}},
    ],
)
def test_convert_rejects_malformed_record(overrides):
    with pytest.raises(ValueError):
        convert_record(make_example(0, **overrides))


@pytest.mark.parametrize("missing", ["id", "text", "audio"])
def test_convert_rejects_missing_field(missing):
    ex = make_example(0)
    del ex[missing]
    with pytest.raises(ValueError):
        convert_record(ex)


# --- selection -------------------------------------------------------------
def test_select_indices_deterministic_and_bounded():
    a = select_indices(100, 10, seed=42)
    assert a == select_indices(100, 10, seed=42)
    assert len(a) == len(set(a)) == 10
    assert a == sorted(a) and all(0 <= i < 100 for i in a)


def test_select_indices_differs_across_seeds():
    assert select_indices(100, 10, seed=1) != select_indices(100, 10, seed=2)


def test_select_indices_returns_all_when_dataset_small():
    assert select_indices(5, 20, seed=0) == [0, 1, 2, 3, 4]


def test_select_indices_does_not_touch_global_random_state():
    np.random.seed(123)
    expected = np.random.random()
    np.random.seed(123)
    select_indices(100, 10, seed=7)
    assert np.random.random() == expected


# --- loader ----------------------------------------------------------------
def test_loader_returns_valid_records(patch_dataset):
    patch_dataset([make_example(i) for i in range(30)])
    records = load_librispeech_subset(n_utterances=5, seed=42)
    assert len(records) == 5
    for rec in records:
        assert set(rec) == {"utterance_id", "audio", "sample_rate", "reference"}
        assert rec["audio"].ndim == 1 and rec["sample_rate"] == SR


def test_loader_never_exceeds_requested_count(patch_dataset):
    patch_dataset([make_example(i) for i in range(3)])
    assert len(load_librispeech_subset(n_utterances=20)) == 3
    assert len(load_librispeech_subset(n_utterances=2)) == 2


def test_loader_same_seed_same_ids_different_seed_different_ids(patch_dataset):
    patch_dataset([make_example(i) for i in range(200)])
    ids = lambda s: [r["utterance_id"] for r in load_librispeech_subset(10, seed=s)]
    assert ids(42) == ids(42)
    assert ids(1) != ids(2)


def test_loader_decodes_only_selected_examples(monkeypatch):
    class CountingDataset(FakeDataset):
        def __init__(self, examples):
            super().__init__(examples)
            self.accessed = []

        def __getitem__(self, index):
            self.accessed.append(index)
            return super().__getitem__(index)

    ds = CountingDataset([make_example(i) for i in range(100)])
    monkeypatch.setattr(data, "_load_split", lambda: ds)
    load_librispeech_subset(n_utterances=4, seed=0)
    assert len(ds.accessed) == 4


def test_loader_defaults(patch_dataset):
    patch_dataset([make_example(i) for i in range(50)])
    assert len(load_librispeech_subset()) == 20


@pytest.mark.parametrize("bad_n", [0, -1, 1.5, "5", None, True])
def test_loader_rejects_invalid_n_utterances(patch_dataset, bad_n):
    patch_dataset([make_example(0)])
    with pytest.raises(ValueError):
        load_librispeech_subset(n_utterances=bad_n)


@pytest.mark.parametrize("bad_seed", [-1, 1.5, "1", None, True])
def test_loader_rejects_invalid_seed(patch_dataset, bad_seed):
    patch_dataset([make_example(0)])
    with pytest.raises(ValueError):
        load_librispeech_subset(seed=bad_seed)


def test_loader_rejects_empty_dataset(patch_dataset):
    patch_dataset([])
    with pytest.raises(ValueError, match="empty"):
        load_librispeech_subset()


def test_loader_rejects_malformed_record_with_context(patch_dataset):
    examples = [make_example(i) for i in range(5)]
    examples[2] = make_example(2, text="  ")
    patch_dataset(examples)
    with pytest.raises(ValueError, match="index 2"):
        load_librispeech_subset(n_utterances=5)


def test_loader_rejects_corrupt_audio_with_context(patch_dataset):
    examples = [make_example(i) for i in range(5)]
    examples[1] = make_example(1, audio={"bytes": b"corrupt", "path": None})
    patch_dataset(examples)
    with pytest.raises(ValueError, match="index 1.*could not decode"):
        load_librispeech_subset(n_utterances=5)


def test_loader_rejects_duplicate_ids(patch_dataset):
    patch_dataset([make_example(0), make_example(1, id="1-2-0000")])
    with pytest.raises(ValueError, match="duplicate"):
        load_librispeech_subset(n_utterances=2)


def test_loader_does_not_repair_or_skip_bad_records(patch_dataset):
    examples = [make_example(i) for i in range(3)]
    bad = encode(np.array([np.nan, 0.1]), "WAV", "FLOAT")
    examples[1] = make_example(1, audio={"bytes": bad, "path": None})
    patch_dataset(examples)
    with pytest.raises(ValueError):
        load_librispeech_subset(n_utterances=3)


# --- _load_split with a fake `datasets` module (fully offline) ----------------
def fake_datasets_module(load_dataset, audio_cls=None):
    class FakeAudio:
        def __init__(self, decode=True):
            self.decode = decode

    return types.SimpleNamespace(load_dataset=load_dataset, Audio=audio_cls or FakeAudio)


EXPECTED_PATTERN = "hf://datasets/openslr/librispeech_asr/clean/test/*.parquet"


def test_load_split_requests_only_explicit_clean_test_parquet_pattern(monkeypatch):
    calls, ds = [], FakeDataset([make_example(0)])

    def fake_load_dataset(*args, **kwargs):
        calls.append((args, kwargs))
        return ds

    monkeypatch.setitem(sys.modules, "datasets", fake_datasets_module(fake_load_dataset))
    data._load_split()
    assert data.DATA_FILES_PATTERN == EXPECTED_PATTERN
    # exactly one call, via the generic parquet builder, one explicit file pattern -> "test"
    assert calls == [(("parquet",), {"data_files": {"test": EXPECTED_PATTERN}, "split": "test"})]
    assert len(ds.casts) == 1
    name, feature = ds.casts[0]
    assert name == "audio" and feature.decode is False


def test_load_split_never_names_the_repo_config_or_other_splits(monkeypatch):
    calls = []

    def fake_load_dataset(*args, **kwargs):
        calls.append((args, kwargs))
        return FakeDataset([make_example(0)])

    monkeypatch.setitem(sys.modules, "datasets", fake_datasets_module(fake_load_dataset))
    data._load_split()
    (args, kwargs), = calls
    assert args == ("parquet",)  # not the repo id, so the whole `clean` config cannot be resolved
    assert "name" not in kwargs and set(kwargs) == {"data_files", "split"}
    (split_name, pattern), = kwargs["data_files"].items()  # exactly one split, one pattern
    assert split_name == "test" and isinstance(pattern, str)
    for forbidden in ("train", "validation", "other", "**"):
        assert forbidden not in pattern


def test_pattern_matches_only_clean_test_files_of_a_simulated_repository(monkeypatch):
    """Simulate the Hub repo file tree and resolve the requested pattern against it."""
    import fnmatch

    repo = [
        "clean/test/0000.parquet",
        "clean/test/0001.parquet",
        "clean/validation/0000.parquet",
        *[f"clean/train.100/{i:04d}.parquet" for i in range(5)],
        *[f"clean/train.360/{i:04d}.parquet" for i in range(5)],
        "other/test/0000.parquet",
        "all/test.clean/0000.parquet",
        "README.md",
    ]
    prefix = "hf://datasets/openslr/librispeech_asr/"
    resolved = []

    def fake_load_dataset(builder, data_files=None, split=None, **kwargs):
        assert builder == "parquet" and list(data_files) == [split] == ["test"]
        pattern = data_files[split]
        assert pattern.startswith(prefix)
        resolved.extend(p for p in repo if fnmatch.fnmatchcase(p, pattern[len(prefix):]))
        return FakeDataset([make_example(0)])

    monkeypatch.setitem(sys.modules, "datasets", fake_datasets_module(fake_load_dataset))
    data._load_split()
    assert resolved == ["clean/test/0000.parquet", "clean/test/0001.parquet"]
    assert not any("train" in p or "validation" in p or p.startswith(("other", "all")) for p in resolved)


def test_load_split_wraps_loading_failure(monkeypatch):
    def failing(*args, **kwargs):
        raise ConnectionError("offline")

    monkeypatch.setitem(sys.modules, "datasets", fake_datasets_module(failing))
    with pytest.raises(RuntimeError, match="Failed to load dataset.*openslr/librispeech_asr.*offline"):
        load_librispeech_subset()


def test_load_split_wraps_cast_failure(monkeypatch):
    class NoAudioColumn(FakeDataset):
        def cast_column(self, name, feature):
            raise ValueError("Column audio not in the dataset")

    monkeypatch.setitem(
        sys.modules, "datasets", fake_datasets_module(lambda *a, **k: NoAudioColumn([make_example(0)]))
    )
    with pytest.raises(RuntimeError, match="Column audio"):
        load_librispeech_subset()


def test_load_split_reports_missing_datasets_package(monkeypatch):
    monkeypatch.setitem(sys.modules, "datasets", None)  # makes `import datasets` raise ImportError
    with pytest.raises(ImportError, match="datasets"):
        load_librispeech_subset()


# --- real `datasets` library, in-memory data only (no network) ------------------
def test_end_to_end_with_real_datasets_library_in_memory(monkeypatch):
    """Parquet-style audio struct -> Audio(decode=False) -> loader, using the real library."""
    datasets = pytest.importorskip("datasets")
    features = datasets.Features(
        {
            "audio": {"bytes": datasets.Value("binary"), "path": datasets.Value("string")},
            "text": datasets.Value("string"),
            "id": datasets.Value("string"),
        }
    )
    n = 12
    table = {
        "audio": [{"bytes": encode(make_samples(i)), "path": f"1-2-{i:04d}.flac"} for i in range(n)],
        "text": [f"UTTERANCE NUMBER {i}" for i in range(n)],
        "id": [f"1-2-{i:04d}" for i in range(n)],
    }
    dataset = datasets.Dataset.from_dict(table, features=features)
    seen = {}

    def fake_load_dataset(*args, **kwargs):
        seen["args"], seen["kwargs"] = args, kwargs
        return dataset

    monkeypatch.setattr(datasets, "load_dataset", fake_load_dataset)
    records = load_librispeech_subset(n_utterances=5, seed=42)

    assert seen["args"] == ("parquet",)
    assert seen["kwargs"] == {"data_files": {"test": EXPECTED_PATTERN}, "split": "test"}
    assert len(records) == 5
    ids = [r["utterance_id"] for r in records]
    assert ids == [r["utterance_id"] for r in load_librispeech_subset(n_utterances=5, seed=42)]
    for rec in records:
        i = int(rec["utterance_id"].rsplit("-", 1)[1])
        np.testing.assert_allclose(rec["audio"], make_samples(i), atol=1 / 2**14)
        assert rec["sample_rate"] == SR


# --- configuration / loader consistency ---------------------------------------
def test_baseline_config_matches_loader_constants():
    from pathlib import Path

    import yaml

    config_path = Path(__file__).resolve().parent.parent / "configs" / "baseline.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    assert config["dataset_id"] == data.DATASET_ID == "openslr/librispeech_asr"
    assert config["dataset_config"] == data.DATASET_CONFIG == "clean"
    assert config["dataset_split"] == data.DATASET_SPLIT == "test"
