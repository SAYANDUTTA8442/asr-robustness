"""Offline tests for the experiment runner, model wrapper and CLI.

Everything is mocked: no dataset, model, network or GPU is touched, and neither torch nor
transformers needs to be installed.
"""

import copy
import importlib.util
import json
import os
import subprocess
import sys
import types
from pathlib import Path

import numpy as np
import pytest

from asr_robustness import experiment, model
from asr_robustness.experiment import (
    condition_name,
    derive_noise_seed,
    load_config,
    run_experiment,
    save_report,
    validate_config,
)
from asr_robustness.metrics import corpus_word_error_rate, word_error_rate

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SR = 16000
SNRS = [20, 10, 5]
CONDITIONS = ["clean", "snr_20db", "snr_10db", "snr_5db"]

REFERENCES = [
    "the quick brown fox jumps over the lazy dog",
    "hello world",
    "she sells sea shells by the sea shore today",
]


def make_config(**overrides) -> dict:
    config = {
        "seed": 42,
        "n_utterances": 3,
        "snr_db": list(SNRS),
        "model_id": "fake/whisper",
        "dataset_id": "openslr/librispeech_asr",
        "dataset_config": "clean",
        "dataset_split": "test",
    }
    config.update(overrides)
    return config


def make_records(references=REFERENCES) -> list[dict]:
    records = []
    for i, text in enumerate(references):
        t = np.arange(1600 + 100 * i) / SR
        audio = 0.3 * np.sin(2 * np.pi * (220 + 40 * i) * t)
        records.append(
            {"utterance_id": f"1-2-{i:04d}", "audio": audio, "sample_rate": SR, "reference": text}
        )
    return records


class FakeLoader:
    """Stand-in for load_librispeech_subset; hands out copies and counts calls."""

    def __init__(self, records):
        self.records = records
        self.calls = []

    def __call__(self, n_utterances, seed):
        self.calls.append({"n_utterances": n_utterances, "seed": seed})
        return copy.deepcopy(self.records)


class FakeRecognizer:
    """Records every waveform it is given and answers via ``responder(call_index)``."""

    def __init__(self, responder=None, model_id="fake/whisper", device="fake-device"):
        self.responder = responder or (lambda i: "some text")
        self.model_id = model_id
        self.device = device
        self.received = []  # (copy of audio, sample_rate, writeable flag)

    def transcribe(self, audio, sample_rate):
        self.received.append((np.array(audio, copy=True), sample_rate, audio.flags.writeable))
        return self.responder(len(self.received) - 1)


def exact_then_degraded(records):
    """Perfect transcripts for the clean pass, then one dropped word per later pass."""
    n = len(records)

    def responder(call_index):
        text = records[call_index % n]["reference"]
        if call_index < n:
            return text
        return " ".join(text.split()[:-1])

    return responder


def run(config=None, records=None, recognizer=None, timer=None):
    records = records if records is not None else make_records()
    loader = FakeLoader(records)
    recognizer = recognizer or FakeRecognizer(exact_then_degraded(records))
    kwargs = {} if timer is None else {"timer": timer}
    report = run_experiment(config or make_config(), recognizer, load_subset=loader, **kwargs)
    return report, loader, recognizer


# --- conditions and dataset consistency ---------------------------------------
def test_four_conditions_in_order():
    report, _, _ = run()
    assert list(report["conditions"]) == CONDITIONS
    assert report["conditions"]["clean"]["type"] == "clean"
    assert [report["conditions"][c]["requested_snr_db"] for c in CONDITIONS[1:]] == [20.0, 10.0, 5.0]
    assert report["snr_db_levels"] == [20.0, 10.0, 5.0]


def test_dataset_loaded_once_with_config_settings():
    _, loader, _ = run(make_config(seed=7, n_utterances=3))
    assert loader.calls == [{"n_utterances": 3, "seed": 7}]


def test_utterance_ids_and_references_identical_across_conditions():
    records = make_records()
    report, _, _ = run(records=records)
    ids = [r["utterance_id"] for r in records]
    assert report["utterance_ids"] == ids
    for name in CONDITIONS:
        utts = report["conditions"][name]["utterances"]
        assert [u["utterance_id"] for u in utts] == ids
        assert [u["reference"] for u in utts] == [r["reference"] for r in records]


def test_each_condition_calls_recognizer_once_per_utterance():
    _, _, recognizer = run()
    assert len(recognizer.received) == 4 * len(REFERENCES)
    assert all(sr == SR for _, sr, _ in recognizer.received)


def test_conditions_follow_configured_snr_levels():
    report, _, _ = run(make_config(snr_db=[15, 0.5]))
    assert list(report["conditions"]) == ["clean", "snr_15db", "snr_0.5db"]
    assert condition_name(20) == condition_name(20.0) == "snr_20db"


def test_report_identifiers_come_from_config():
    config = make_config(model_id="other/model", seed=5)
    report, _, _ = run(config, recognizer=FakeRecognizer(model_id="other/model"))
    assert report["model_id"] == "other/model"
    assert report["seed"] == 5
    assert report["dataset"] == {"id": "openslr/librispeech_asr", "config": "clean", "split": "test"}


def test_recognizer_model_mismatch_is_rejected():
    with pytest.raises(ValueError, match="does not match configured model_id"):
        run(recognizer=FakeRecognizer(model_id="different/model"))


# --- noise determinism and non-mutation ---------------------------------------
def test_noise_is_deterministic_across_runs():
    _, _, rec_a = run()
    _, _, rec_b = run()
    for (a, _, _), (b, _, _) in zip(rec_a.received, rec_b.received):
        np.testing.assert_array_equal(a, b)


def test_noise_seeds_stable_and_distinct_per_utterance_and_snr():
    assert derive_noise_seed(42, "1-2-0000", 20) == derive_noise_seed(42, "1-2-0000", 20.0)
    seeds = {derive_noise_seed(42, f"1-2-{i:04d}", s) for i in range(5) for s in SNRS}
    assert len(seeds) == 15
    assert derive_noise_seed(42, "1-2-0000", 20) != derive_noise_seed(43, "1-2-0000", 20)
    # fixed value: guards against the derivation (and thus past results) silently changing
    assert derive_noise_seed(42, "1-2-0000", 20) == 2496084460093379629
    assert derive_noise_seed(42, "1-2-0000", 5) == 4010833419536534099


def test_noise_seed_independent_of_utterance_order():
    records = make_records()
    forward, _, _ = run(records=records)
    backward, _, _ = run(records=list(reversed(records)))

    def seeds(report):
        return {
            (name, u["utterance_id"]): u["noise_seed"]
            for name, cond in report["conditions"].items()
            for u in cond["utterances"]
        }

    assert seeds(forward) == seeds(backward)


def test_recorded_noise_seeds_reproduce_the_noise():
    records = make_records()
    report, _, recognizer = run(records=records)
    n = len(records)
    for c, name in enumerate(CONDITIONS[1:], start=1):
        for i, utt in enumerate(report["conditions"][name]["utterances"]):
            audio, _, _ = recognizer.received[c * n + i]
            expected = experiment.add_noise_at_snr(records[i]["audio"], SNRS[c - 1], utt["noise_seed"])
            np.testing.assert_array_equal(audio, expected)


def test_clean_arrays_are_not_mutated_and_passed_read_only():
    records = make_records()
    snapshot = [r["audio"].copy() for r in records]
    loader = FakeLoader(records)
    seen = []
    original_call = loader.__call__

    def capturing_loader(n_utterances, seed):
        loaded = original_call(n_utterances, seed)
        seen.extend(loaded)
        return loaded

    recognizer = FakeRecognizer(exact_then_degraded(records))
    run_experiment(make_config(), recognizer, load_subset=capturing_loader)
    for original, record in zip(snapshot, seen):
        np.testing.assert_array_equal(record["audio"], original)
    assert all(writeable is False for _, _, writeable in recognizer.received)


def test_clean_condition_receives_clean_audio_and_noisy_conditions_do_not():
    records = make_records()
    _, _, recognizer = run(records=records)
    n = len(records)
    for i, record in enumerate(records):
        np.testing.assert_array_equal(recognizer.received[i][0], record["audio"])
        for c in range(1, 4):
            assert not np.array_equal(recognizer.received[c * n + i][0], record["audio"])


def test_requested_and_measured_snr_recorded():
    report, _, _ = run()
    clean = report["conditions"]["clean"]
    assert clean["mean_measured_snr_db"] is None
    assert all(u["requested_snr_db"] is None and u["measured_snr_db"] is None for u in clean["utterances"])
    for name, snr in zip(CONDITIONS[1:], SNRS):
        cond = report["conditions"][name]
        for utt in cond["utterances"]:
            assert utt["requested_snr_db"] == float(snr)
            assert utt["measured_snr_db"] == pytest.approx(snr, abs=1e-6)
        assert cond["mean_measured_snr_db"] == pytest.approx(snr, abs=1e-6)


# --- metrics aggregation ------------------------------------------------------
def test_metrics_match_existing_functions():
    records = make_records()
    report, _, _ = run(records=records)
    for name in CONDITIONS:
        cond = report["conditions"][name]
        refs = [u["reference"] for u in cond["utterances"]]
        preds = [u["prediction"] for u in cond["utterances"]]
        assert cond["corpus_wer"] == corpus_word_error_rate(refs, preds)
        for u in cond["utterances"]:
            assert u["wer"] == word_error_rate(u["reference"], u["prediction"])
    assert report["conditions"]["clean"]["corpus_wer"] == 0.0
    assert report["conditions"]["clean"]["corpus_cer"] == 0.0
    assert report["conditions"]["snr_5db"]["corpus_wer"] > 0.0


def test_corpus_wer_is_total_errors_over_total_words_not_mean_of_utterance_wers():
    refs = ["a b c d e f g h i j", "x y"]  # 10 + 2 reference words
    records = make_records(refs)
    predictions = {0: refs[0], 1: "x z"}  # clean: 0 and 1 error

    def responder(call_index):
        return predictions[call_index % 2]

    config = make_config(snr_db=[10], n_utterances=2)
    report, _, _ = run(config, records, FakeRecognizer(responder))
    cond = report["conditions"]["clean"]
    assert [u["wer"] for u in cond["utterances"]] == [0.0, 0.5]
    assert cond["corpus_wer"] == pytest.approx(1 / 12)
    assert cond["corpus_wer"] != pytest.approx(np.mean([0.0, 0.5]))


def test_predictions_recorded_per_utterance():
    records = make_records()
    report, _, _ = run(records=records)
    clean = report["conditions"]["clean"]["utterances"]
    assert [u["prediction"] for u in clean] == [r["reference"] for r in records]
    noisy = report["conditions"]["snr_20db"]["utterances"]
    assert noisy[0]["prediction"] == "the quick brown fox jumps over the lazy"


def test_non_string_prediction_rejected():
    with pytest.raises(ValueError, match="must return str"):
        run(recognizer=FakeRecognizer(lambda i: None))


# --- timing, report structure, saving -----------------------------------------
def test_inference_time_counts_only_transcribe_calls():
    ticks = iter(float(i) for i in range(1000))
    report, _, _ = run(timer=lambda: next(ticks))  # each transcribe measures exactly 1.0
    n = len(REFERENCES)
    for name in CONDITIONS:
        assert report["conditions"][name]["inference_seconds"] == float(n)
    assert report["elapsed_inference_seconds"] == float(4 * n)


def test_report_structure_and_json_serializable():
    report, _, _ = run()
    for key in (
        "schema_version", "model_id", "dataset", "seed", "n_utterances", "utterance_ids",
        "device", "elapsed_inference_seconds", "snr_db_levels", "warnings", "conditions",
        "long_form_utterance_ids", "long_form_threshold_seconds",
    ):
        assert key in report
    assert report["device"] == "fake-device"
    assert report["n_utterances"] == len(REFERENCES)
    for cond in report["conditions"].values():
        for key in ("type", "requested_snr_db", "corpus_wer", "corpus_cer", "inference_seconds", "utterances"):
            assert key in cond
        for utt in cond["utterances"]:
            for key in ("utterance_id", "reference", "prediction", "wer", "cer",
                        "requested_snr_db", "measured_snr_db", "noise_seed", "duration_s", "long_form"):
                assert key in utt
    assert json.loads(json.dumps(report, allow_nan=False)) == report


def test_long_form_clips_are_reported_accurately_not_as_truncated():
    records = make_records()
    records[0]["audio"] = np.full(SR * 31, 0.1)  # 31 s: needs long-form
    records[1]["audio"] = np.full(SR * 30, 0.1)  # exactly 30 s: still one window
    report, _, _ = run(records=records)
    assert report["long_form_utterance_ids"] == ["1-2-0000"]
    assert report["long_form_threshold_seconds"] == 30.0
    assert report["warnings"] == []  # nothing is truncated any more
    assert "truncat" not in json.dumps(report).lower()
    for cond in report["conditions"].values():
        flags = {u["utterance_id"]: u["long_form"] for u in cond["utterances"]}
        assert flags == {"1-2-0000": True, "1-2-0001": False, "1-2-0002": False}


def test_no_long_form_ids_for_short_clips():
    report, _, _ = run()
    assert report["long_form_utterance_ids"] == []
    assert all(not u["long_form"] for c in report["conditions"].values() for u in c["utterances"])


def test_long_clips_keep_full_length_noisy_audio_and_consistent_ids():
    records = make_records()
    records[2]["audio"] = 0.2 * np.sin(np.arange(SR * 35) * 0.05)  # 35 s
    report, _, recognizer = run(records=records)
    n = len(records)
    for c in range(4):  # every condition hands the full-length waveform to the recognizer
        assert recognizer.received[c * n + 2][0].size == SR * 35
    ids = [r["utterance_id"] for r in records]
    assert all(
        [u["utterance_id"] for u in cond["utterances"]] == ids
        for cond in report["conditions"].values()
    )
    assert report["long_form_utterance_ids"] == ["1-2-0002"]


def test_save_report_creates_directories_and_round_trips(tmp_path):
    report, _, _ = run()
    target = tmp_path / "nested" / "dir" / "report.json"
    assert save_report(report, target) == target
    assert json.loads(target.read_text(encoding="utf-8")) == report
    assert [p.name for p in target.parent.iterdir()] == ["report.json"]  # no temp files left


# --- configuration ------------------------------------------------------------
def test_baseline_config_loads_and_validates():
    config = load_config(PROJECT_ROOT / "configs" / "baseline.yaml")
    assert config["snr_db"] == [20, 10, 5] and config["model_id"] == "openai/whisper-tiny.en"


@pytest.mark.parametrize(
    "overrides",
    [
        {"snr_db": []},
        {"snr_db": [20, 20.0]},
        {"snr_db": [float("nan")]},
        {"snr_db": [True]},
        {"snr_db": "20"},
        {"seed": -1},
        {"seed": True},
        {"n_utterances": 0},
        {"model_id": ""},
        {"dataset_split": "validation"},
        {"dataset_id": "someone/else"},
    ],
)
def test_invalid_config_rejected(overrides):
    with pytest.raises(ValueError):
        validate_config(make_config(**overrides))


@pytest.mark.parametrize("key", experiment.REQUIRED_KEYS)
def test_missing_config_key_rejected(key):
    config = make_config()
    del config[key]
    with pytest.raises(ValueError, match=key):
        validate_config(config)


# --- model.py (no torch / transformers needed) ---------------------------------
class FakeTorch:
    def __init__(self, cuda=False, devices=1):
        self.cuda = types.SimpleNamespace(is_available=lambda: cuda, device_count=lambda: devices)

    def inference_mode(self):
        class _Ctx:
            def __enter__(self_inner):
                return None

            def __exit__(self_inner, *exc):
                return False

        return _Ctx()


def test_auto_device_prefers_cuda_else_cpu():
    assert model.resolve_device("auto", FakeTorch(cuda=True)) == "cuda"
    assert model.resolve_device("auto", FakeTorch(cuda=False)) == "cpu"


def test_explicit_device_override():
    assert model.resolve_device("cpu", FakeTorch(cuda=True)) == "cpu"
    assert model.resolve_device("cuda", FakeTorch(cuda=True)) == "cuda"
    assert model.resolve_device("cuda:1", FakeTorch(cuda=True, devices=2)) == "cuda:1"


def test_explicit_cuda_without_cuda_fails_loudly():
    with pytest.raises(RuntimeError, match="CUDA is not available"):
        model.resolve_device("cuda", FakeTorch(cuda=False))
    with pytest.raises(RuntimeError, match="only 1 CUDA"):
        model.resolve_device("cuda:3", FakeTorch(cuda=True, devices=1))


@pytest.mark.parametrize("bad", ["gpu", "CUDA", "cuda:", "cuda:x", "", None, 0])
def test_invalid_device_names_rejected(bad):
    with pytest.raises(ValueError, match="invalid device"):
        model.validate_device_name(bad)


def test_importing_package_modules_does_not_import_torch_or_transformers():
    code = (
        "import sys, asr_robustness.model, asr_robustness.experiment;"
        "assert 'torch' not in sys.modules and 'transformers' not in sys.modules"
    )
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([str(PROJECT_ROOT / "src"), env.get("PYTHONPATH", "")])
    result = subprocess.run(
        [sys.executable, "-c", code], env=env, capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr


N_WINDOW = 30 * SR  # samples in one Whisper window


class FakeFeatures:
    def __init__(self, n_frames, events, label):
        self.shape = (1, 80, n_frames)
        self._events, self._label = events, label

    def to(self, device):
        self._events["to"].append((self._label, device))
        return self


class FakeBatch(dict):
    """Dict that also allows attribute access, like transformers' BatchFeature."""

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc


def install_fake_transformers(monkeypatch, torch_module, truncate_always=False):
    events = {"processor": [], "model": [], "to": [], "generate": [], "calls": []}

    class Processor:
        @classmethod
        def from_pretrained(cls, model_id):
            events["processor"].append(model_id)
            return cls()

        def __call__(self, waveform, sampling_rate, return_tensors, **kwargs):
            events["calls"].append({"n_samples": waveform.size, "kwargs": kwargs})
            events["waveform"] = waveform
            samples = waveform.size
            if kwargs.get("truncation", True) or truncate_always:  # real default: truncate
                samples = min(samples, N_WINDOW)
            if "padding" not in kwargs:  # short-form default pads to a full window
                frames = 3000
            else:
                frames = samples // 160
            batch = FakeBatch(input_features=FakeFeatures(frames, events, "input_features"))
            if kwargs.get("return_attention_mask"):
                batch["attention_mask"] = FakeFeatures(frames, events, "attention_mask")
            return batch

        def batch_decode(self, ids, skip_special_tokens):
            assert skip_special_tokens is True
            events["decoded_ids"] = ids
            return {(1, 2, 3): ["  hello there  "], (7, 8, 9): ["  first half second half  "]}[
                tuple(ids[0])
            ]

    class Model:
        @classmethod
        def from_pretrained(cls, model_id):
            events["model"].append(model_id)
            return cls()

        def to(self, device):
            events["model_device"] = device
            return self

        def eval(self):
            events["eval"] = True

        def generate(self, *args, **kwargs):
            events["generate"].append({"args": args, "kwargs": kwargs})
            if "return_timestamps" in kwargs:  # long-form call
                result = [[7, 8, 9]]
                if events.get("wrap_sequences"):
                    return types.SimpleNamespace(sequences=result)
                return result
            return [[1, 2, 3]]

    monkeypatch.setitem(sys.modules, "torch", torch_module)
    monkeypatch.setitem(
        sys.modules,
        "transformers",
        types.SimpleNamespace(WhisperProcessor=Processor, WhisperForConditionalGeneration=Model),
    )
    return events


def test_requires_long_form_boundary():
    assert not model.requires_long_form(N_WINDOW, SR)
    assert model.requires_long_form(N_WINDOW + 1, SR)
    assert not model.requires_long_form(100, SR)


def test_whisper_recognizer_loads_model_once_and_transcribes(monkeypatch):
    events = install_fake_transformers(monkeypatch, FakeTorch(cuda=False))
    recognizer = model.WhisperRecognizer("openai/whisper-tiny.en", "auto")
    assert recognizer.device == "cpu" and recognizer.model_id == "openai/whisper-tiny.en"
    first = recognizer.transcribe(np.zeros(1600), SR)
    second = recognizer.transcribe(np.zeros(1600), SR)
    assert first == second == "hello there"  # stripped
    assert events["processor"] == events["model"] == ["openai/whisper-tiny.en"]  # loaded once
    assert events["model_device"] == "cpu" and events["eval"] is True
    assert events["waveform"].dtype == np.float32


def test_short_clip_keeps_existing_inference_path(monkeypatch):
    events = install_fake_transformers(monkeypatch, FakeTorch())
    recognizer = model.load_recognizer("fake/whisper", "cpu")
    recognizer.transcribe(np.zeros(N_WINDOW), SR)  # exactly 30 s is still short-form
    (call,) = events["calls"]
    assert call["kwargs"] == {}  # processor called exactly as before: no long-form kwargs
    (gen,) = events["generate"]
    assert len(gen["args"]) == 1  # features passed positionally, as before
    assert gen["kwargs"] == {"do_sample": False, "num_beams": 1}  # no timestamps
    assert "return_timestamps" not in gen["kwargs"]


def test_long_clip_passes_full_waveform_without_truncation(monkeypatch):
    events = install_fake_transformers(monkeypatch, FakeTorch())
    recognizer = model.load_recognizer("fake/whisper", "cpu")
    n = 75 * SR + 123  # 75 s
    recognizer.transcribe(np.zeros(n), SR)
    (call,) = events["calls"]
    assert call["n_samples"] == n  # the whole waveform reaches the processor
    assert call["kwargs"] == {
        "truncation": False,
        "padding": "longest",
        "return_attention_mask": True,
    }
    (gen,) = events["generate"]
    assert gen["kwargs"]["input_features"].shape[-1] == n // 160 > 3000  # more than one window


def test_long_clip_uses_native_long_form_generate_and_decodes_result(monkeypatch):
    events = install_fake_transformers(monkeypatch, FakeTorch())
    recognizer = model.load_recognizer("fake/whisper", "cpu")
    text = recognizer.transcribe(np.zeros(45 * SR), SR)
    assert text == "first half second half"  # decoded + stripped
    (gen,) = events["generate"]
    assert gen["args"] == ()  # all inputs by keyword, including the attention mask
    assert set(gen["kwargs"]) == {
        "input_features", "attention_mask", "return_timestamps", "do_sample", "num_beams",
    }
    assert gen["kwargs"]["return_timestamps"] is True
    assert gen["kwargs"]["do_sample"] is False and gen["kwargs"]["num_beams"] == 1
    assert events["decoded_ids"] == [[7, 8, 9]]
    # every tensor was moved to the selected device
    assert {label for label, _ in events["to"]} == {"input_features", "attention_mask"}
    assert {device for _, device in events["to"]} == {"cpu"}


def test_long_form_accepts_generate_output_with_sequences_attribute(monkeypatch):
    events = install_fake_transformers(monkeypatch, FakeTorch())
    events["wrap_sequences"] = True
    recognizer = model.load_recognizer("fake/whisper", "cpu")
    assert recognizer.transcribe(np.zeros(40 * SR), SR) == "first half second half"


def test_long_form_never_uses_pipeline_chunking(monkeypatch):
    events = install_fake_transformers(monkeypatch, FakeTorch())
    model.load_recognizer("fake/whisper", "cpu").transcribe(np.zeros(40 * SR), SR)
    flat = json.dumps(
        [list(g["kwargs"]) for g in events["generate"]] + [list(c["kwargs"]) for c in events["calls"]]
    )
    assert "chunk_length" not in flat


def test_model_loaded_once_across_short_and_long_clips(monkeypatch):
    events = install_fake_transformers(monkeypatch, FakeTorch())
    recognizer = model.load_recognizer("fake/whisper", "cpu")
    for n in (SR, 40 * SR, SR, 50 * SR):
        recognizer.transcribe(np.zeros(n), SR)
    assert events["model"] == events["processor"] == ["fake/whisper"]
    assert len(events["generate"]) == 4


def test_long_form_refuses_a_processor_that_truncates(monkeypatch):
    install_fake_transformers(monkeypatch, FakeTorch(), truncate_always=True)
    recognizer = model.load_recognizer("fake/whisper", "cpu")
    with pytest.raises(RuntimeError, match="truncated"):
        recognizer.transcribe(np.zeros(40 * SR), SR)


def test_whisper_recognizer_rejects_wrong_sample_rate_and_shape(monkeypatch):
    install_fake_transformers(monkeypatch, FakeTorch())
    recognizer = model.load_recognizer("fake/whisper", "cpu")
    with pytest.raises(ValueError, match="16000"):
        recognizer.transcribe(np.zeros(100), 8000)
    with pytest.raises(ValueError, match="one-dimensional"):
        recognizer.transcribe(np.zeros((2, 100)), SR)


def test_whisper_recognizer_reports_missing_packages(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", None)  # makes `import torch` raise ImportError
    with pytest.raises(ImportError, match="PyTorch and Transformers"):
        model.WhisperRecognizer("fake/whisper")


# --- CLI ----------------------------------------------------------------------
@pytest.fixture(scope="module")
def cli():
    spec = importlib.util.spec_from_file_location(
        "run_experiment_script", PROJECT_ROOT / "scripts" / "run_experiment.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_cli_parses_windows_invocation(cli):
    args = cli.build_parser().parse_args(
        ["--config", r"configs\baseline.yaml", "--output", r"results\robustness_results.json",
         "--device", "auto"]
    )
    assert args.config == r"configs\baseline.yaml"
    assert args.output == r"results\robustness_results.json"
    assert args.device == "auto"


def test_cli_defaults_and_device_validation(cli):
    args = cli.build_parser().parse_args([])
    assert args.device == "auto"
    assert args.config.endswith("baseline.yaml") and args.output.endswith("robustness_results.json")
    assert cli.build_parser().parse_args(["--device", "cuda:0"]).device == "cuda:0"
    with pytest.raises(SystemExit):
        cli.build_parser().parse_args(["--device", "tpu"])


def test_cli_main_runs_end_to_end_with_fakes(cli, tmp_path, capsys):
    records = make_records()
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "seed: 42\nn_utterances: 3\nsnr_db: [20, 10, 5]\nmodel_id: fake/whisper\n"
        "dataset_id: openslr/librispeech_asr\ndataset_config: clean\ndataset_split: test\n",
        encoding="utf-8",
    )
    output = tmp_path / "out" / "results.json"
    factory_calls = []

    def factory(model_id, device):
        factory_calls.append((model_id, device))
        return FakeRecognizer(exact_then_degraded(records), model_id=model_id, device="cpu")

    code = cli.main(
        ["--config", str(config_path), "--output", str(output), "--device", "cpu"],
        recognizer_factory=factory,
        load_subset=FakeLoader(records),
    )
    assert code == 0
    assert factory_calls == [("fake/whisper", "cpu")]  # model built once, from config + CLI
    report = json.loads(output.read_text(encoding="utf-8"))
    assert list(report["conditions"]) == CONDITIONS
    assert report["device"] == "cpu"
    out = capsys.readouterr().out
    assert "snr_5db" in out and str(output) in out


def test_cli_main_reports_errors_without_traceback(cli, tmp_path, capsys):
    code = cli.main(["--config", str(tmp_path / "missing.yaml"), "--output", str(tmp_path / "o.json")],
                    recognizer_factory=lambda *a: pytest.fail("model must not load"))
    assert code == 1
    assert "ERROR" in capsys.readouterr().err
