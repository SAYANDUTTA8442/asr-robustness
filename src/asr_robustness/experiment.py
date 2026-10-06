"""Reproducible ASR robustness experiment: clean audio plus Gaussian noise at configured SNRs.

The runner is independent of any concrete model: it needs a *recognizer* (``device``,
``model_id``, ``transcribe(audio, sample_rate) -> str``) and a subset loader, both
injectable. Model ID, dataset settings, seed and SNR levels all come from the configuration.
"""

import hashlib
import json
import math
import os
import tempfile
import time
from pathlib import Path

import numpy as np
import yaml

from asr_robustness import data as data_module
from asr_robustness.audio import add_noise_at_snr, measure_snr_db
from asr_robustness.data import load_librispeech_subset
from asr_robustness.metrics import (
    character_error_rate,
    corpus_character_error_rate,
    corpus_word_error_rate,
    word_error_rate,
)
from asr_robustness.model import WHISPER_MAX_SECONDS, requires_long_form

SCHEMA_VERSION = 2
CLEAN_CONDITION = "clean"
REQUIRED_KEYS = (
    "seed",
    "n_utterances",
    "snr_db",
    "model_id",
    "dataset_id",
    "dataset_config",
    "dataset_split",
)


def _is_int(value) -> bool:
    return isinstance(value, (int, np.integer)) and not isinstance(value, bool)


def _is_real(value) -> bool:
    return isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(value, bool)


def validate_config(config: object) -> dict:
    """Validate a configuration mapping (no defaults are filled in) and return it."""
    if not isinstance(config, dict):
        raise ValueError("configuration must be a mapping")
    missing = [key for key in REQUIRED_KEYS if key not in config]
    if missing:
        raise ValueError(f"configuration is missing required key(s): {', '.join(missing)}")

    if not _is_int(config["seed"]) or config["seed"] < 0:
        raise ValueError("seed must be a non-negative integer")
    if not _is_int(config["n_utterances"]) or config["n_utterances"] < 1:
        raise ValueError("n_utterances must be an integer >= 1")
    for key in ("model_id", "dataset_id", "dataset_config", "dataset_split"):
        if not isinstance(config[key], str) or not config[key].strip():
            raise ValueError(f"{key} must be a non-empty string")

    levels = config["snr_db"]
    if not isinstance(levels, (list, tuple)) or not levels:
        raise ValueError("snr_db must be a non-empty list of numbers")
    names = set()
    for level in levels:
        if not _is_real(level) or not math.isfinite(level):
            raise ValueError(f"snr_db entries must be finite real numbers, got {level!r}")
        name = condition_name(level)
        if name in names:
            raise ValueError(f"snr_db contains a duplicate level: {level!r}")
        names.add(name)

    # The loader's dataset is fixed in data.py; refuse to report a dataset we did not load.
    expected = (data_module.DATASET_ID, data_module.DATASET_CONFIG, data_module.DATASET_SPLIT)
    actual = (config["dataset_id"], config["dataset_config"], config["dataset_split"])
    if actual != expected:
        raise ValueError(
            f"configured dataset {actual} does not match the dataset the loader provides {expected}"
        )
    return config


def load_config(path) -> dict:
    """Read and validate a YAML configuration file."""
    with Path(path).open("r", encoding="utf-8") as handle:
        return validate_config(yaml.safe_load(handle))


def _format_snr(snr_db) -> str:
    return format(float(snr_db), ".6g")


def condition_name(snr_db) -> str:
    """Condition label for a noisy condition, e.g. ``snr_20db``."""
    return f"snr_{_format_snr(snr_db)}db"


def derive_noise_seed(base_seed: int, utterance_id: str, snr_db) -> int:
    """Deterministic non-negative noise seed for one (utterance, SNR) pair.

    Derived from SHA-256 (not ``hash()``, which is randomized per process), so it is stable
    across runs, machines and utterance ordering. ``20`` and ``20.0`` give the same seed.
    """
    key = f"{int(base_seed)}|{utterance_id}|{_format_snr(snr_db)}".encode()
    return int.from_bytes(hashlib.sha256(key).digest()[:8], "big")


def _read_only_view(array: np.ndarray) -> np.ndarray:
    view = array.view()
    view.flags.writeable = False
    return view


def _transcribe(recognizer, audio, sample_rate, timer) -> tuple[str, float]:
    start = timer()
    prediction = recognizer.transcribe(_read_only_view(audio), sample_rate)
    elapsed = timer() - start
    if not isinstance(prediction, str):
        raise ValueError(f"recognizer must return str, got {type(prediction).__name__}")
    return prediction, elapsed


def _evaluate_condition(name, requested_snr, records, recognizer, seed, timer) -> dict:
    utterances = []
    inference_seconds = 0.0
    for record in records:
        clean = record["audio"]
        entry = {
            "utterance_id": record["utterance_id"],
            "duration_s": float(clean.size / record["sample_rate"]),
            "long_form": bool(requires_long_form(clean.size, record["sample_rate"])),
        }
        if requested_snr is None:
            audio = clean
            entry.update(requested_snr_db=None, measured_snr_db=None, noise_seed=None)
        else:
            noise_seed = derive_noise_seed(seed, record["utterance_id"], requested_snr)
            audio = add_noise_at_snr(clean, requested_snr, noise_seed)  # new array
            entry.update(
                requested_snr_db=float(requested_snr),
                measured_snr_db=float(measure_snr_db(clean, audio)),
                noise_seed=noise_seed,
            )

        prediction, elapsed = _transcribe(recognizer, audio, record["sample_rate"], timer)
        inference_seconds += elapsed
        entry.update(
            reference=record["reference"],
            prediction=prediction,
            wer=float(word_error_rate(record["reference"], prediction)),
            cer=float(character_error_rate(record["reference"], prediction)),
        )
        utterances.append(entry)

    references = [u["reference"] for u in utterances]
    predictions = [u["prediction"] for u in utterances]
    measured = [u["measured_snr_db"] for u in utterances if u["measured_snr_db"] is not None]
    return {
        "type": "clean" if requested_snr is None else "gaussian_noise",
        "requested_snr_db": None if requested_snr is None else float(requested_snr),
        "mean_measured_snr_db": float(np.mean(measured)) if measured else None,
        "corpus_wer": float(corpus_word_error_rate(references, predictions)),
        "corpus_cer": float(corpus_character_error_rate(references, predictions)),
        "inference_seconds": float(inference_seconds),
        "utterances": utterances,
    }


def run_experiment(config, recognizer, load_subset=load_librispeech_subset, timer=time.perf_counter):
    """Run all conditions on one shared utterance subset and return the report dict.

    The subset is loaded once; conditions run in order: clean, then ``snr_db`` as configured.
    Inference time covers only ``recognizer.transcribe`` calls (not noise or metrics).
    """
    config = validate_config(config)
    recognizer_model = getattr(recognizer, "model_id", None)
    if recognizer_model is not None and recognizer_model != config["model_id"]:
        raise ValueError(
            f"recognizer model {recognizer_model!r} does not match configured "
            f"model_id {config['model_id']!r}"
        )

    records = load_subset(n_utterances=config["n_utterances"], seed=config["seed"])
    if not records:
        raise ValueError("dataset loader returned no utterances")
    ids = [r["utterance_id"] for r in records]
    if len(set(ids)) != len(ids):
        raise ValueError("dataset loader returned duplicate utterance ids")

    levels = [None, *config["snr_db"]]
    conditions = {}
    for level in levels:
        name = CLEAN_CONDITION if level is None else condition_name(level)
        conditions[name] = _evaluate_condition(
            name, level, records, recognizer, config["seed"], timer
        )

    # Clips over one 30 s window need Whisper long-form transcription; recorded for the report.
    long_form_ids = [
        r["utterance_id"]
        for r in records
        if requires_long_form(r["audio"].size, r["sample_rate"])
    ]
    return {
        "schema_version": SCHEMA_VERSION,
        "model_id": config["model_id"],
        "dataset": {
            "id": config["dataset_id"],
            "config": config["dataset_config"],
            "split": config["dataset_split"],
        },
        "seed": int(config["seed"]),
        "n_utterances_requested": int(config["n_utterances"]),
        "n_utterances": len(records),
        "utterance_ids": ids,
        "snr_db_levels": [float(x) for x in config["snr_db"]],
        "device": str(recognizer.device),
        "elapsed_inference_seconds": float(sum(c["inference_seconds"] for c in conditions.values())),
        "long_form_threshold_seconds": float(WHISPER_MAX_SECONDS),
        "long_form_utterance_ids": long_form_ids,
        "warnings": [],
        "conditions": conditions,
    }


def save_report(report: dict, path) -> Path:
    """Write ``report`` as UTF-8 JSON, creating parent directories; replaces atomically."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=path.name + ".", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
        os.replace(tmp_name, path)
    except BaseException:
        Path(tmp_name).unlink(missing_ok=True)
        raise
    return path
