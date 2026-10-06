"""LibriSpeech test-clean subset loader.

Source: Hugging Face ``openslr/librispeech_asr`` (parquet), config ``clean``, split ``test``.
Schema (from the dataset card): ``file``, ``audio`` (16 kHz), ``text``, ``speaker_id``,
``chapter_id``, ``id``.

Only the files matching ``clean/test/*.parquet`` are requested: the generic ``parquet``
loader is given that single explicit ``hf://`` pattern, so no other split of the repository
(``train.100``, ``train.360``, ``validation``) is ever resolved or downloaded.

Automatic audio decoding is disabled (``Audio(decode=False)``), so ``audio`` arrives as
``{"bytes": <encoded FLAC bytes or None>, "path": <str or None>}``. Only the selected
examples are decoded here with ``soundfile``; no TorchCodec is needed. Records are
validated, never repaired.
"""

import io
from collections.abc import Mapping

import numpy as np

DATASET_ID = "openslr/librispeech_asr"
DATASET_CONFIG = "clean"
DATASET_SPLIT = "test"
DATA_FILES_PATTERN = f"hf://datasets/{DATASET_ID}/{DATASET_CONFIG}/{DATASET_SPLIT}/*.parquet"


def _load_split():
    """Load only ``clean/test/*.parquet`` with audio decoding disabled.

    Uses the generic ``parquet`` builder with one explicit ``hf://`` file pattern mapped to
    the ``test`` split, instead of the repository's ``clean`` configuration, which would
    also pull the training shards. Needs the ``datasets`` package.
    """
    try:
        from datasets import Audio, load_dataset
    except ImportError as exc:
        raise ImportError(
            "The 'datasets' package is required to load LibriSpeech; "
            "install it before calling load_librispeech_subset()."
        ) from exc
    try:
        dataset = load_dataset(
            "parquet", data_files={DATASET_SPLIT: DATA_FILES_PATTERN}, split=DATASET_SPLIT
        )
        return dataset.cast_column("audio", Audio(decode=False))
    except Exception as exc:
        raise RuntimeError(
            f"Failed to load dataset {DATASET_ID!r} (config={DATASET_CONFIG!r}, "
            f"split={DATASET_SPLIT!r}, files={DATA_FILES_PATTERN!r}): {exc}"
        ) from exc


def _validate_int(name: str, value: object, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer, got {type(value).__name__}")
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {value}")
    return int(value)


def select_indices(n_total: int, n_utterances: int, seed: int) -> list[int]:
    """Deterministically pick up to ``n_utterances`` indices (sorted) using a local RNG."""
    if n_total <= n_utterances:
        return list(range(n_total))
    rng = np.random.default_rng(seed)
    return sorted(int(i) for i in rng.choice(n_total, size=n_utterances, replace=False))


def decode_audio(audio_field: object, utterance_id: str = "<unknown>") -> tuple[np.ndarray, int]:
    """Decode an undecoded dataset audio field into ``(waveform, sample_rate)``.

    ``audio_field`` must be a mapping with ``bytes`` (encoded audio) and/or ``path``
    (local file). Embedded bytes take precedence, because in the parquet dataset ``path``
    is only a file name. Channels are never down-mixed: multichannel audio is returned as
    given (2-D) and rejected later by validation. Raises ValueError for unsupported
    representations or undecodable audio.
    """
    if not isinstance(audio_field, Mapping):
        raise ValueError(
            f"{utterance_id}: audio must be a mapping with 'bytes'/'path', "
            f"got {type(audio_field).__name__}"
        )
    if "array" in audio_field:
        raise ValueError(
            f"{utterance_id}: audio is already decoded; this loader expects "
            "undecoded audio ({'bytes', 'path'}) from Audio(decode=False)"
        )
    audio_bytes = audio_field.get("bytes")
    audio_path = audio_field.get("path")

    if audio_bytes is not None:
        if not isinstance(audio_bytes, (bytes, bytearray)):
            raise ValueError(f"{utterance_id}: audio 'bytes' must be bytes, got {type(audio_bytes).__name__}")
        source = io.BytesIO(bytes(audio_bytes))
    elif audio_path is not None:
        if not isinstance(audio_path, str) or not audio_path.strip():
            raise ValueError(f"{utterance_id}: audio 'path' must be a non-empty string")
        source = audio_path
    else:
        raise ValueError(f"{utterance_id}: audio has neither 'bytes' nor 'path'")

    try:
        import soundfile as sf
    except ImportError as exc:
        raise ImportError("The 'soundfile' package is required to decode LibriSpeech audio.") from exc
    try:
        waveform, sample_rate = sf.read(source, dtype="float64", always_2d=False)
    except (RuntimeError, OSError, ValueError, TypeError) as exc:
        origin = "embedded bytes" if audio_bytes is not None else f"path {audio_path!r}"
        raise ValueError(f"{utterance_id}: could not decode audio from {origin}: {exc}") from exc
    return waveform, sample_rate


def convert_record(example: object) -> dict:
    """Validate one raw (undecoded) dataset example and convert it to the project record.

    Returns ``{"utterance_id", "audio", "sample_rate", "reference"}``.
    Raises ValueError on any malformed field.
    """
    try:
        utterance_id = example["id"]
        reference = example["text"]
        audio_field = example["audio"]
    except (KeyError, TypeError, IndexError) as exc:
        raise ValueError(f"record is missing required field(s) 'id', 'text', 'audio': {exc!r}") from exc

    if not isinstance(utterance_id, str) or not utterance_id.strip():
        raise ValueError("utterance id must be a non-empty string")
    if not isinstance(reference, str) or not reference.strip():
        raise ValueError(f"{utterance_id}: reference transcript must be a non-empty string")

    audio, sample_rate = decode_audio(audio_field, utterance_id)

    try:
        audio = np.asarray(audio, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{utterance_id}: audio array is not numeric") from exc
    if audio.ndim != 1:
        raise ValueError(f"{utterance_id}: audio must be one-dimensional, got {audio.ndim} dimensions")
    if audio.size == 0:
        raise ValueError(f"{utterance_id}: audio is empty")
    if not np.all(np.isfinite(audio)):
        raise ValueError(f"{utterance_id}: audio contains non-finite values")

    if isinstance(sample_rate, bool) or not isinstance(sample_rate, (int, np.integer)):
        raise ValueError(f"{utterance_id}: sample rate must be an integer")
    if sample_rate <= 0:
        raise ValueError(f"{utterance_id}: sample rate must be positive, got {sample_rate}")

    return {
        "utterance_id": utterance_id,
        "audio": audio,
        "sample_rate": int(sample_rate),
        "reference": reference,
    }


def load_librispeech_subset(n_utterances: int = 20, seed: int = 42) -> list[dict]:
    """Load a deterministic subset of LibriSpeech test-clean.

    Selection uses a local ``numpy`` generator seeded with ``seed``; the same seed picks
    the same utterances for an unchanged dataset. Only selected examples are decoded.
    Returns at most ``n_utterances`` records (see ``convert_record``), sorted by dataset
    index. Raises ValueError for invalid arguments, empty datasets, malformed records or
    duplicate ids, and RuntimeError/ImportError if the dataset cannot be loaded.
    """
    n_utterances = _validate_int("n_utterances", n_utterances, minimum=1)
    seed = _validate_int("seed", seed, minimum=0)

    dataset = _load_split()
    n_total = len(dataset)
    if n_total == 0:
        raise ValueError(f"dataset {DATASET_ID!r} split {DATASET_SPLIT!r} is empty")

    records = []
    for index in select_indices(n_total, n_utterances, seed):
        try:
            records.append(convert_record(dataset[index]))
        except ValueError as exc:
            raise ValueError(f"invalid dataset record at index {index}: {exc}") from exc

    ids = [r["utterance_id"] for r in records]
    if len(set(ids)) != len(ids):
        duplicates = sorted({i for i in ids if ids.count(i) > 1})
        raise ValueError(f"duplicate utterance ids in selected subset: {duplicates}")
    return records
