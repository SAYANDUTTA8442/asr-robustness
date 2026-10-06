"""Whisper recognizer wrapper with lazy PyTorch/Transformers imports.

Importing this module never imports ``torch`` or ``transformers``; they are imported only
when a device must be resolved against CUDA or when a ``WhisperRecognizer`` is built. This
keeps the offline test suite runnable without those packages.

A recognizer is any object with ``device`` (str), ``model_id`` (str) and
``transcribe(audio, sample_rate) -> str``; the experiment runner depends only on that.
"""

import re

import numpy as np

WHISPER_SAMPLE_RATE = 16000
WHISPER_MAX_SECONDS = 30.0  # one Whisper input window; longer audio needs long-form generation
WHISPER_HOP_LENGTH = 160  # samples per feature frame (fixed for all Whisper models)

_DEVICE_PATTERN = re.compile(r"^(auto|cpu|cuda(:\d+)?)$")


def validate_device_name(device: str) -> str:
    """Return ``device`` if it is ``auto``, ``cpu``, ``cuda`` or ``cuda:N``; else ValueError."""
    if not isinstance(device, str) or not _DEVICE_PATTERN.match(device):
        raise ValueError(
            f"invalid device {device!r}: expected 'auto', 'cpu', 'cuda' or 'cuda:<index>'"
        )
    return device


def resolve_device(requested: str = "auto", torch_module=None) -> str:
    """Resolve a requested device to a concrete device string.

    ``auto`` -> ``cuda`` when CUDA is available, otherwise ``cpu``. An explicit ``cpu`` never
    touches PyTorch. An explicit ``cuda``/``cuda:N`` is honoured only if CUDA is available
    (and the index exists); otherwise RuntimeError is raised rather than silently falling
    back. ``torch_module`` may be supplied (tests); otherwise ``torch`` is imported lazily.
    """
    requested = validate_device_name(requested)
    if requested == "cpu":
        return "cpu"

    if torch_module is None:
        try:
            import torch as torch_module
        except ImportError as exc:
            if requested == "auto":
                return "cpu"
            raise ImportError(
                f"device {requested!r} requires PyTorch, which is not installed"
            ) from exc

    cuda_available = bool(torch_module.cuda.is_available())
    if requested == "auto":
        return "cuda" if cuda_available else "cpu"

    if not cuda_available:
        raise RuntimeError(f"device {requested!r} was requested but CUDA is not available")
    if ":" in requested:
        index = int(requested.split(":", 1)[1])
        count = int(torch_module.cuda.device_count())
        if index >= count:
            raise RuntimeError(
                f"device {requested!r} was requested but only {count} CUDA device(s) exist"
            )
    return requested


def requires_long_form(num_samples: int, sample_rate: int) -> bool:
    """True when a clip is longer than one 30 s Whisper window (strictly greater)."""
    return num_samples > WHISPER_MAX_SECONDS * sample_rate


class WhisperRecognizer:
    """A Whisper model loaded once and reused for every ``transcribe`` call.

    Clips that fit in one 30 s window use the normal short-form path. Longer clips use
    Whisper's native sequential long-form generation (``model.generate`` with
    ``return_timestamps=True`` on un-truncated features), never pipeline ``chunk_length_s``.
    """

    def __init__(self, model_id: str, device: str = "auto"):
        if not isinstance(model_id, str) or not model_id.strip():
            raise ValueError("model_id must be a non-empty string")
        try:
            import torch
            from transformers import WhisperForConditionalGeneration, WhisperProcessor
        except ImportError as exc:
            raise ImportError(
                "PyTorch and Transformers are required to run Whisper; "
                f"install them before building a WhisperRecognizer ({exc})"
            ) from exc

        self.model_id = model_id
        self.device = resolve_device(device, torch)
        self._torch = torch
        self.processor = WhisperProcessor.from_pretrained(model_id)
        model = WhisperForConditionalGeneration.from_pretrained(model_id)
        self.model = model.to(self.device)
        self.model.eval()

    def transcribe(self, audio: np.ndarray, sample_rate: int) -> str:
        """Greedy-decode one mono waveform and return the stripped transcript."""
        if sample_rate != WHISPER_SAMPLE_RATE:
            raise ValueError(
                f"Whisper expects {WHISPER_SAMPLE_RATE} Hz audio, got {sample_rate} Hz"
            )
        waveform = np.asarray(audio, dtype=np.float32)
        if waveform.ndim != 1 or waveform.size == 0:
            raise ValueError("audio must be a non-empty one-dimensional array")

        if requires_long_form(waveform.size, sample_rate):
            return self._transcribe_long_form(waveform, sample_rate)

        inputs = self.processor(waveform, sampling_rate=sample_rate, return_tensors="pt")
        features = inputs.input_features.to(self.device)
        with self._torch.inference_mode():
            generated = self.model.generate(features, do_sample=False, num_beams=1)
        return self._decode(generated)

    def _transcribe_long_form(self, waveform: np.ndarray, sample_rate: int) -> str:
        """Transcribe audio longer than 30 s with Whisper's native long-form generation."""
        inputs = self.processor(
            waveform,
            sampling_rate=sample_rate,
            return_tensors="pt",
            truncation=False,  # keep the whole waveform
            padding="longest",  # do not pad to 30 s; generate() slices the windows itself
            return_attention_mask=True,  # required by long-form generate()
        )
        n_frames = int(inputs["input_features"].shape[-1])
        expected_frames = waveform.size // WHISPER_HOP_LENGTH
        if n_frames < expected_frames:
            raise RuntimeError(
                f"processor truncated long-form audio: {n_frames} feature frames for "
                f"{expected_frames} expected; refusing to transcribe a truncated waveform"
            )
        batch = {name: value.to(self.device) for name, value in inputs.items()}
        with self._torch.inference_mode():
            generated = self.model.generate(
                **batch, return_timestamps=True, do_sample=False, num_beams=1
            )
        return self._decode(generated)

    def _decode(self, generated) -> str:
        # long-form generate() may return a ModelOutput-like object with ``sequences``
        sequences = getattr(generated, "sequences", generated)
        text = self.processor.batch_decode(sequences, skip_special_tokens=True)[0]
        return text.strip()


def load_recognizer(model_id: str, device: str = "auto") -> WhisperRecognizer:
    """Build a ``WhisperRecognizer`` (loads the model once)."""
    return WhisperRecognizer(model_id, device)
