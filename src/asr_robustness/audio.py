"""Deterministic Gaussian-noise utilities for SNR-controlled audio perturbation.

SNR is power-based: ``SNR_dB = 10 * log10(signal_power / noise_power)``,
where power is the mean square of the samples.
"""

import numpy as np


def _validate_waveform(name: str, x: object) -> np.ndarray:
    """Return ``x`` as a float64 1-D array, or raise ``ValueError``."""
    try:
        arr = np.asarray(x, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a numeric array") from exc
    if arr.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional, got {arr.ndim} dimensions")
    if arr.size == 0:
        raise ValueError(f"{name} must not be empty")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain only finite values")
    return arr


def _power(x: np.ndarray) -> float:
    return float(np.mean(x * x))


def add_noise_at_snr(signal: np.ndarray, snr_db: float, seed: int) -> np.ndarray:
    """Return ``signal`` plus Gaussian noise scaled to the requested SNR.

    The noise is drawn from a local generator seeded with ``seed`` (global
    random state is untouched) and rescaled using its *measured* power, so the
    achieved SNR matches ``snr_db`` up to floating-point error. The input is
    not modified and the output is not clipped. Output dtype is float64.
    """
    x = _validate_waveform("signal", signal)

    if isinstance(snr_db, bool) or not isinstance(snr_db, (int, float, np.integer, np.floating)):
        raise ValueError("snr_db must be a real number")
    if not np.isfinite(snr_db):
        raise ValueError("snr_db must be finite")
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be an integer")
    if seed < 0:
        raise ValueError("seed must be non-negative")

    signal_power = _power(x)
    if not np.isfinite(signal_power) or signal_power <= 0.0:
        raise ValueError("signal must have finite, non-zero power")

    noise = np.random.default_rng(int(seed)).standard_normal(x.size)
    noise_power = _power(noise)
    if noise_power <= 0.0:  # practically impossible; guards the division below
        raise ValueError("generated noise has zero power")

    target_noise_power = signal_power / (10.0 ** (float(snr_db) / 10.0))
    noise *= np.sqrt(target_noise_power / noise_power)
    return x + noise


def measure_snr_db(reference: np.ndarray, noisy: np.ndarray) -> float:
    """Measure SNR in dB, treating ``noisy - reference`` as the added noise."""
    ref = _validate_waveform("reference", reference)
    nsy = _validate_waveform("noisy", noisy)
    if ref.shape != nsy.shape:
        raise ValueError(
            f"reference and noisy must have the same length, got {ref.size} and {nsy.size}"
        )

    signal_power = _power(ref)
    if not np.isfinite(signal_power) or signal_power <= 0.0:
        raise ValueError("reference must have finite, non-zero power")
    noise_power = _power(nsy - ref)
    if not np.isfinite(noise_power) or noise_power <= 0.0:
        raise ValueError("noise (noisy - reference) must have finite, non-zero power")

    return float(10.0 * np.log10(signal_power / noise_power))
