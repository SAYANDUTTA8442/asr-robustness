"""Unit tests for asr_robustness.audio (deterministic synthetic signals only)."""

import numpy as np
import pytest

from asr_robustness.audio import add_noise_at_snr, measure_snr_db

SR = 16000


@pytest.fixture
def tone() -> np.ndarray:
    t = np.arange(SR) / SR
    return (0.5 * np.sin(2 * np.pi * 440.0 * t)).astype(np.float64)


@pytest.fixture
def speechlike() -> np.ndarray:
    t = np.arange(SR) / SR
    envelope = 0.2 + 0.8 * np.abs(np.sin(2 * np.pi * 3.0 * t))
    return 0.3 * envelope * np.sin(2 * np.pi * 220.0 * t + 0.5)


def test_output_length_and_input_preserved(tone):
    original = tone.copy()
    out = add_noise_at_snr(tone, 10, seed=0)
    assert out.shape == tone.shape
    assert out is not tone
    np.testing.assert_array_equal(tone, original)


def test_global_random_state_untouched(tone):
    np.random.seed(123)
    expected = np.random.random()
    np.random.seed(123)
    add_noise_at_snr(tone, 10, seed=7)
    assert np.random.random() == expected


def test_reproducible_for_same_seed(tone):
    a = add_noise_at_snr(tone, 10, seed=42)
    b = add_noise_at_snr(tone, 10, seed=42)
    np.testing.assert_array_equal(a, b)


def test_different_seeds_differ(tone):
    a = add_noise_at_snr(tone, 10, seed=1)
    b = add_noise_at_snr(tone, 10, seed=2)
    assert not np.array_equal(a, b)


@pytest.mark.parametrize("snr_db", [20, 10, 5])
@pytest.mark.parametrize("fixture_name", ["tone", "speechlike"])
def test_achieved_snr_within_tolerance(request, fixture_name, snr_db):
    signal = request.getfixturevalue(fixture_name)
    noisy = add_noise_at_snr(signal, snr_db, seed=42)
    assert measure_snr_db(signal, noisy) == pytest.approx(snr_db, abs=0.1)


def test_output_not_clipped(tone):
    noisy = add_noise_at_snr(tone, 0, seed=3)
    assert np.max(np.abs(noisy)) > 1.0  # 0 dB noise on a 0.5-amplitude tone exceeds [-1, 1]


@pytest.mark.parametrize(
    "bad",
    [
        np.array([]),
        np.zeros(100),
        np.array([0.1, np.nan, 0.2]),
        np.array([0.1, np.inf, 0.2]),
        np.ones((4, 4)),
    ],
    ids=["empty", "silent", "nan", "inf", "2d"],
)
def test_add_noise_rejects_invalid_signal(bad):
    with pytest.raises(ValueError):
        add_noise_at_snr(bad, 10, seed=0)


@pytest.mark.parametrize("bad_snr", [float("nan"), float("inf"), float("-inf"), "10", None, True])
def test_add_noise_rejects_invalid_snr(tone, bad_snr):
    with pytest.raises(ValueError):
        add_noise_at_snr(tone, bad_snr, seed=0)


@pytest.mark.parametrize("bad_seed", [-1, 1.5, "1", None, True])
def test_add_noise_rejects_invalid_seed(tone, bad_seed):
    with pytest.raises(ValueError):
        add_noise_at_snr(tone, 10, seed=bad_seed)


def test_measure_snr_rejects_mismatched_lengths(tone):
    with pytest.raises(ValueError):
        measure_snr_db(tone, tone[:-1] + 0.1)


def test_measure_snr_rejects_empty():
    with pytest.raises(ValueError):
        measure_snr_db(np.array([]), np.array([]))


def test_measure_snr_rejects_invalid_dimensions(tone):
    with pytest.raises(ValueError):
        measure_snr_db(tone.reshape(2, -1), tone.reshape(2, -1) + 0.1)
    with pytest.raises(ValueError):
        measure_snr_db(tone, (tone + 0.1).reshape(2, -1))


def test_measure_snr_rejects_silent_reference(tone):
    with pytest.raises(ValueError):
        measure_snr_db(np.zeros_like(tone), tone)


def test_measure_snr_rejects_zero_power_noise(tone):
    with pytest.raises(ValueError):
        measure_snr_db(tone, tone.copy())


def test_measure_snr_rejects_non_finite(tone):
    bad = tone.copy()
    bad[0] = np.nan
    with pytest.raises(ValueError):
        measure_snr_db(tone, bad)


def test_measure_snr_known_value():
    ref = np.ones(100)
    noisy = ref + 0.1  # noise power 0.01, signal power 1 -> 20 dB
    assert measure_snr_db(ref, noisy) == pytest.approx(20.0)
