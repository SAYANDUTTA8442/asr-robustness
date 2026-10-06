"""Deterministic tests for asr_robustness.metrics."""

import pytest

from asr_robustness.metrics import (
    character_error_rate,
    corpus_character_error_rate,
    corpus_word_error_rate,
    edit_distance,
    normalize_text,
    word_error_rate,
)


# --- normalization ---------------------------------------------------------
def test_normalize_case_punctuation_whitespace():
    assert normalize_text("  Hello,   WORLD!!\n\tHow  are you? ") == "hello world how are you"


def test_normalize_apostrophes_deleted_other_punctuation_split():
    assert normalize_text("Don't") == "dont"
    assert normalize_text("It\u2019s") == "its"
    assert normalize_text("well-known") == "well known"


def test_normalize_preserves_alphanumerics_and_is_idempotent():
    out = normalize_text("Room 101, Caf\u00e9!")
    assert out == "room 101 caf\u00e9"
    assert normalize_text(out) == out


def test_normalize_empty_and_punctuation_only():
    assert normalize_text("") == ""
    assert normalize_text(" ... ?! ") == ""


@pytest.mark.parametrize("bad", [None, 5, 1.5, ["a"], b"bytes"])
def test_normalize_rejects_non_string(bad):
    with pytest.raises(ValueError):
        normalize_text(bad)


# --- edit distance ---------------------------------------------------------
def test_edit_distance_basics():
    assert edit_distance("kitten", "sitting") == 3
    assert edit_distance("", "abc") == 3
    assert edit_distance("abc", "") == 3
    assert edit_distance([], []) == 0
    assert edit_distance(["a", "b"], ["a", "c"]) == 1


# --- WER -------------------------------------------------------------------
def test_wer_exact_match_is_zero():
    assert word_error_rate("the cat sat", "the cat sat") == 0.0


def test_wer_substitution_deletion_insertion():
    ref = "the cat sat"
    assert word_error_rate(ref, "the dog sat") == pytest.approx(1 / 3)
    assert word_error_rate(ref, "the sat") == pytest.approx(1 / 3)
    assert word_error_rate(ref, "the big cat sat") == pytest.approx(1 / 3)


def test_wer_normalization_applied_to_both_sides():
    assert word_error_rate("The CAT, sat.", "the   cat sat") == 0.0
    assert word_error_rate("don't stop", "DONT STOP!") == 0.0


def test_wer_differing_lengths():
    assert word_error_rate("a b", "a b c d e f") == pytest.approx(2.0)  # 4 insertions / 2
    assert word_error_rate("a b c d", "") == pytest.approx(1.0)  # 4 deletions / 4
    assert word_error_rate("a b c d", "x") == pytest.approx(1.0)  # 1 sub + 3 del / 4


@pytest.mark.parametrize("ref", ["", "   ", "...!?"])
def test_wer_rejects_empty_reference(ref):
    with pytest.raises(ValueError):
        word_error_rate(ref, "something")


@pytest.mark.parametrize("ref,hyp", [(None, "a"), ("a", None), (1, "a"), ("a", ["a"])])
def test_wer_rejects_non_string(ref, hyp):
    with pytest.raises(ValueError):
        word_error_rate(ref, hyp)


# --- CER -------------------------------------------------------------------
def test_cer_exact_match_is_zero():
    assert character_error_rate("Hello, World", "hello world") == 0.0


def test_cer_known_values():
    assert character_error_rate("cat", "cut") == pytest.approx(1 / 3)  # substitution
    assert character_error_rate("cats", "cat") == pytest.approx(1 / 4)  # deletion
    assert character_error_rate("cat", "cart") == pytest.approx(1 / 3)  # insertion
    assert character_error_rate("kitten", "sitting") == pytest.approx(3 / 6)


def test_cer_counts_spaces_after_normalization():
    # "the cat" -> 7 chars including the space; one substitution
    assert character_error_rate("the cat", "the cut") == pytest.approx(1 / 7)
    assert character_error_rate("the  cat", "the cat") == 0.0


def test_cer_differing_lengths():
    assert character_error_rate("ab", "abcdef") == pytest.approx(2.0)
    assert character_error_rate("abcd", "") == pytest.approx(1.0)


@pytest.mark.parametrize("ref", ["", " ", "?!"])
def test_cer_rejects_empty_reference(ref):
    with pytest.raises(ValueError):
        character_error_rate(ref, "abc")


def test_cer_rejects_non_string():
    with pytest.raises(ValueError):
        character_error_rate(None, "a")
    with pytest.raises(ValueError):
        character_error_rate("a", 3)


# --- corpus metrics --------------------------------------------------------
def test_corpus_wer_uses_total_errors_not_mean_of_rates():
    refs = ["a b c d", "e"]
    hyps = ["a b c d", "x"]
    mean_of_rates = (word_error_rate(refs[0], hyps[0]) + word_error_rate(refs[1], hyps[1])) / 2
    assert mean_of_rates == pytest.approx(0.5)
    assert corpus_word_error_rate(refs, hyps) == pytest.approx(1 / 5)


def test_corpus_cer_uses_total_errors_not_mean_of_rates():
    refs = ["abcdefghij", "k"]
    hyps = ["abcdefghij", "x"]
    mean_of_rates = (
        character_error_rate(refs[0], hyps[0]) + character_error_rate(refs[1], hyps[1])
    ) / 2
    assert mean_of_rates == pytest.approx(0.5)
    assert corpus_character_error_rate(refs, hyps) == pytest.approx(1 / 11)


def test_corpus_exact_match_and_single_item_agree_with_utterance_metrics():
    refs = ["The cat sat.", "On the mat"]
    assert corpus_word_error_rate(refs, ["the cat sat", "on the mat"]) == 0.0
    assert corpus_character_error_rate(refs, ["the cat sat", "on the mat"]) == 0.0
    assert corpus_word_error_rate(["a b c"], ["a x c"]) == word_error_rate("a b c", "a x c")
    assert corpus_character_error_rate(["abc"], ["axc"]) == character_error_rate("abc", "axc")


def test_corpus_allows_empty_reference_item_if_total_nonzero():
    # empty reference contributes its insertions to the numerator only
    assert corpus_word_error_rate(["a b", ""], ["a b", "x"]) == pytest.approx(1 / 2)


@pytest.mark.parametrize("fn", [corpus_word_error_rate, corpus_character_error_rate])
def test_corpus_rejects_empty_lists(fn):
    with pytest.raises(ValueError):
        fn([], [])


@pytest.mark.parametrize("fn", [corpus_word_error_rate, corpus_character_error_rate])
def test_corpus_rejects_mismatched_lengths(fn):
    with pytest.raises(ValueError):
        fn(["a b", "c"], ["a b"])


@pytest.mark.parametrize("fn", [corpus_word_error_rate, corpus_character_error_rate])
def test_corpus_rejects_zero_total_reference_length(fn):
    with pytest.raises(ValueError):
        fn(["", "..."], ["a", "b"])


@pytest.mark.parametrize("fn", [corpus_word_error_rate, corpus_character_error_rate])
def test_corpus_rejects_invalid_types(fn):
    with pytest.raises(ValueError):
        fn(["a", 1], ["a", "b"])
    with pytest.raises(ValueError):
        fn(["a", "b"], ["a", None])
    with pytest.raises(ValueError):
        fn("abc", "abc")  # a bare string is not a list of transcripts
    with pytest.raises(ValueError):
        fn(None, ["a"])
