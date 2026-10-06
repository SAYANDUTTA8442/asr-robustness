"""Offline tests for scripts/analyze_results.py.

All reports are synthetic fixtures with arbitrary numbers; expectations are derived from the
fixture values (or from the formulas), never from any real experiment's observed WER/CER.
"""

import copy
import csv
import importlib.util
import io
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location(
    "analyze_results_script", PROJECT_ROOT / "scripts" / "analyze_results.py"
)
ar = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ar)

LEVELS = [20, 10, 5]
NAMES = ["clean", "snr_20db", "snr_10db", "snr_5db"]


def make_report(n=4, levels=LEVELS, corpus_wer=(0.05, 0.10, 0.30, 0.60), long_ids=(), version=2):
    """Synthetic report in the run_experiment.py schema. Utterance-level wer/cer are arbitrary
    and deliberately inconsistent with the corpus values (corpus values are authoritative)."""
    ids = [f"1-2-{i:04d}" for i in range(n)]
    names = ["clean"] + [f"snr_{format(float(x), '.6g')}db" for x in levels]
    conditions = {}
    for c, name in enumerate(names):
        requested = None if c == 0 else float(levels[c - 1])
        utts = []
        for i, uid in enumerate(ids):
            utts.append(
                {
                    "utterance_id": uid,
                    "duration_s": 5.0 + i,
                    "long_form": uid in long_ids,
                    "requested_snr_db": requested,
                    "measured_snr_db": None if requested is None else requested + 1e-9 * (i + 1),
                    "noise_seed": None if requested is None else 1000 * c + i,
                    "reference": f"reference words number {i}",
                    "prediction": f"prediction words number {i}",
                    "wer": 0.9 if i == 0 else 0.0,  # mean != corpus on purpose
                    "cer": 0.8 if i == 0 else 0.1,
                }
            )
        conditions[name] = {
            "type": "clean" if c == 0 else "gaussian_noise",
            "requested_snr_db": requested,
            "mean_measured_snr_db": requested,
            "corpus_wer": corpus_wer[c],
            "corpus_cer": corpus_wer[c] / 2,
            "inference_seconds": 1.5,
            "utterances": utts,
        }
    report = {
        "schema_version": version,
        "model_id": "fake/whisper",
        "dataset": {"id": "openslr/librispeech_asr", "config": "clean", "split": "test"},
        "seed": 42,
        "n_utterances_requested": n,
        "n_utterances": n,
        "utterance_ids": ids,
        "snr_db_levels": [float(x) for x in levels],
        "device": "cpu",
        "elapsed_inference_seconds": 6.0,
        "long_form_threshold_seconds": 30.0,
        "long_form_utterance_ids": list(long_ids),
        "warnings": [],
        "conditions": conditions,
    }
    if version == 1:  # schema 1 had no long-form fields
        for key in ("long_form_threshold_seconds", "long_form_utterance_ids"):
            del report[key]
        for cond in conditions.values():
            for utt in cond["utterances"]:
                del utt["long_form"]
    return report


def write_report(path, report):
    path.write_text(json.dumps(report), encoding="utf-8")
    return path


# --- metric calculations ---------------------------------------------------------
def test_rows_use_stored_corpus_metrics_not_utterance_means():
    report = make_report()
    rows = ar.compute_rows(report)
    utterance_mean = np.mean([u["wer"] for u in report["conditions"]["clean"]["utterances"]])
    assert rows[0]["corpus_wer"] == 0.05
    assert rows[0]["corpus_wer"] != pytest.approx(utterance_mean)
    assert [r["corpus_wer"] for r in rows] == [0.05, 0.10, 0.30, 0.60]
    assert [r["corpus_cer"] for r in rows] == [0.025, 0.05, 0.15, 0.30]


def test_absolute_change_is_percentage_points_and_relative_is_percent_of_clean():
    rows = ar.compute_rows(make_report(corpus_wer=(0.05, 0.10, 0.30, 0.60)))
    by_name = {r["condition"]: r for r in rows}
    assert by_name["snr_20db"]["wer_abs_change_pp"] == pytest.approx(5.0)  # 0.10 - 0.05
    assert by_name["snr_20db"]["wer_rel_change_pct"] == pytest.approx(100.0)  # doubled
    assert by_name["snr_5db"]["wer_abs_change_pp"] == pytest.approx(55.0)
    assert by_name["snr_5db"]["wer_rel_change_pct"] == pytest.approx(1100.0)
    assert by_name["snr_10db"]["cer_abs_change_pp"] == pytest.approx((0.15 - 0.025) * 100)
    assert by_name["snr_10db"]["cer_rel_change_pct"] == pytest.approx(500.0)
    assert by_name["clean"]["wer_abs_change_pp"] == 0.0
    assert by_name["clean"]["wer_rel_change_pct"] == 0.0


def test_improvement_gives_negative_changes():
    rows = ar.compute_rows(make_report(corpus_wer=(0.20, 0.10, 0.30, 0.60)))
    assert rows[1]["wer_abs_change_pp"] == pytest.approx(-10.0)
    assert rows[1]["wer_rel_change_pct"] == pytest.approx(-50.0)


def test_relative_change_undefined_when_clean_is_zero():
    rows = ar.compute_rows(make_report(corpus_wer=(0.0, 0.10, 0.30, 0.60)))
    assert rows[1]["wer_abs_change_pp"] == pytest.approx(10.0)
    assert rows[1]["wer_rel_change_pct"] is None
    text = ar.build_summary(make_report(corpus_wer=(0.0, 0.1, 0.3, 0.6)), rows, "r.json", "0" * 64)
    assert "n/a" in text
    assert ",," in ar.build_csv(rows)  # empty cell, no crash


def test_rows_follow_report_order_and_record_snr_and_sample_size():
    report = make_report(n=6)
    rows = ar.compute_rows(report)
    assert [r["condition"] for r in rows] == NAMES
    assert [r["requested_snr_db"] for r in rows] == [None, 20.0, 10.0, 5.0]
    assert rows[0]["mean_measured_snr_db"] is None
    assert rows[1]["mean_measured_snr_db"] == pytest.approx(20.0, abs=1e-6)
    assert all(r["n_utterances"] == 6 for r in rows)


def test_nice_axis_covers_value_with_round_step():
    for peak in (0.3, 4.6, 23.0, 61.0, 100.0, 0.0):
        step, axis_max = ar.nice_axis(peak)
        assert axis_max >= peak and step > 0
        assert axis_max / step == pytest.approx(round(axis_max / step))
        assert axis_max / step <= 6


# --- validation: ordering, IDs, malformed data -----------------------------------
def test_valid_report_passes_both_schema_versions():
    ar.validate_report(make_report())
    ar.validate_report(make_report(version=1))


def test_custom_snr_levels_are_supported():
    report = make_report(levels=[15, 0.5], corpus_wer=(0.1, 0.2, 0.3, 0.0))
    ar.validate_report(report)
    assert [r["condition"] for r in ar.compute_rows(report)] == ["clean", "snr_15db", "snr_0.5db"]


def test_condition_order_must_be_clean_then_configured_levels():
    report = make_report()
    reordered = dict(reversed(list(report["conditions"].items())))
    with pytest.raises(ar.ReportError, match="expected conditions in order"):
        ar.validate_report({**report, "conditions": reordered})
    swapped = copy.deepcopy(report)
    c = swapped["conditions"]
    swapped["conditions"] = {k: c[k] for k in ("clean", "snr_10db", "snr_20db", "snr_5db")}
    with pytest.raises(ar.ReportError, match="expected conditions in order"):
        ar.validate_report(swapped)


def test_missing_condition_rejected():
    report = make_report()
    del report["conditions"]["snr_10db"]
    with pytest.raises(ar.ReportError, match="expected conditions in order"):
        ar.validate_report(report)


def test_utterance_id_order_mismatch_rejected():
    report = make_report()
    utts = report["conditions"]["snr_10db"]["utterances"]
    utts[0], utts[1] = utts[1], utts[0]
    with pytest.raises(ar.ReportError, match="snr_10db.*different order"):
        ar.validate_report(report)


def test_utterance_id_value_mismatch_rejected():
    report = make_report()
    report["conditions"]["snr_5db"]["utterances"][2]["utterance_id"] = "9-9-9999"
    with pytest.raises(ar.ReportError, match="snr_5db.*position 2"):
        ar.validate_report(report)


def test_missing_utterance_in_one_condition_rejected():
    report = make_report()
    report["conditions"]["snr_20db"]["utterances"].pop()
    with pytest.raises(ar.ReportError, match="snr_20db.*3 utterances"):
        ar.validate_report(report)


def test_reference_text_mismatch_rejected():
    report = make_report()
    report["conditions"]["snr_20db"]["utterances"][1]["reference"] = "something else"
    with pytest.raises(ar.ReportError, match="reference text"):
        ar.validate_report(report)


def test_top_level_id_list_must_match_conditions():
    report = make_report()
    report["utterance_ids"] = list(reversed(report["utterance_ids"]))
    with pytest.raises(ar.ReportError, match="different order"):
        ar.validate_report(report)


DELETE = object()  # sentinel: remove the key instead of replacing its value


def mutate(path_keys, value):
    def apply(report):
        node = report
        for key in path_keys[:-1]:
            node = node[key]
        if value is DELETE:
            del node[path_keys[-1]]
        else:
            node[path_keys[-1]] = value
        return report

    return apply


NOISY = ["conditions", "snr_10db"]
NOISY_UTT = NOISY + ["utterances", 1]


@pytest.mark.parametrize(
    "path_keys,value,match",
    [
        (["schema_version"], 99, "unsupported"),
        (["schema_version"], True, "unsupported"),
        (["model_id"], "", "model_id"),
        (["dataset"], "x", "expected an object"),
        (["seed"], -1, "seed"),
        (["device"], DELETE, "device"),
        (["utterance_ids"], [], "non-empty"),
        (["n_utterances"], 99, "n_utterances"),
        (["snr_db_levels"], [], "non-empty"),
        (["snr_db_levels"], [20, "ten", 5], "finite number"),
        (["conditions"], [], "expected an object"),
        (NOISY + ["corpus_wer"], None, "corpus_wer"),
        (NOISY + ["corpus_wer"], "0.3", "corpus_wer"),
        (NOISY + ["corpus_wer"], True, "corpus_wer"),
        (NOISY + ["corpus_wer"], -0.1, "corpus_wer"),
        (NOISY + ["corpus_cer"], DELETE, "corpus_cer"),
        (NOISY + ["type"], "clean", "type"),
        (NOISY + ["requested_snr_db"], None, "requested_snr_db"),
        (NOISY + ["requested_snr_db"], 11.0, "does not match snr_db_levels"),
        (NOISY + ["utterances"], [], "non-empty"),
        (NOISY_UTT + ["measured_snr_db"], None, "measured_snr_db"),
        (NOISY_UTT + ["measured_snr_db"], DELETE, "measured_snr_db"),
        (NOISY_UTT + ["measured_snr_db"], "10", "measured_snr_db"),
        (NOISY_UTT + ["measured_snr_db"], 14.0, "more than"),
        (NOISY_UTT + ["requested_snr_db"], DELETE, "requested_snr_db"),
        (NOISY_UTT + ["requested_snr_db"], 7.0, "differs from condition"),
        (NOISY_UTT + ["wer"], None, "wer"),
        (NOISY_UTT + ["cer"], -1, "cer"),
        (NOISY_UTT + ["reference"], "  ", "reference"),
        (NOISY_UTT + ["prediction"], 5, "prediction"),
        (NOISY_UTT + ["utterance_id"], "", "utterance_id"),
        (["conditions", "clean", "requested_snr_db"], 20.0, "null for clean"),
        (["conditions", "clean", "utterances", 0, "measured_snr_db"], 20.0, "clean audio"),
    ],
)
def test_malformed_reports_fail_with_clear_message(path_keys, value, match):
    report = mutate(path_keys, value)(make_report())
    with pytest.raises(ar.ReportError, match=match):
        ar.validate_report(report)


@pytest.mark.parametrize("not_a_report", [None, [], "text", 5])
def test_non_object_report_rejected(not_a_report):
    with pytest.raises(ar.ReportError, match="JSON object"):
        ar.validate_report(not_a_report)


def test_duplicate_utterance_ids_rejected():
    report = make_report()
    report["utterance_ids"][1] = report["utterance_ids"][0]
    with pytest.raises(ar.ReportError, match="duplicate"):
        ar.validate_report(report)


def test_load_report_rejects_bad_files(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text("{not json", encoding="utf-8")
    with pytest.raises(ar.ReportError, match="not valid JSON"):
        ar.load_report(bad)
    nan = tmp_path / "nan.json"
    nan.write_text('{"corpus_wer": NaN}', encoding="utf-8")
    with pytest.raises(ar.ReportError, match="NaN"):
        ar.load_report(nan)
    binary = tmp_path / "bin.json"
    binary.write_bytes(b"\xff\xfe\x00")
    with pytest.raises(ar.ReportError, match="UTF-8"):
        ar.load_report(binary)
    with pytest.raises(OSError):
        ar.load_report(tmp_path / "missing.json")


# --- long-form reporting ------------------------------------------------------------
def test_long_form_ids_from_report_list_or_derived_for_schema_1():
    assert ar.long_form_ids(make_report()) == []
    assert ar.long_form_ids(make_report(long_ids=("1-2-0001",))) == ["1-2-0001"]
    old = make_report(version=1)
    old["conditions"]["clean"]["utterances"][2]["duration_s"] = 41.0
    assert ar.long_form_ids(old) == ["1-2-0002"]


# --- outputs -----------------------------------------------------------------------
def run_analyzer(tmp_path, report, out_name="out"):
    source = write_report(tmp_path / "robustness_results.json", report)
    out_dir = tmp_path / out_name
    assert ar.main(["--input", str(source), "--output-dir", str(out_dir)]) == 0
    return out_dir


def test_outputs_are_written_and_output_directory_created(tmp_path, capsys):
    out_dir = run_analyzer(tmp_path, make_report(), "nested/results")
    assert sorted(p.name for p in out_dir.iterdir()) == [
        "summary.md", "wer_cer_by_snr.csv", "wer_cer_by_snr.svg",
    ]
    printed = capsys.readouterr().out
    assert all(name in printed for name in ("summary.md", "wer_cer_by_snr.csv", "wer_cer_by_snr.svg"))


def test_csv_has_one_row_per_condition_with_expected_values(tmp_path):
    report = make_report(corpus_wer=(0.04, 0.08, 0.2, 0.4))
    out_dir = run_analyzer(tmp_path, report)
    with (out_dir / "wer_cer_by_snr.csv").open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        assert reader.fieldnames == ar.CSV_COLUMNS
        rows = list(reader)
    assert [r["condition"] for r in rows] == NAMES
    assert [r["requested_snr_db"] for r in rows] == ["", "20", "10", "5"]
    assert [float(r["corpus_wer"]) for r in rows] == [0.04, 0.08, 0.2, 0.4]
    assert float(rows[3]["corpus_wer_pct"]) == pytest.approx(40.0)
    assert float(rows[1]["wer_abs_change_pp"]) == pytest.approx(4.0)
    assert float(rows[1]["wer_rel_change_pct"]) == pytest.approx(100.0)
    assert {r["n_utterances"] for r in rows} == {"4"}


def test_summary_contains_required_content(tmp_path):
    out_dir = run_analyzer(tmp_path, make_report(n=4))
    text = (out_dir / "summary.md").read_text(encoding="utf-8")
    for needle in (
        "4-utterance pilot using synthetic Gaussian noise",
        "no confidence intervals",
        "generalize to real acoustic environments",
        "No selected utterances used the long-form path",
        "clean_baseline.json",
        "pipeline-based pilot",
        "fake/whisper",
        "openslr/librispeech_asr",
        "Seed | 42",
        "percentage points",
        "relative",
        "Limitations",
        "Sample size",
        "SHA-256",
    ):
        assert needle in text, needle
    for label in ("Clean", "20 dB", "10 dB", "5 dB"):
        assert label in text
    assert "10.00" in text and "+5.00" in text and "+100.0%" in text  # from fixture values


def test_summary_sample_size_follows_data(tmp_path):
    text = (run_analyzer(tmp_path, make_report(n=7)) / "summary.md").read_text(encoding="utf-8")
    assert "7-utterance pilot" in text and "Utterances per condition: **7**" in text


def test_summary_lists_long_form_utterances_when_present(tmp_path):
    report = make_report(long_ids=("1-2-0003",))
    text = (run_analyzer(tmp_path, report) / "summary.md").read_text(encoding="utf-8")
    assert "was used for 1 of 4 utterances" in text and "`1-2-0003`" in text
    assert "No selected utterances used the long-form path" not in text


def test_summary_includes_report_warnings(tmp_path):
    report = make_report()
    report["warnings"] = ["something odd happened"]
    text = (run_analyzer(tmp_path, report) / "summary.md").read_text(encoding="utf-8")
    assert "something odd happened" in text


def test_svg_is_valid_xml_with_a_bar_per_condition_and_metric(tmp_path):
    out_dir = run_analyzer(tmp_path, make_report(corpus_wer=(0.05, 0.10, 0.30, 0.60)))
    root = ET.parse(out_dir / "wer_cer_by_snr.svg").getroot()
    ns = {"s": "http://www.w3.org/2000/svg"}
    assert root.tag == "{http://www.w3.org/2000/svg}svg"
    wer = root.findall(".//s:rect[@class='bar-wer']", ns)
    cer = root.findall(".//s:rect[@class='bar-cer']", ns)
    assert len(wer) == len(cer) == 4
    heights = [float(r.get("height")) for r in wer]
    assert heights == sorted(heights) and heights[0] < heights[-1]  # taller = larger WER
    assert all(float(c.get("height")) < float(w.get("height")) for c, w in zip(cer, wer))
    labels = [t.text for t in root.findall(".//s:text[@class='x-label']", ns)]
    assert labels == ["Clean", "20 dB", "10 dB", "5 dB"]
    all_text = " ".join(t.text or "" for t in root.iter("{http://www.w3.org/2000/svg}text"))
    assert "Error rate (%)" in all_text and "WER" in all_text and "CER" in all_text
    assert "60.00" in all_text and "5.00" in all_text  # value labels from fixture


def test_svg_handles_all_zero_error_rates(tmp_path):
    report = make_report(corpus_wer=(0.0, 0.0, 0.0, 0.0))
    out_dir = run_analyzer(tmp_path, report)
    root = ET.parse(out_dir / "wer_cer_by_snr.svg").getroot()
    assert len(root.findall(".//{http://www.w3.org/2000/svg}rect[@class='bar-wer']")) == 4


def test_outputs_are_byte_for_byte_deterministic(tmp_path):
    report = make_report()
    first = run_analyzer(tmp_path, report, "run1")
    second = run_analyzer(tmp_path, report, "run2")
    for name in ("summary.md", "wer_cer_by_snr.csv", "wer_cer_by_snr.svg"):
        assert (first / name).read_bytes() == (second / name).read_bytes()


def test_analysis_does_not_modify_input(tmp_path):
    source = write_report(tmp_path / "r.json", make_report())
    before = source.read_bytes()
    ar.main(["--input", str(source), "--output-dir", str(tmp_path / "o")])
    assert source.read_bytes() == before


def test_invalid_report_fails_cleanly_and_writes_nothing(tmp_path, capsys):
    report = make_report()
    report["conditions"]["snr_20db"]["utterances"][0]["measured_snr_db"] = None
    source = write_report(tmp_path / "bad.json", report)
    out_dir = tmp_path / "out"
    assert ar.main(["--input", str(source), "--output-dir", str(out_dir)]) == 1
    assert "ERROR" in capsys.readouterr().err
    assert not out_dir.exists()


def test_missing_input_file_fails_cleanly(tmp_path, capsys):
    assert ar.main(["--input", str(tmp_path / "nope.json"), "--output-dir", str(tmp_path)]) == 1
    assert "ERROR" in capsys.readouterr().err


# --- CLI arguments ---------------------------------------------------------------------
def test_cli_parses_windows_style_arguments():
    args = ar.build_parser().parse_args(
        ["--input", r"results\robustness_results.json", "--output-dir", "results"]
    )
    assert args.input == r"results\robustness_results.json" and args.output_dir == "results"


def test_cli_defaults_point_at_results_directory():
    args = ar.build_parser().parse_args([])
    assert args.input.endswith("robustness_results.json")
    assert Path(args.output_dir).name == "results"


# --- compatibility with the real report writer ------------------------------------------
def test_report_from_run_experiment_is_accepted(tmp_path):
    """The analyzer must accept what experiment.run_experiment/save_report actually produce."""
    from asr_robustness.experiment import run_experiment, save_report

    records = []
    for i, text in enumerate(["alpha beta gamma delta", "epsilon zeta eta"]):
        audio = 0.3 * np.sin(2 * np.pi * (200 + 30 * i) * np.arange(2000) / 16000)
        records.append(
            {"utterance_id": f"7-8-{i:04d}", "audio": audio, "sample_rate": 16000, "reference": text}
        )
    records[1]["audio"] = np.full(16000 * 31, 0.1)  # one long-form clip

    class Recognizer:
        model_id, device = "fake/whisper", "cpu"

        def transcribe(self, audio, sample_rate):
            return "alpha beta gamma"

    config = {
        "seed": 3, "n_utterances": 2, "snr_db": [20, 10, 5], "model_id": "fake/whisper",
        "dataset_id": "openslr/librispeech_asr", "dataset_config": "clean", "dataset_split": "test",
    }
    report = run_experiment(config, Recognizer(), load_subset=lambda n_utterances, seed: records)
    path = save_report(report, tmp_path / "results" / "robustness_results.json")

    out_dir = tmp_path / "analysis"
    assert ar.main(["--input", str(path), "--output-dir", str(out_dir)]) == 0
    text = (out_dir / "summary.md").read_text(encoding="utf-8")
    assert "2-utterance pilot" in text and "`7-8-0001`" in text  # long-form clip reported
    rows = ar.compute_rows(json.loads(path.read_text(encoding="utf-8")))
    assert [r["condition"] for r in rows] == NAMES
    assert csv.reader(io.StringIO(ar.build_csv(rows)))
