#!/usr/bin/env python3
"""Turn ``results/robustness_results.json`` into a Markdown summary, a CSV and an SVG chart.

Standard library only; offline and deterministic (same input bytes -> same output bytes).
It never runs inference, loads Whisper, or touches a dataset.

Windows:
    python scripts\\analyze_results.py --input results\\robustness_results.json --output-dir results

Outputs (in ``--output-dir``): ``summary.md``, ``wer_cer_by_snr.csv``, ``wer_cer_by_snr.svg``.

The condition-level ``corpus_wer`` / ``corpus_cer`` stored in the JSON are the authoritative
metrics and are used as-is. Per-utterance values are only validated, never averaged into a
replacement. Reports from ``run_experiment.py`` (schema versions 1 and 2) are accepted.
"""

import argparse
import csv
import hashlib
import json
import math
import sys
from pathlib import Path
from xml.sax.saxutils import escape

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SUPPORTED_SCHEMA_VERSIONS = (1, 2)
CLEAN = "clean"
SNR_TOLERANCE_DB = 0.1  # |measured - requested| above this means a corrupted report
DEFAULT_LONG_FORM_SECONDS = 30.0

CSV_COLUMNS = [
    "condition",
    "requested_snr_db",
    "mean_measured_snr_db",
    "n_utterances",
    "corpus_wer",
    "corpus_cer",
    "corpus_wer_pct",
    "corpus_cer_pct",
    "wer_abs_change_pp",
    "cer_abs_change_pp",
    "wer_rel_change_pct",
    "cer_rel_change_pct",
]


class ReportError(ValueError):
    """The experiment report is malformed, incomplete or internally inconsistent."""


# --- loading and validation ----------------------------------------------------
def _reject_constant(name):
    raise ReportError(f"invalid JSON constant {name!r} (NaN/Infinity are not allowed)")


def load_report(path) -> tuple[dict, str]:
    """Read a report; return ``(report, sha256-of-file-bytes)``. Raises ReportError/OSError."""
    raw = Path(path).read_bytes()
    try:
        report = json.loads(raw.decode("utf-8"), parse_constant=_reject_constant)
    except UnicodeDecodeError as exc:
        raise ReportError(f"{path}: not valid UTF-8: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise ReportError(f"{path}: not valid JSON: {exc}") from exc
    return report, hashlib.sha256(raw).hexdigest()


def _is_number(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _number(value, where: str, minimum=None) -> float:
    if not _is_number(value) or not math.isfinite(value):
        raise ReportError(f"{where}: expected a finite number, got {value!r}")
    if minimum is not None and value < minimum:
        raise ReportError(f"{where}: expected a value >= {minimum}, got {value!r}")
    return float(value)


def _string(value, where: str, allow_empty=False) -> str:
    if not isinstance(value, str) or (not allow_empty and not value.strip()):
        raise ReportError(f"{where}: expected a non-empty string, got {value!r}")
    return value


def _require(mapping, key: str, where: str):
    if not isinstance(mapping, dict):
        raise ReportError(f"{where}: expected an object, got {type(mapping).__name__}")
    if key not in mapping:
        raise ReportError(f"{where}: missing required key {key!r}")
    return mapping[key]


def _fmt_snr_key(value: float) -> str:
    return format(float(value), ".6g")


def condition_name_for(snr_db: float) -> str:
    """Condition key used by run_experiment.py, e.g. ``snr_20db``."""
    return f"snr_{_fmt_snr_key(snr_db)}db"


def validate_report(report: object) -> dict:
    """Check the whole report; raise ReportError with a precise location on any problem."""
    if not isinstance(report, dict):
        raise ReportError(f"report: expected a JSON object, got {type(report).__name__}")

    version = _require(report, "schema_version", "report")
    if version not in SUPPORTED_SCHEMA_VERSIONS or isinstance(version, bool):
        raise ReportError(
            f"report.schema_version: unsupported {version!r}; supported: {SUPPORTED_SCHEMA_VERSIONS}"
        )
    _string(_require(report, "model_id", "report"), "report.model_id")
    dataset = _require(report, "dataset", "report")
    for key in ("id", "config", "split"):
        _string(_require(dataset, key, "report.dataset"), f"report.dataset.{key}")
    seed = _require(report, "seed", "report")
    if not _is_number(seed) or seed != int(seed) or seed < 0:
        raise ReportError(f"report.seed: expected a non-negative integer, got {seed!r}")
    _string(_require(report, "device", "report"), "report.device")

    top_ids = _require(report, "utterance_ids", "report")
    if not isinstance(top_ids, list) or not top_ids:
        raise ReportError("report.utterance_ids: expected a non-empty list")
    for i, uid in enumerate(top_ids):
        _string(uid, f"report.utterance_ids[{i}]")
    if len(set(top_ids)) != len(top_ids):
        raise ReportError("report.utterance_ids: contains duplicate IDs")
    n = _require(report, "n_utterances", "report")
    if n != len(top_ids) or isinstance(n, bool):
        raise ReportError(f"report.n_utterances: {n!r} does not match {len(top_ids)} utterance_ids")

    levels = _require(report, "snr_db_levels", "report")
    if not isinstance(levels, list) or not levels:
        raise ReportError("report.snr_db_levels: expected a non-empty list")
    for i, level in enumerate(levels):
        _number(level, f"report.snr_db_levels[{i}]")

    conditions = _require(report, "conditions", "report")
    if not isinstance(conditions, dict):
        raise ReportError("report.conditions: expected an object")
    expected_names = [CLEAN] + [condition_name_for(x) for x in levels]
    if list(conditions) != expected_names:
        raise ReportError(
            f"report.conditions: expected conditions in order {expected_names}, "
            f"found {list(conditions)}"
        )

    reference_texts = None
    for name, cond in conditions.items():
        where = f"conditions[{name!r}]"
        is_clean = name == CLEAN
        expected_type = "clean" if is_clean else "gaussian_noise"
        if _require(cond, "type", where) != expected_type:
            raise ReportError(f"{where}.type: expected {expected_type!r}, got {cond['type']!r}")
        _number(_require(cond, "corpus_wer", where), f"{where}.corpus_wer", minimum=0)
        _number(_require(cond, "corpus_cer", where), f"{where}.corpus_cer", minimum=0)

        requested = _require(cond, "requested_snr_db", where)
        if is_clean:
            if requested is not None:
                raise ReportError(f"{where}.requested_snr_db: must be null for clean audio")
        else:
            level = levels[expected_names.index(name) - 1]
            requested = _number(requested, f"{where}.requested_snr_db")
            if requested != float(level):
                raise ReportError(
                    f"{where}.requested_snr_db: {requested!r} does not match snr_db_levels entry {level!r}"
                )

        utterances = _require(cond, "utterances", where)
        if not isinstance(utterances, list) or not utterances:
            raise ReportError(f"{where}.utterances: expected a non-empty list")
        ids = []
        for i, utt in enumerate(utterances):
            uw = f"{where}.utterances[{i}]"
            ids.append(_string(_require(utt, "utterance_id", uw), f"{uw}.utterance_id"))
            _string(_require(utt, "reference", uw), f"{uw}.reference")
            _string(_require(utt, "prediction", uw), f"{uw}.prediction", allow_empty=True)
            _number(_require(utt, "wer", uw), f"{uw}.wer", minimum=0)
            _number(_require(utt, "cer", uw), f"{uw}.cer", minimum=0)
            utt_requested = _require(utt, "requested_snr_db", uw)
            measured = _require(utt, "measured_snr_db", uw)
            if is_clean:
                if utt_requested is not None or measured is not None:
                    raise ReportError(f"{uw}: clean audio must have null requested/measured SNR")
            else:
                utt_requested = _number(utt_requested, f"{uw}.requested_snr_db")
                measured = _number(measured, f"{uw}.measured_snr_db")
                if utt_requested != requested:
                    raise ReportError(
                        f"{uw}.requested_snr_db: {utt_requested!r} differs from condition value {requested!r}"
                    )
                if abs(measured - requested) > SNR_TOLERANCE_DB:
                    raise ReportError(
                        f"{uw}.measured_snr_db: {measured!r} dB is more than {SNR_TOLERANCE_DB} dB "
                        f"from the requested {requested!r} dB"
                    )

        if ids != top_ids:
            raise ReportError(_id_mismatch_message(name, ids, top_ids))
        texts = [u["reference"] for u in utterances]
        if reference_texts is None:
            reference_texts = texts
        elif texts != reference_texts:
            bad = next(i for i, (a, b) in enumerate(zip(texts, reference_texts)) if a != b)
            raise ReportError(
                f"{where}: reference text of utterance {ids[bad]!r} differs from the clean condition"
            )
    return report


def _id_mismatch_message(name: str, ids: list, expected: list) -> str:
    where = f"conditions[{name!r}].utterances"
    if len(ids) != len(expected):
        return f"{where}: has {len(ids)} utterances but report.utterance_ids has {len(expected)}"
    if sorted(ids) == sorted(expected):
        return f"{where}: utterance IDs are in a different order than report.utterance_ids"
    bad = next(i for i, (a, b) in enumerate(zip(ids, expected)) if a != b)
    return (
        f"{where}: utterance IDs differ from report.utterance_ids at position {bad} "
        f"({ids[bad]!r} vs {expected[bad]!r})"
    )


# --- metrics ---------------------------------------------------------------------
def _change(value: float, baseline: float) -> tuple[float, "float | None"]:
    """Absolute change in percentage points and relative change in percent (None if undefined)."""
    absolute_pp = (value - baseline) * 100.0
    relative = (value - baseline) / baseline * 100.0 if baseline > 0 else None
    return absolute_pp, relative


def compute_rows(report: dict) -> list[dict]:
    """One row per condition in report order, using the stored corpus-level metrics."""
    conditions = report["conditions"]
    clean = conditions[CLEAN]
    rows = []
    for name, cond in conditions.items():
        wer, cer = float(cond["corpus_wer"]), float(cond["corpus_cer"])
        wer_abs, wer_rel = _change(wer, float(clean["corpus_wer"]))
        cer_abs, cer_rel = _change(cer, float(clean["corpus_cer"]))
        measured = [u["measured_snr_db"] for u in cond["utterances"] if u["measured_snr_db"] is not None]
        rows.append(
            {
                "condition": name,
                "requested_snr_db": cond["requested_snr_db"],
                "mean_measured_snr_db": sum(measured) / len(measured) if measured else None,
                "n_utterances": len(cond["utterances"]),
                "corpus_wer": wer,
                "corpus_cer": cer,
                "corpus_wer_pct": wer * 100.0,
                "corpus_cer_pct": cer * 100.0,
                "wer_abs_change_pp": wer_abs,
                "cer_abs_change_pp": cer_abs,
                "wer_rel_change_pct": wer_rel,
                "cer_rel_change_pct": cer_rel,
            }
        )
    return rows


def long_form_ids(report: dict) -> list[str]:
    """Utterance IDs that needed Whisper long-form transcription (from the report itself)."""
    listed = report.get("long_form_utterance_ids")
    if isinstance(listed, list):
        return [str(x) for x in listed]
    threshold = report.get("long_form_threshold_seconds", DEFAULT_LONG_FORM_SECONDS)
    threshold = threshold if _is_number(threshold) else DEFAULT_LONG_FORM_SECONDS
    ids = []
    for utt in report["conditions"][CLEAN]["utterances"]:
        if utt.get("long_form") is True or (
            _is_number(utt.get("duration_s")) and utt["duration_s"] > threshold
        ):
            ids.append(utt["utterance_id"])
    return ids


def condition_label(row: dict) -> str:
    if row["requested_snr_db"] is None:
        return "Clean"
    return f"{_fmt_snr_key(row['requested_snr_db'])} dB"


# --- CSV -------------------------------------------------------------------------
def _csv_value(value):
    if value is None:
        return ""
    if isinstance(value, float):
        return format(value, ".10g")
    return value


def build_csv(rows: list[dict]) -> str:
    import io

    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=CSV_COLUMNS, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({key: _csv_value(row[key]) for key in CSV_COLUMNS})
    return buffer.getvalue()


# --- Markdown --------------------------------------------------------------------
def _pct(value: float) -> str:
    return f"{value:.2f}"


def _pp(value: float) -> str:
    return f"{value:+.2f}"


def _rel(value) -> str:
    return "n/a (clean = 0)" if value is None else f"{value:+.1f}%"


def _snr(value) -> str:
    return "—" if value is None else f"{value:.2f}"


def build_summary(report: dict, rows: list[dict], source_name: str, source_sha256: str) -> str:
    n = len(report["utterance_ids"])
    levels = ", ".join(_fmt_snr_key(x) for x in report["snr_db_levels"])
    long_ids = long_form_ids(report)
    condition_list = ", ".join("`" + r["condition"] + "`" for r in rows)
    clean_utts = report["conditions"][CLEAN]["utterances"]
    ref_words = sum(len(u["reference"].split()) for u in clean_utts)
    ref_chars = sum(len(u["reference"]) for u in clean_utts)

    lines = [
        "# ASR robustness results: Whisper under synthetic Gaussian noise",
        "",
        (f"**This is a {n}-utterance pilot using synthetic Gaussian noise.** It reports no "
         "confidence intervals and makes no claim that the results generalize to real acoustic "
         "environments."),
        "",
        "## Configuration",
        "",
        "| Setting | Value |",
        "|---|---|",
        f"| Model | `{report['model_id']}` |",
        (f"| Dataset | `{report['dataset']['id']}` (config `{report['dataset']['config']}`, "
         f"split `{report['dataset']['split']}`) |"),
        f"| Seed | {int(report['seed'])} |",
        f"| Utterances | {n} (identical IDs and references in every condition) |",
        f"| Conditions | {condition_list} |",
        f"| Requested SNR levels (dB) | {levels} |",
        f"| Device | `{report['device']}` |",
        f"| Report schema version | {report['schema_version']} |",
        f"| Input file | `{source_name}` (SHA-256 `{source_sha256}`) |",
        "",
        "## Sample size",
        "",
        (f"- Utterances per condition: **{n}** ({len(rows)} conditions, "
         f"{n * len(rows)} transcriptions in total)."),
        (f"- Reference size: {ref_words} words and {ref_chars} characters "
         "(whitespace-split, before the scoring normalization)."),
    ]
    if long_ids:
        lines.append(
            f"- Long-form transcription (clips over {_fmt_snr_key(report.get('long_form_threshold_seconds', DEFAULT_LONG_FORM_SECONDS))} s) "
            f"was used for {len(long_ids)} of {n} utterances: " + ", ".join(f"`{i}`" for i in long_ids) + "."
        )
    else:
        lines.append(
            f"- No selected utterances used the long-form path: all {n} fit within one 30-second "
            "Whisper window."
        )

    lines += [
        "",
        "## Corpus-level error rates",
        "",
        ("Corpus WER/CER are total errors divided by total reference words/characters, taken "
         "directly from the experiment JSON (not averaged over utterances)."),
        "",
        "| Condition | Requested SNR (dB) | Mean measured SNR (dB) | Corpus WER (%) | Corpus CER (%) |",
        "|---|---|---|---|---|",
    ]
    for r in rows:
        requested = "—" if r["requested_snr_db"] is None else _fmt_snr_key(r["requested_snr_db"])
        lines.append(
            f"| {condition_label(r)} | {requested} | {_snr(r['mean_measured_snr_db'])} | "
            f"{_pct(r['corpus_wer_pct'])} | {_pct(r['corpus_cer_pct'])} |"
        )

    lines += [
        "",
        "## Change versus clean",
        "",
        ("Absolute change is in percentage points (pp); relative change is the change as a "
         "percentage of the clean value. Both use the `clean` condition of this same report."),
        "",
        "| Condition | WER change (pp) | WER change (relative) | CER change (pp) | CER change (relative) |",
        "|---|---|---|---|---|",
    ]
    for r in rows[1:]:
        lines.append(
            f"| {condition_label(r)} | {_pp(r['wer_abs_change_pp'])} | {_rel(r['wer_rel_change_pct'])} | "
            f"{_pp(r['cer_abs_change_pp'])} | {_rel(r['cer_rel_change_pct'])} |"
        )

    worst = max(rows, key=lambda r: r["corpus_wer"])
    lines += [
        "",
        (f"Highest corpus WER in this pilot: **{condition_label(worst)}** "
         f"({_pct(worst['corpus_wer_pct'])}%)."),
        "",
        "## Baseline note",
        "",
        ("The earlier `results/clean_baseline.json` came from the earlier pipeline-based pilot "
         "(a different inference path). Compare noisy conditions only against the `clean` "
         "condition in `robustness_results.json`, which uses the same code path, utterances and "
         "decoding settings as the noisy conditions."),
        "",
        "## Limitations",
        "",
        (f"- **Small pilot:** {n} utterances from one LibriSpeech test-clean subset; single seed, "
         "single run, one noise realization per utterance and SNR level."),
        ("- **No uncertainty estimates:** no confidence intervals, significance tests or "
         "repeated-seed variability are reported, so small differences should not be interpreted."),
        ("- **Synthetic noise only:** white Gaussian noise added at a power-based SNR. This is not "
         "evidence about real acoustic environments (colored or non-stationary noise, reverberation, "
         "microphones, accents, other languages)."),
        (f"- **One model and decoding setup:** `{report['model_id']}` with greedy decoding; "
         "results do not transfer to other model sizes or decoders."),
        "- **Read speech:** LibriSpeech test-clean is clean, read, English audiobook speech.",
        ("- **Scoring policy:** WER/CER use the project's normalization (lowercasing, apostrophes "
         "removed, other punctuation treated as spaces)."),
        ("- **Relative changes** are unstable when the clean error rate is small; prefer the "
         "absolute (pp) changes."),
        ("- **Inference times** in the JSON are not benchmarks (the first condition can include "
         "device warm-up) and are not analysed here."),
    ]
    if report.get("warnings"):
        lines += ["", "## Warnings recorded by the experiment", ""]
        lines += [f"- {w}" for w in report["warnings"]]
    return "\n".join(lines) + "\n"


# --- SVG -------------------------------------------------------------------------
WER_COLOR = "#1f5fa8"
CER_COLOR = "#d9731a"


def nice_axis(max_value: float, max_ticks: int = 5) -> tuple[float, float]:
    """Return ``(step, axis_max)`` with a round step so the axis covers ``max_value``."""
    if not max_value > 0:
        return 1.0, 1.0
    target = max_value / max_ticks
    magnitude = 10.0 ** math.floor(math.log10(target))
    step = next(m * magnitude for m in (1, 2, 2.5, 5, 10) if m * magnitude >= target * (1 - 1e-12))
    axis_max = math.ceil(max_value / step - 1e-9) * step
    return step, axis_max


def build_svg(report: dict, rows: list[dict]) -> str:
    n = len(report["utterance_ids"])
    width, height = 760, 470
    left, right, top, bottom = 72, 28, 96, 96
    plot_w, plot_h = width - left - right, height - top - bottom
    y_base = top + plot_h

    peak = max(max(r["corpus_wer_pct"], r["corpus_cer_pct"]) for r in rows)
    step, axis_max = nice_axis(peak * 1.12)

    def y_of(value: float) -> float:
        return y_base - value / axis_max * plot_h

    out = [
        (f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
         f'viewBox="0 0 {width} {height}" role="img" font-family="Helvetica, Arial, sans-serif">'),
        "<title>Corpus WER and CER by noise condition</title>",
        (f"<desc>Grouped bar chart of corpus-level WER and CER (percent) for "
         f"{', '.join(escape(condition_label(r)) for r in rows)}; {n} utterances, synthetic "
         "Gaussian noise.</desc>"),
        f'<rect width="{width}" height="{height}" fill="#ffffff"/>',
        (f'<text x="{width / 2:.1f}" y="30" text-anchor="middle" font-size="18" font-weight="bold" '
         'fill="#222222">Whisper WER and CER under synthetic Gaussian noise</text>'),
        (f'<text x="{width / 2:.1f}" y="50" text-anchor="middle" font-size="12" fill="#555555">'
         f"Corpus-level metrics, {n}-utterance pilot, no confidence intervals</text>"),
    ]

    # legend
    legend_x = width / 2 - 70
    for i, (label, color) in enumerate((("WER", WER_COLOR), ("CER", CER_COLOR))):
        x = legend_x + i * 80
        out.append(f'<rect x="{x:.1f}" y="62" width="14" height="14" fill="{color}"/>')
        out.append(f'<text x="{x + 20:.1f}" y="74" font-size="13" fill="#222222">{label}</text>')

    # grid and y axis
    ticks = round(axis_max / step)
    for i in range(ticks + 1):
        value = i * step
        y = y_of(value)
        out.append(
            f'<line x1="{left}" y1="{y:.2f}" x2="{left + plot_w}" y2="{y:.2f}" '
            f'stroke="{"#444444" if i == 0 else "#dddddd"}" stroke-width="1"/>'
        )
        out.append(
            f'<text x="{left - 8}" y="{y + 4:.2f}" text-anchor="end" font-size="12" '
            f'fill="#444444">{value:.10g}</text>'
        )
    out.append(
        f'<text x="18" y="{top + plot_h / 2:.1f}" text-anchor="middle" font-size="13" fill="#222222" '
        f'transform="rotate(-90 18 {top + plot_h / 2:.1f})">Error rate (%)</text>'
    )

    # bars
    group_w = plot_w / len(rows)
    bar_w = min(54.0, group_w * 0.3)
    gap = 6.0
    for i, r in enumerate(rows):
        center = left + group_w * (i + 0.5)
        label = condition_label(r)
        for j, (key, cls, color, metric) in enumerate(
            (
                ("corpus_wer_pct", "bar-wer", WER_COLOR, "WER"),
                ("corpus_cer_pct", "bar-cer", CER_COLOR, "CER"),
            )
        ):
            value = r[key]
            x = center - bar_w - gap / 2 if j == 0 else center + gap / 2
            bar_h = value / axis_max * plot_h
            out.append(
                f'<rect class="{cls}" x="{x:.2f}" y="{y_base - bar_h:.2f}" width="{bar_w:.2f}" '
                f'height="{bar_h:.2f}" fill="{color}"><title>{metric}, {escape(label)}: '
                f"{value:.2f}%</title></rect>"
            )
            out.append(
                f'<text x="{x + bar_w / 2:.2f}" y="{y_base - bar_h - 5:.2f}" text-anchor="middle" '
                f'font-size="12" fill="#222222">{value:.2f}</text>'
            )
        out.append(
            f'<text class="x-label" x="{center:.2f}" y="{y_base + 20}" text-anchor="middle" '
            f'font-size="13" fill="#222222">{escape(label)}</text>'
        )
        if r["mean_measured_snr_db"] is not None:
            out.append(
                f'<text x="{center:.2f}" y="{y_base + 36}" text-anchor="middle" font-size="11" '
                f'fill="#666666">measured {r["mean_measured_snr_db"]:.2f} dB</text>'
            )
    out.append(
        f'<text x="{left + plot_w / 2:.1f}" y="{height - 22}" text-anchor="middle" font-size="13" '
        'fill="#222222">Condition (requested signal-to-noise ratio)</text>'
    )
    out.append("</svg>")
    return "\n".join(out) + "\n"


# --- CLI -------------------------------------------------------------------------
def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(text)


def analyze(input_path, output_dir) -> list[Path]:
    """Validate the report and write the three outputs; return the written paths."""
    report, digest = load_report(input_path)
    validate_report(report)
    rows = compute_rows(report)
    output_dir = Path(output_dir)
    targets = {
        "summary.md": build_summary(report, rows, Path(input_path).name, digest),
        "wer_cer_by_snr.csv": build_csv(rows),
        "wer_cer_by_snr.svg": build_svg(report, rows),
    }
    written = []
    for name, text in targets.items():
        write_text(output_dir / name, text)
        written.append(output_dir / name)
    return written


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Summarize results/robustness_results.json (offline, deterministic)."
    )
    parser.add_argument(
        "--input",
        default=str(PROJECT_ROOT / "results" / "robustness_results.json"),
        help="experiment JSON report (default: results/robustness_results.json)",
    )
    parser.add_argument(
        "--output-dir",
        default=str(PROJECT_ROOT / "results"),
        help="directory for summary.md, wer_cer_by_snr.csv and wer_cer_by_snr.svg (default: results)",
    )
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        written = analyze(args.input, args.output_dir)
    except (ReportError, OSError) as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    for path in written:
        print(f"Wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
