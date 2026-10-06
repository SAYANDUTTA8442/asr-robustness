#!/usr/bin/env python3
"""Run the ASR robustness experiment (clean + Gaussian noise at configured SNRs).

Windows:
    python scripts\\run_experiment.py --config configs\\baseline.yaml ^
        --output results\\robustness_results.json --device auto
"""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

try:
    import asr_robustness  # noqa: F401
except ImportError:  # not pip-installed: fall back to the source tree next to this script
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from asr_robustness.experiment import load_config, run_experiment, save_report
from asr_robustness.model import load_recognizer, validate_device_name


def _device(text: str) -> str:
    try:
        return validate_device_name(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the ASR robustness experiment.")
    parser.add_argument(
        "--config",
        default=str(PROJECT_ROOT / "configs" / "baseline.yaml"),
        help="YAML configuration file (default: configs/baseline.yaml)",
    )
    parser.add_argument(
        "--output",
        default=str(PROJECT_ROOT / "results" / "robustness_results.json"),
        help="JSON report path; parent directories are created (default: results/robustness_results.json)",
    )
    parser.add_argument(
        "--device",
        type=_device,
        default="auto",
        help="'auto' (cuda if available, else cpu), 'cpu', 'cuda' or 'cuda:N' (default: auto)",
    )
    return parser


def format_summary(report: dict) -> str:
    lines = [
        (f"model={report['model_id']} device={report['device']} seed={report['seed']} "
         f"utterances={report['n_utterances']}"),
        f"{'condition':<12} {'corpus WER':>11} {'corpus CER':>11} {'infer s':>9}",
    ]
    for name, cond in report["conditions"].items():
        lines.append(
            f"{name:<12} {cond['corpus_wer']:>11.4f} {cond['corpus_cer']:>11.4f} "
            f"{cond['inference_seconds']:>9.2f}"
        )
    lines += [f"WARNING: {w}" for w in report["warnings"]]
    return "\n".join(lines)


def main(argv=None, recognizer_factory=load_recognizer, load_subset=None) -> int:
    """CLI body. ``recognizer_factory`` and ``load_subset`` are injectable for tests."""
    args = build_parser().parse_args(argv)
    try:
        config = load_config(args.config)
        recognizer = recognizer_factory(config["model_id"], args.device)  # model loaded once
        kwargs = {} if load_subset is None else {"load_subset": load_subset}
        report = run_experiment(config, recognizer, **kwargs)
        output = save_report(report, args.output)
    except (OSError, ValueError, RuntimeError, ImportError) as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(format_summary(report))
    print(f"Report written to {output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
