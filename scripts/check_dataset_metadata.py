#!/usr/bin/env python3
"""Metadata-only check of the Hugging Face dataset ``openslr/librispeech_asr``.

Queries two public Hub API endpoints (standard library only):

  GET /api/datasets/openslr/librispeech_asr                    -> dataset card metadata
  GET /api/datasets/openslr/librispeech_asr/tree/main/clean/test -> files under clean/test/

It never loads the dataset, never reads audio or rows, and never downloads a Parquet file.
This does NOT verify the actual Parquet schema; see section 3 of the report.

Exit codes: 0 = metadata found, 1 = API reached but expected metadata is missing,
2 = network/API failure (or invalid command-line usage).
"""

import argparse
import json
import sys
import urllib.error
import urllib.request

# These mirror asr_robustness.data (a test keeps them in sync); the script stays standalone.
DATASET_ID = "openslr/librispeech_asr"
DATASET_CONFIG = "clean"
DATASET_SPLIT = "test"

REPO_PREFIX = f"{DATASET_CONFIG}/{DATASET_SPLIT}"
API_ROOT = "https://huggingface.co/api/datasets"
DEFAULT_TIMEOUT = 10.0
MAX_RESPONSE_BYTES = 10_000_000

NOT_VERIFIABLE = [
    "the physical Parquet schema (column names/types actually stored in the files)",
    "whether 'audio' is stored as a struct with 'bytes' and 'path' fields",
    "the audio encoding of the stored bytes (FLAC vs WAV) and whether 'bytes' is ever empty",
    "whether 'path' is a bare file name or a usable local path",
    "the real sample rate and channel count of individual clips",
    "the contents of 'id' and 'text' values, row counts per shard, and row-group sizes",
    "that the dataset card matches the Parquet files (the card is written by the dataset authors)",
]


class MetadataError(RuntimeError):
    """Network or API failure while fetching metadata."""

    def __init__(self, message: str, status: int | None = None):
        super().__init__(message)
        self.status = status


def fetch_json(url: str, timeout: float = DEFAULT_TIMEOUT):
    """GET ``url`` and parse JSON. ``timeout`` (seconds) applies to connect and each read."""
    request = urllib.request.Request(
        url,
        headers={"Accept": "application/json", "User-Agent": "asr-robustness-metadata-check/0.1"},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read(MAX_RESPONSE_BYTES)
    except urllib.error.HTTPError as exc:
        raise MetadataError(f"HTTP {exc.code} ({exc.reason}) from {url}", status=exc.code) from exc
    except TimeoutError as exc:
        raise MetadataError(f"timed out after {timeout:g}s contacting {url}") from exc
    except urllib.error.URLError as exc:
        raise MetadataError(f"cannot reach {url}: {exc.reason}") from exc
    except OSError as exc:
        raise MetadataError(f"network error contacting {url}: {exc}") from exc
    try:
        return json.loads(body)
    except ValueError as exc:
        raise MetadataError(f"response from {url} is not valid JSON") from exc


def parse_test_files(tree) -> list[dict]:
    """Extract files under ``clean/test/`` from a Hub tree listing.

    For Git-LFS files the real size is ``lfs.size``; ``size`` is only the pointer size.
    """
    if not isinstance(tree, list):
        raise MetadataError("unexpected response shape for the file listing (expected a list)")
    files = []
    for entry in tree:
        if not isinstance(entry, dict) or entry.get("type") != "file":
            continue
        path = entry.get("path")
        if not isinstance(path, str) or not path.startswith(REPO_PREFIX + "/"):
            continue
        lfs = entry.get("lfs")
        size = lfs.get("size") if isinstance(lfs, dict) else None
        if size is None:
            size = entry.get("size")
        files.append({"path": path, "size": size if isinstance(size, int) else None})
    return sorted(files, key=lambda f: f["path"])


def _as_list(value) -> list:
    if value is None:
        return []
    return value if isinstance(value, list) else [value]


def parse_card(info) -> dict:
    """Extract card-declared metadata for the clean config and test split."""
    card = info.get("cardData") if isinstance(info, dict) else None
    result = {
        "license": None,
        "config_declared": False,
        "data_files": None,
        "split": None,
        "features": None,
    }
    if not isinstance(card, dict):
        return result
    license_ = card.get("license")
    result["license"] = ", ".join(map(str, license_)) if isinstance(license_, list) else license_

    for cfg in _as_list(card.get("configs")):
        if isinstance(cfg, dict) and cfg.get("config_name") == DATASET_CONFIG:
            for item in _as_list(cfg.get("data_files")):
                if isinstance(item, dict) and item.get("split") == DATASET_SPLIT:
                    result["data_files"] = item.get("path")

    for entry in _as_list(card.get("dataset_info")):
        if not isinstance(entry, dict) or entry.get("config_name") != DATASET_CONFIG:
            continue
        result["config_declared"] = True
        features = [
            (f.get("name"), f.get("dtype"))
            for f in _as_list(entry.get("features"))
            if isinstance(f, dict) and f.get("name")
        ]
        result["features"] = features or None
        for split in _as_list(entry.get("splits")):
            if isinstance(split, dict) and split.get("name") == DATASET_SPLIT:
                result["split"] = {
                    "num_examples": split.get("num_examples"),
                    "num_bytes": split.get("num_bytes"),
                }
    return result


def _fmt_size(n) -> str:
    return "size not reported" if n is None else f"{n:,} bytes ({n / 2**20:.1f} MiB)"


def _fmt_dtype(dtype) -> str:
    return dtype if isinstance(dtype, str) else json.dumps(dtype, sort_keys=True, separators=(",", ":"))


def find_missing(files: list[dict], card: dict) -> list[str]:
    missing = []
    if not files:
        missing.append(f"no files listed under {REPO_PREFIX}/ in the repository")
    if not card["config_declared"]:
        missing.append(f"dataset card does not declare config {DATASET_CONFIG!r} (dataset_info)")
    else:
        if card["split"] is None:
            missing.append(f"dataset card does not declare split {DATASET_SPLIT!r} for {DATASET_CONFIG!r}")
        if not card["features"]:
            missing.append(f"dataset card lists no features for config {DATASET_CONFIG!r}")
    if card["data_files"] is None:
        missing.append(f"dataset card has no data_files entry for {DATASET_CONFIG!r}/{DATASET_SPLIT!r}")
    return missing


def build_report(files: list[dict], card: dict, missing: list[str]) -> str:
    lines = [
        f"LibriSpeech metadata check: {DATASET_ID} (config={DATASET_CONFIG}, split={DATASET_SPLIT})",
        "Source: Hugging Face Hub API, metadata only. No dataset loading, no rows, no audio,",
        "no Parquet download.",
        "",
        f"[1] CONFIRMED BY LIVE API - repository files under {REPO_PREFIX}/",
    ]
    if files:
        for f in files:
            lines.append(f"  {f['path']}  {_fmt_size(f['size'])}")
        known = [f["size"] for f in files if f["size"] is not None]
        total = f"{sum(known):,} bytes ({sum(known) / 2**20:.1f} MiB)" if known else "unknown"
        lines.append(f"  files: {len(files)}, total reported size: {total}")
        non_parquet = [f["path"] for f in files if not f["path"].endswith(".parquet")]
        if non_parquet:
            lines.append(f"  note: non-Parquet files present: {', '.join(non_parquet)}")
    else:
        lines.append("  (none)")

    lines += [
        "",
        "[2] DECLARED BY THE DATASET CARD (returned live by the API; authored by the dataset",
        "    maintainers, NOT derived from the Parquet files)",
        f"  license: {card['license'] or 'not declared'}",
        f"  config {DATASET_CONFIG!r} declared: {'yes' if card['config_declared'] else 'no'}",
        f"  data_files for split {DATASET_SPLIT!r}: {card['data_files'] or 'not declared'}",
    ]
    if card["split"]:
        lines.append(
            f"  split {DATASET_SPLIT!r}: num_examples={card['split']['num_examples']}, "
            f"num_bytes={card['split']['num_bytes']}"
        )
    else:
        lines.append(f"  split {DATASET_SPLIT!r}: not declared")
    if card["features"]:
        lines.append("  declared features:")
        for name, dtype in card["features"]:
            lines.append(f"    {name}: {_fmt_dtype(dtype)}")
    else:
        lines.append("  declared features: not declared")

    lines += ["", "[3] NOT AVAILABLE FROM METADATA ALONE (still unverified)"]
    lines += [f"  - {item}" for item in NOT_VERIFIABLE]

    lines.append("")
    if missing:
        lines.append("MISSING METADATA:")
        lines += [f"  - {m}" for m in missing]
        lines.append("Result: incomplete metadata (exit code 1).")
    else:
        lines.append("Result: file listing and card metadata found. This does not verify the")
        lines.append("actual Parquet schema; confirm that with a one-row streaming check (section 3).")
    return "\n".join(lines)


def _positive_float(text: str) -> float:
    value = float(text)
    if not value > 0:
        raise argparse.ArgumentTypeError("timeout must be a positive number of seconds")
    return value


def main(argv=None, fetch=fetch_json) -> int:
    parser = argparse.ArgumentParser(description="Metadata-only check of openslr/librispeech_asr.")
    parser.add_argument("--timeout", type=_positive_float, default=DEFAULT_TIMEOUT,
                        help=f"seconds for connect and each read (default {DEFAULT_TIMEOUT:g})")
    args = parser.parse_args(argv)

    base = f"{API_ROOT}/{DATASET_ID}"
    try:
        info = fetch(base, args.timeout)
        try:
            files = parse_test_files(fetch(f"{base}/tree/main/{REPO_PREFIX}", args.timeout))
        except MetadataError as exc:
            if exc.status != 404:
                raise
            files = []  # path does not exist in the repo: report as missing, not as a network failure
    except MetadataError as exc:
        print(f"ERROR: could not fetch dataset metadata: {exc}", file=sys.stderr)
        print("Check your internet connection, proxy settings (HTTPS_PROXY) and firewall; "
              "huggingface.co must be reachable. Nothing was downloaded.", file=sys.stderr)
        return 2

    card = parse_card(info)
    missing = find_missing(files, card)
    print(build_report(files, card, missing))
    return 1 if missing else 0


if __name__ == "__main__":
    sys.exit(main())
