"""Offline tests for scripts/check_dataset_metadata.py (all HTTP is mocked)."""

import importlib.util
import io
import socket
import urllib.error
from pathlib import Path

import pytest

from asr_robustness import data

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "check_dataset_metadata.py"
spec = importlib.util.spec_from_file_location("check_dataset_metadata", SCRIPT)
cdm = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cdm)

INFO_URL = "https://huggingface.co/api/datasets/openslr/librispeech_asr"
TREE_URL = INFO_URL + "/tree/main/clean/test"

INFO_OK = {
    "id": "openslr/librispeech_asr",
    "cardData": {
        "license": ["cc-by-4.0"],
        "dataset_info": [
            {
                "config_name": "clean",
                "features": [
                    {"name": "file", "dtype": "string"},
                    {"name": "audio", "dtype": {"audio": {"sampling_rate": 16000}}},
                    {"name": "text", "dtype": "string"},
                    {"name": "id", "dtype": "string"},
                ],
                "splits": [
                    {"name": "validation", "num_bytes": 1, "num_examples": 2},
                    {"name": "test", "num_bytes": 367705423, "num_examples": 2620},
                ],
            }
        ],
        "configs": [
            {"config_name": "clean", "data_files": [{"split": "test", "path": "clean/test/*.parquet"}]}
        ],
    },
}

TREE_OK = [
    {"type": "file", "path": "clean/test/0000.parquet", "size": 135, "lfs": {"size": 200_000_000, "oid": "x"}},
    {"type": "file", "path": "clean/test/0001.parquet", "size": 135, "lfs": {"size": 167_705_423, "oid": "y"}},
    {"type": "directory", "path": "clean/test/subdir"},
]


class FakeFetch:
    """Records calls and serves canned JSON (or raises) per URL."""

    def __init__(self, responses):
        self.responses = responses
        self.calls = []

    def __call__(self, url, timeout):
        self.calls.append((url, timeout))
        result = self.responses[url]
        if isinstance(result, Exception):
            raise result
        return result


def run(fetch, capsys, argv=()):
    code = cdm.main(list(argv), fetch=fetch)
    return code, capsys.readouterr()


# --- success ---------------------------------------------------------------
def test_success_reports_files_card_and_unverified_items(capsys):
    fetch = FakeFetch({INFO_URL: INFO_OK, TREE_URL: TREE_OK})
    code, out = run(fetch, capsys)
    assert code == 0
    text = out.out
    assert "clean/test/0000.parquet" in text and "200,000,000 bytes" in text  # LFS size, not pointer size
    assert "files: 2, total reported size: 367,705,423 bytes" in text
    assert "subdir" not in text
    assert "num_examples=2620" in text and "num_bytes=367705423" in text
    assert "audio: " in text and "sampling_rate" in text
    assert "clean/test/*.parquet" in text and "cc-by-4.0" in text
    assert "CONFIRMED BY LIVE API" in text and "DECLARED BY THE DATASET CARD" in text
    assert "NOT AVAILABLE FROM METADATA ALONE" in text
    assert "does not verify the" in text and "Parquet schema" in text
    assert "bytes' and 'path'" in text


def test_success_uses_only_the_two_metadata_urls_with_timeout(capsys):
    fetch = FakeFetch({INFO_URL: INFO_OK, TREE_URL: TREE_OK})
    run(fetch, capsys, ["--timeout", "3"])
    assert [u for u, _ in fetch.calls] == [INFO_URL, TREE_URL]
    assert all(t == 3.0 for _, t in fetch.calls)
    assert not any("/resolve/" in u or u.endswith(".parquet") for u, _ in fetch.calls)


def test_default_timeout_is_short(capsys):
    fetch = FakeFetch({INFO_URL: INFO_OK, TREE_URL: TREE_OK})
    run(fetch, capsys)
    assert all(t == cdm.DEFAULT_TIMEOUT <= 15 for _, t in fetch.calls)


def test_output_is_ascii_for_windows_consoles(capsys):
    fetch = FakeFetch({INFO_URL: INFO_OK, TREE_URL: TREE_OK})
    _, out = run(fetch, capsys)
    out.out.encode("ascii")


@pytest.mark.parametrize("bad", ["0", "-1", "abc"])
def test_rejects_invalid_timeout(bad):
    with pytest.raises(SystemExit):
        cdm.main(["--timeout", bad], fetch=FakeFetch({}))


# --- missing metadata ------------------------------------------------------
def test_missing_files_exit_1(capsys):
    fetch = FakeFetch({INFO_URL: INFO_OK, TREE_URL: []})
    code, out = run(fetch, capsys)
    assert code == 1
    assert "MISSING METADATA" in out.out and "no files listed under clean/test/" in out.out


def test_tree_404_is_missing_metadata_not_network_error(capsys):
    fetch = FakeFetch({INFO_URL: INFO_OK, TREE_URL: cdm.MetadataError("HTTP 404", status=404)})
    code, out = run(fetch, capsys)
    assert code == 1 and "MISSING METADATA" in out.out


def test_missing_card_data_exit_1(capsys):
    fetch = FakeFetch({INFO_URL: {"id": "openslr/librispeech_asr"}, TREE_URL: TREE_OK})
    code, out = run(fetch, capsys)
    assert code == 1
    assert "does not declare config 'clean'" in out.out
    assert "not declared" in out.out
    assert "clean/test/0000.parquet" in out.out  # file listing is still reported


def test_missing_test_split_and_features(capsys):
    info = {"cardData": {"dataset_info": [{"config_name": "clean", "splits": [{"name": "validation"}]}]}}
    fetch = FakeFetch({INFO_URL: info, TREE_URL: TREE_OK})
    code, out = run(fetch, capsys)
    assert code == 1
    assert "does not declare split 'test'" in out.out
    assert "lists no features" in out.out
    assert "no data_files entry" in out.out


def test_dataset_info_as_single_dict_is_accepted():
    info = {"cardData": {"dataset_info": INFO_OK["cardData"]["dataset_info"][0], "configs": INFO_OK["cardData"]["configs"]}}
    card = cdm.parse_card(info)
    assert card["config_declared"] and card["split"]["num_examples"] == 2620


def test_lfs_missing_falls_back_to_plain_size():
    files = cdm.parse_test_files([{"type": "file", "path": "clean/test/a.parquet", "size": 42}])
    assert files == [{"path": "clean/test/a.parquet", "size": 42}]


def test_files_outside_clean_test_are_ignored():
    tree = [{"type": "file", "path": "clean/train.100/0000.parquet", "size": 1}]
    assert cdm.parse_test_files(tree) == []


def test_unexpected_tree_shape_is_an_error():
    with pytest.raises(cdm.MetadataError):
        cdm.parse_test_files({"error": "nope"})


# --- network / API errors ----------------------------------------------------
@pytest.mark.parametrize("failing_url", [INFO_URL, TREE_URL])
def test_network_error_exit_2_with_clear_message(capsys, failing_url):
    responses = {INFO_URL: INFO_OK, TREE_URL: TREE_OK}
    responses[failing_url] = cdm.MetadataError("cannot reach host: name resolution failed")
    code, out = run(FakeFetch(responses), capsys)
    assert code == 2
    assert "ERROR: could not fetch dataset metadata" in out.err
    assert "name resolution failed" in out.err and "Nothing was downloaded" in out.err
    assert out.out == ""


def test_info_404_is_a_hard_error(capsys):
    fetch = FakeFetch({INFO_URL: cdm.MetadataError("HTTP 404", status=404), TREE_URL: TREE_OK})
    code, _out = run(fetch, capsys)
    assert code == 2


class FakeResponse(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def test_fetch_json_success_passes_timeout(monkeypatch):
    seen = {}

    def fake_urlopen(request, timeout):
        seen["timeout"], seen["url"] = timeout, request.full_url
        return FakeResponse(b'{"ok": true}')

    monkeypatch.setattr(cdm.urllib.request, "urlopen", fake_urlopen)
    assert cdm.fetch_json("https://example.test/x", timeout=4) == {"ok": True}
    assert seen == {"timeout": 4, "url": "https://example.test/x"}


@pytest.mark.parametrize(
    "exc,match",
    [
        (urllib.error.URLError(socket.gaierror("Name or service not known")), "cannot reach"),
        (TimeoutError("timed out"), "timed out after 2s"),
        (urllib.error.HTTPError("https://x", 503, "Service Unavailable", {}, None), "HTTP 503"),
        (ConnectionResetError("reset"), "network error"),
    ],
)
def test_fetch_json_translates_network_errors(monkeypatch, exc, match):
    def fake_urlopen(request, timeout):
        raise exc

    monkeypatch.setattr(cdm.urllib.request, "urlopen", fake_urlopen)
    with pytest.raises(cdm.MetadataError, match=match):
        cdm.fetch_json("https://example.test/x", timeout=2)


def test_fetch_json_http_error_keeps_status(monkeypatch):
    def fake_urlopen(request, timeout):
        raise urllib.error.HTTPError("https://x", 404, "Not Found", {}, None)

    monkeypatch.setattr(cdm.urllib.request, "urlopen", fake_urlopen)
    with pytest.raises(cdm.MetadataError) as info:
        cdm.fetch_json("https://example.test/x")
    assert info.value.status == 404


def test_fetch_json_rejects_invalid_json(monkeypatch):
    monkeypatch.setattr(cdm.urllib.request, "urlopen", lambda request, timeout: FakeResponse(b"<html>"))
    with pytest.raises(cdm.MetadataError, match="not valid JSON"):
        cdm.fetch_json("https://example.test/x")


# --- safety / consistency ------------------------------------------------------
def test_script_constants_match_loader():
    assert cdm.DATASET_ID == data.DATASET_ID
    assert cdm.DATASET_CONFIG == data.DATASET_CONFIG
    assert cdm.DATASET_SPLIT == data.DATASET_SPLIT


def test_script_never_loads_the_dataset():
    source = SCRIPT.read_text(encoding="utf-8")
    assert "load_dataset" not in source
    assert "import datasets" not in source and "from datasets" not in source
    assert "/resolve/" not in source
