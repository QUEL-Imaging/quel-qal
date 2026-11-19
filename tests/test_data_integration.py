"""Manually verify the detected branch, applied URLs, and live Git LFS downloads.

Run from the repository root with this checkout installed in your Python
environment (``python -m pip install -e .``) and pytest available. Switch to
any branch containing these tests with ``git switch <branch>`` first.

Branch/URL inspection only (no HTTP, works on all systems)::

    python -m pytest -v -s tests/test_data_integration.py -k detected_branch

Live LFS TIFF and ordinary PNG downloads, plus branch/URL inspection:

macOS / Linux (bash or zsh)::

    QAL_TEST_REMOTE=1 python -m pytest -v -s tests/test_data_integration.py -k "detected_branch or remote"

Windows PowerShell::

    $env:QAL_TEST_REMOTE = "1"
    python -m pytest -v -s tests/test_data_integration.py -k "detected_branch or remote"
    Remove-Item Env:QAL_TEST_REMOTE

Windows Command Prompt::

    set QAL_TEST_REMOTE=1
    python -m pytest -v -s tests/test_data_integration.py -k "detected_branch or remote"
    set QAL_TEST_REMOTE=

For automatic branch selection, leave QAL_DATA_REF and QAL_DATA_REPOSITORY
unset. Their values are printed if configured; explicit overrides take priority.
To test a different remote ref without switching, set QAL_DATA_REF (for example
``QAL_DATA_REF=dev QAL_TEST_REMOTE=1 python -m pytest -v -s tests/test_data_integration.py -k remote``
on macOS/Linux; use $env:QAL_DATA_REF or set QAL_DATA_REF on Windows).

Live tests bypass checkout files and existing caches, use temporary storage,
and verify downloaded hashes and LFS pointer rejection through the real fetcher.
They need network access and a pushed branch with uploaded LFS objects. Missing
remote data fails the test; it does not fall back to main. Without
QAL_TEST_REMOTE=1, live tests are skipped during ordinary pytest runs.

Download ALL registered datasets into a disposable cache, validate them, and
remove the downloaded files automatically (including on failure):

macOS / Linux (bash or zsh)::

    QAL_TEST_REMOTE_ALL=1 python -m pytest -v -s tests/test_data_integration.py -k all_remote_downloads_temporary

Windows PowerShell::

    $env:QAL_TEST_REMOTE_ALL = "1"
    python -m pytest -v -s tests/test_data_integration.py -k all_remote_downloads_temporary
    Remove-Item Env:QAL_TEST_REMOTE_ALL

Windows Command Prompt::

    set QAL_TEST_REMOTE_ALL=1
    python -m pytest -v -s tests/test_data_integration.py -k all_remote_downloads_temporary
    set QAL_TEST_REMOTE_ALL=

This uses the new repository/ref schema and real fetcher in remote-only mode.
It bypasses checkout files and existing caches, and never clears your normal
cache. Every invocation starts empty. It downloads every registered dataset
and may transfer substantial data. Hash/pointer/download errors fail the test;
cleanup still runs. It stops at the first failure. The test is skipped unless
QAL_TEST_REMOTE_ALL=1 (QAL_TEST_REMOTE alone only enables the small smoke tests).
To check another remote branch, also set QAL_DATA_REF before running it.
Files must already be pushed with their LFS objects; no branch fallback occurs.

Optional local hydration check (run ``git lfs pull`` first): set
QAL_TEST_LOCAL_LFS=1 and run this file with ``-k checkout``.
"""
import os
import subprocess
from tempfile import TemporaryDirectory
from urllib.parse import unquote, urlsplit
from pathlib import Path

import pytest

from qal.data import _fetchers as fetch
from qal.data import _source as source


@pytest.mark.parametrize("filename", [
    "concentration_targets/cn_sample_1.tiff",
    "RET_images/ret_sample_1_lf.tiff",
    "RET_images/ret_sample_1_df.tiff",
    "RET_images/ret_sample_2_lf.tiff",
    "RET_images/ret_sample_2_df.tiff",
])
def test_detected_branch_and_applied_url(filename):
    """Compare automatic detection with Git and inspect the fetcher's real URL."""
    checkout = source.checkout_source()
    selected = source.resolve_source()
    if checkout is not None:
        root = checkout.checkout.parent.parent
        branch = subprocess.run(
            ["git", "-C", str(root), "symbolic-ref", "--quiet", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5,
        ).stdout.strip()
        expected = branch or subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"], text=True, timeout=5,
        ).strip()
        assert checkout.ref == expected
        assert selected.ref == (os.environ.get("QAL_DATA_REF") or expected)
        print(f"\nDetected checkout branch/SHA: {expected}")
    else:
        print("\nNo package Git checkout detected; using build metadata or production fallback.")

    url = fetch._create_image_fetcher(source=selected).get_url(filename)
    parts = urlsplit(url)
    segments = parts.path.split("/")
    assert parts.scheme == "https"
    assert parts.netloc == "media.githubusercontent.com"
    assert segments[1] == "media"
    assert "/".join(segments[2:4]) == selected.repository
    assert unquote(segments[4]) == selected.ref  # Slash-containing refs stay one URL component.
    assert unquote("/".join(segments[5:])) == f"qal/data/{filename}"
    print(f"QAL_DATA_REF override: {os.environ.get('QAL_DATA_REF') or '(unset)'}")
    print(f"QAL_DATA_REPOSITORY override: {os.environ.get('QAL_DATA_REPOSITORY') or '(unset)'}")
    print(f"Selected repository: {selected.repository}")
    print(f"Selected data reference: {selected.ref}")
    print(f"Applied Git LFS URL: {url}")


@pytest.mark.skipif(os.environ.get("QAL_TEST_LOCAL_LFS") != "1", reason="requires git lfs pull")
def test_all_registered_checkout_data_without_network(monkeypatch):
    monkeypatch.delenv("QAL_DATA_REF", raising=False)
    monkeypatch.delenv("QAL_DATA_REPOSITORY", raising=False)
    monkeypatch.delenv("QAL_DATA_ROOT", raising=False)
    monkeypatch.delenv("QAL_DATA_SOURCE", raising=False)
    checkout = source.resolve_source().checkout
    assert checkout is not None
    monkeypatch.setattr(fetch.pooch, "retrieve", lambda **kw: pytest.fail("checkout requires a download"))
    for filename in fetch.registry:
        assert fetch._fetch(filename) == str(checkout / filename)


@pytest.mark.skipif(os.environ.get("QAL_TEST_REMOTE") != "1", reason="requires real GitHub network access")
@pytest.mark.parametrize("filename", [
    "concentration_targets/cn_sample_1.tiff",
    "RET_images/ret_sample_1_lf.tiff",
    "RET_images/ret_sample_1_df.tiff",
    "RET_images/ret_sample_2_lf.tiff",
    "RET_images/ret_sample_2_df.tiff",
    "USAF1951_template/res_source.png",
])
def test_remote_selected_reference(monkeypatch, tmp_path, filename):
    monkeypatch.setattr(fetch.pooch, "os_cache", lambda _: tmp_path / "cache")
    monkeypatch.setenv("QAL_DATA_SOURCE", "remote")
    selected = source.resolve_source()
    # CI provides these separately from the resolver inputs to catch accidental
    # hardcoded default refs/repositories in the implementation.
    if os.environ.get("QAL_EXPECT_DATA_REF"):
        assert selected.ref == os.environ["QAL_EXPECT_DATA_REF"]
    if os.environ.get("QAL_EXPECT_DATA_REPOSITORY"):
        assert selected.repository == os.environ["QAL_EXPECT_DATA_REPOSITORY"]
    url = fetch._create_image_fetcher(source=selected).get_url(filename)
    print(f"\nLive fetch: {selected.repository} at {selected.ref}")
    print(f"Requested URL: {url}")
    result = Path(fetch._fetch(filename))
    assert fetch._valid_file(result, filename)
    print(f"Validated download: {filename} ({result.stat().st_size} bytes)")


@pytest.mark.skipif(
    os.environ.get("QAL_TEST_REMOTE_ALL") != "1",
    reason="requires explicit opt-in to download every registered remote dataset",
)
def test_all_remote_downloads_temporary(monkeypatch):
    """Validate every remote file, then discard only this test's downloads."""
    selected = source.resolve_source()
    print(f"\nValidating all {len(fetch.registry)} datasets: "
          f"{selected.repository} at {selected.ref}")
    with TemporaryDirectory(prefix="qal-all-remote-") as temporary:
        cache_root = Path(temporary)
        with monkeypatch.context() as isolated:
            isolated.setattr(fetch.pooch, "os_cache", lambda _: cache_root / "cache")
            isolated.setenv("QAL_DATA_SOURCE", "remote")
            try:
                print(f"Disposable cache: {fetch.get_cache_dir()}")
                fetch.download_all()
                for filename in fetch.registry:
                    downloaded = Path(fetch.get_cache_dir()) / filename
                    assert fetch._valid_file(downloaded, filename), filename
                    print(f"Validated: {filename} ({downloaded.stat().st_size} bytes)")
            finally:
                print("Removing disposable downloads.")
    assert not cache_root.exists()
    print("All remote datasets validated; disposable downloads removed.")
