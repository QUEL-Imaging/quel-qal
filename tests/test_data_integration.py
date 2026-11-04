"""Opt-in hydrated-checkout and real GitHub downloads; no mocked networking."""
import os
from pathlib import Path

import pytest

from qal.data import _fetchers as fetch
from qal.data import _source as source


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
@pytest.mark.parametrize("filename", ["concentration_targets/cn_sample_1.tiff", "USAF1951_template/res_source.png"])
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
    result = Path(fetch._fetch(filename))
    assert fetch._valid_file(result, filename)
