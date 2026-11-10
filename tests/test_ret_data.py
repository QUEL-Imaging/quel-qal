"""Exercise RET's public loaders through hash-validated local and download paths."""

from pathlib import Path

import numpy as np
import pytest

from qal import data
from qal.data import _fetchers as fetch
from qal.data import _source as source
from qal.util import load_grayscale_image


@pytest.fixture(params=[
    "ret_sample_1_lf", "ret_sample_1_df", "ret_sample_2_lf", "ret_sample_2_df",
])
def sample(request):
    name = f"RET_images/{request.param}.tiff"
    path = Path(fetch.__file__).parent / name
    if not path.is_file() or fetch._is_pointer(path):
        pytest.skip("requires hydrated RET sample images (git lfs pull)")
    return getattr(data, request.param), name, path


def test_ret_loader_preserves_checkout_pixels(sample, monkeypatch):
    load, name, path = sample
    monkeypatch.setattr(fetch, "resolve_source", lambda: source.Source(
        "bugfix/jupyter-nb", "QUEL-Imaging/quel-qal", path.parents[1]
    ))
    monkeypatch.setenv("QAL_DATA_SOURCE", "auto")
    monkeypatch.delenv("QAL_DATA_ROOT", raising=False)
    monkeypatch.setattr(fetch.pooch, "retrieve", lambda **kw: pytest.fail("unnecessary download"))
    assert fetch._valid_file(path, name)
    image = load()
    assert image.ndim == 2
    assert image.dtype == np.uint16
    np.testing.assert_array_equal(image, load_grayscale_image(path))


def test_ret_download_uses_selected_ref_and_validates_hash(sample, monkeypatch, tmp_path):
    load, name, path = sample
    selected = source.Source("bugfix/jupyter-nb", "QUEL-Imaging/quel-qal", path.parents[1])
    monkeypatch.setattr(fetch, "resolve_source", lambda: selected)
    monkeypatch.setenv("QAL_DATA_SOURCE", "remote")
    monkeypatch.setattr(fetch.pooch, "os_cache", lambda _: tmp_path / "cache")
    requests = []

    def http(**kwargs):
        def download(url, output, pooch):
            requests.append(url)
            # Keep real Pooch hash verification and image decoding in the test.
            Path(output).write_bytes(path.read_bytes())
        return download

    monkeypatch.setattr(fetch.pooch, "HTTPDownloader", http)
    np.testing.assert_array_equal(load(), load_grayscale_image(path))
    assert requests == [
        "https://media.githubusercontent.com/media/QUEL-Imaging/quel-qal/"
        f"bugfix%2Fjupyter-nb/qal/data/{name}"
    ]
    assert fetch._valid_file(Path(fetch.get_cache_dir()) / name, name)

    # A new download with corrupt bytes must fail even with a valid cached copy.
    def corrupt_http(**kwargs):
        return lambda url, output, pooch: Path(output).write_bytes(b"corrupt image")

    monkeypatch.setattr(fetch.pooch, "HTTPDownloader", corrupt_http)
    with pytest.raises(RuntimeError, match="SHA256 hash"):
        load()
    assert fetch._valid_file(Path(fetch.get_cache_dir()) / name, name)
