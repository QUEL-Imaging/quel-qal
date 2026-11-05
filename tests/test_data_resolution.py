"""Offline source, checkout, pointer, and cache contracts."""
import hashlib
import json
from pathlib import Path
import subprocess

import pytest

from qal.data import _fetchers as fetch
from qal.data import _source as source


@pytest.fixture(autouse=True)
def clean_environment(monkeypatch):
    for name in ("QAL_DATA_REF", "QAL_DATA_REPOSITORY", "QAL_DATA_ROOT", "QAL_DATA_SOURCE"):
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def checkout(tmp_path, monkeypatch):
    root = tmp_path / "repo"
    data = root / "qal/data"
    data.mkdir(parents=True)
    (data / "_registry.py").write_text("registry = {}\n")
    def git(*args):
        return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()
    git("init", "-b", "dev")
    git("add", ".")
    git("-c", "user.name=Test", "-c", "user.email=test@example.com", "commit", "-m", "test")
    git("remote", "add", "origin", "https://github.com/testing/another-repo.git")
    monkeypatch.setattr(source, "DATA_DIR", data)
    return data, git


@pytest.fixture
def dataset(tmp_path, monkeypatch):
    name = "test/data file.tiff"
    content = b"hydrated image bytes"
    monkeypatch.setattr(fetch, "registry", {name: hashlib.sha256(content).hexdigest()})
    monkeypatch.setattr(fetch.pooch, "os_cache", lambda _: tmp_path / "cache")
    return name, content


@pytest.mark.parametrize("branch", ["feature/test-data", "dev", "main"])
def test_checkout_ref_outside_cwd(checkout, monkeypatch, tmp_path, branch):
    data, git = checkout
    if branch != "dev":
        git("switch", "-c", branch)
    monkeypatch.chdir(tmp_path)
    resolved = source.resolve_source()
    assert resolved == source.Source(branch, "testing/another-repo", data)
    assert f"/{branch.replace('/', '%2F')}/qal/data/" in resolved.url("image with space.tiff")
    assert "image%20with%20space.tiff" in resolved.url("image with space.tiff")


def test_detached_head_uses_commit(checkout):
    _, git = checkout
    commit = git("rev-parse", "HEAD")
    git("checkout", "--detach")
    assert source.resolve_source().ref == commit


def test_installed_without_git_and_unrelated_parent(checkout, monkeypatch):
    data, _ = checkout
    installed = data.parent.parent / "other/site-packages/qal/data"
    installed.mkdir(parents=True)
    monkeypatch.setattr(source, "DATA_DIR", installed)
    monkeypatch.setattr(source, "_git", lambda *args: pytest.fail("must not invoke Git"))
    assert source.resolve_source().ref == "main"
    (installed / "_data_source.json").write_text(json.dumps({"ref": "dev", "repository": "testing/repo"}))
    assert source.resolve_source().ref == "dev"
    monkeypatch.setenv("QAL_DATA_REF", "release/tag")
    assert source.resolve_source().ref == "release/tag"


def test_untracked_package_under_repo_is_not_checkout(checkout, tmp_path):
    data, git = checkout
    git("rm", "--cached", "qal/data/_registry.py")
    assert source.checkout_source() is None


def test_overrides_change_subsequent_requests(checkout, monkeypatch):
    assert source.resolve_source().checkout
    for ref in ("feature/new-data", "v1", "a" * 40):
        monkeypatch.setenv("QAL_DATA_REF", ref)
        assert source.resolve_source().ref == ref
        assert source.resolve_source().checkout is None
    monkeypatch.setenv("QAL_DATA_REF", "dev")
    assert source.resolve_source().checkout
    monkeypatch.setenv("QAL_DATA_REPOSITORY", "elsewhere/repo")
    assert source.resolve_source().checkout is None


def write_data(root, name, content):
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return path


def test_hydrated_checkout_no_http(checkout, dataset, monkeypatch):
    data, _ = checkout
    name, content = dataset
    path = write_data(data, name, content)
    monkeypatch.setattr(fetch.pooch, "retrieve", lambda **kw: pytest.fail("HTTP/cache accessed"))
    assert fetch._fetch(name) == str(path)


def test_explicit_root_first(checkout, dataset, monkeypatch, tmp_path):
    name, content = dataset
    root = tmp_path / "explicit"
    path = write_data(root, name, content)
    monkeypatch.setenv("QAL_DATA_ROOT", str(root))
    monkeypatch.setenv("QAL_DATA_REF", "other-ref")
    assert fetch._fetch(name) == str(path)


@pytest.mark.parametrize("local", [None, b"broken", b"version https://git-lfs.github.com/spec/v1\noid sha256:abc\nsize 10\n"])
def test_invalid_local_downloads_selected_ref(checkout, dataset, monkeypatch, tmp_path, local):
    data, git = checkout
    git("switch", "-c", "feature/test-data")
    name, content = dataset
    if local:
        write_data(data, name, local)
    def retrieve(**kw):
        assert "/feature%2Ftest-data/" in kw["url"]
        assert kw["known_hash"] == "sha256:" + hashlib.sha256(content).hexdigest()
        return str(write_data(Path(kw["path"]), kw["fname"], content))
    monkeypatch.setattr(fetch.pooch, "retrieve", retrieve)
    assert Path(fetch._fetch(name)).read_bytes() == content


def test_override_skips_wrong_checkout_and_cache_isolation(checkout, dataset, monkeypatch):
    data, _ = checkout
    name, content = dataset
    write_data(data, name, content)
    first = fetch.get_cache_dir()
    monkeypatch.setenv("QAL_DATA_REF", "feature/new")
    assert fetch.get_cache_dir() != first
    def missing(**kw):
        raise FileNotFoundError("404")
    monkeypatch.setattr(fetch.pooch, "retrieve", missing)
    with pytest.raises(RuntimeError, match="test/data file.tiff.*feature/new"):
        fetch._fetch(name)
    second = fetch.get_cache_dir()
    monkeypatch.setenv("QAL_DATA_REPOSITORY", "other/repo")
    assert fetch.get_cache_dir() != second


def test_real_pooch_cache_and_remote_only(checkout, dataset, monkeypatch):
    data, _ = checkout
    name, content = dataset
    downloads = []
    def downloader(src, filename):
        def download(url, output, pooch):
            downloads.append(url)
            Path(output).write_bytes(content)
        return download
    monkeypatch.setattr(fetch, "_downloader", downloader)
    cached = fetch._fetch(name)
    assert fetch._fetch(name) == cached
    assert len(downloads) == 1
    write_data(data, name, content)
    monkeypatch.setenv("QAL_DATA_SOURCE", "remote")
    assert Path(fetch._fetch(name)).read_bytes() == content
    assert len(downloads) == 2
    assert Path(fetch._fetch(name)).read_bytes() == content
    assert len(downloads) == 3


def test_remote_pointer_rejected_before_cache(checkout, dataset, monkeypatch):
    name, _ = dataset
    def http(**kwargs):
        def download(url, output, pooch):
            Path(output).write_bytes(b"version https://git-lfs.github.com/spec/v1\n")
        return download
    monkeypatch.setattr(fetch.pooch, "HTTPDownloader", http)
    with pytest.raises(RuntimeError, match="unresolved Git LFS pointer"):
        fetch._fetch(name)
    assert not (Path(fetch.get_cache_dir()) / name).exists()


def test_non_lfs_blob_fallback_same_ref(monkeypatch, tmp_path):
    from requests import HTTPError, Response
    urls = []
    def http(**kwargs):
        def download(url, output, pooch):
            urls.append(url)
            if len(urls) == 1:
                response = Response()
                response.status_code = 404
                raise HTTPError(response=response)
            Path(output).write_bytes(b"png bytes")
        return download
    monkeypatch.setattr(fetch.pooch, "HTTPDownloader", http)
    src = source.Source("feature/test", "testing/repo")
    fetch._downloader(src, "some.png")(src.url("some.png"), tmp_path / "out", None)
    assert urls == [src.url("some.png"), src.raw_url("some.png")]


def test_directory_loaders_return_local_directory(checkout, dataset):
    data, _ = checkout
    name, content = dataset
    path = write_data(data, name, content)
    assert fetch._fetch_directory("test") == str(path.parent)


def test_unknown_filename(dataset):
    with pytest.raises(ValueError, match="not in the registry"):
        fetch._fetch("missing")


def test_git_worktree_and_ssh_origin(checkout, tmp_path, monkeypatch):
    _, git = checkout
    git("remote", "set-url", "origin", "git@github.com:testing/worktree-repo.git")
    worktree = tmp_path / "worktree"
    git("worktree", "add", "-b", "feature/worktree", str(worktree))
    monkeypatch.setattr(source, "DATA_DIR", worktree / "qal/data")
    assert source.resolve_source() == source.Source(
        "feature/worktree", "testing/worktree-repo", worktree / "qal/data")


def test_local_pointer_read_is_bounded(tmp_path):
    path = tmp_path / "pointer"
    path.write_bytes(b"version https://git-lfs.github.com/spec/v1\n" + b"x" * 5000)
    assert fetch._is_pointer(path)


def test_partial_download_never_becomes_cache_entry(checkout, dataset, monkeypatch):
    name, _ = dataset
    def downloader(src, filename):
        def download(url, output, pooch):
            Path(output).write_bytes(b"partial bytes")
            raise OSError("connection interrupted")
        return download
    monkeypatch.setattr(fetch, "_downloader", downloader)
    with pytest.raises(RuntimeError, match="connection interrupted"):
        fetch._fetch(name)
    assert not (Path(fetch.get_cache_dir()) / name).exists()


def test_mixed_directory_sources_and_legacy_lung_cache(checkout, dataset, monkeypatch):
    data, _ = checkout
    name, content = dataset
    other = "test/second.tiff"
    lung = "lung_reference/Inclusions_info.pkl"
    monkeypatch.setattr(fetch, "registry", dict.fromkeys([name, other, lung], hashlib.sha256(content).hexdigest()))
    write_data(data, name, content)
    write_data(data, lung, content)
    def downloader(src, filename):
        def download(url, output, pooch):
            Path(output).write_bytes(content)
        return download
    monkeypatch.setattr(fetch, "_downloader", downloader)
    result = Path(fetch._fetch_directory("test"))
    assert (result / Path(name).name).read_bytes() == content
    assert (result / Path(other).name).read_bytes() == content
    fetch.load_lung_info()
    assert (Path(fetch.get_cache_dir()) / lung).read_bytes() == content
