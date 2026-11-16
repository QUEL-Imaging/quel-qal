import os
import cv2
import hashlib
import shutil
import pooch
import tempfile
import weakref
from contextlib import nullcontext
from pathlib import Path, PurePosixPath
from ._source import Source, resolve_source
from requests import HTTPError
from qal.util._image import load_grayscale_image

# Track only live sample arrays, so exports can copy the original source file
# without retaining image buffers or guessing a sample from its pixel values.
_SAMPLE_ARRAYS = {}
_SAMPLE_DIRECTORIES = {}


def _sample_source_paths(value):
    entry = _SAMPLE_ARRAYS.get(id(value))
    if entry is not None and entry[0]() is value:
        return (entry[1],)
    if isinstance(value, (str, Path)):
        return _SAMPLE_DIRECTORIES.get(Path(value).expanduser().resolve(), ())
    return ()

try:
    from ._registry import registry
except ImportError:
    # If your registry is simply in the same folder, do a local import:
    from _registry import registry

try:
    from pooch import file_hash
except ModuleNotFoundError:
    def file_hash(fname, alg="sha256"):
        """
        Calculate the hash of a given file.
        Useful for checking if a file has changed or been corrupted.

        Parameters
        ----------
        fname : str
            The name of the file.
        alg : str
            The type of the hashing algorithm.

        Returns
        -------
        hash : str
            The hash of the file.
        """
        if alg not in hashlib.algorithms_available:
            raise ValueError(f"Algorithm '{alg}' not available in hashlib")
        hasher = hashlib.new(alg)
        with open(fname, "rb") as fin:
            for chunk in iter(lambda: fin.read(65536), b""):
                hasher.update(chunk)
        return hasher.hexdigest()

def _create_image_fetcher(branch=None, *, source=None):
    """Create a reference- and repository-isolated Pooch cache."""
    source = source or resolve_source()
    if branch is not None:
        source = Source(branch, source.repository)
    identity = hashlib.sha256(
        f"{source.repository}\0{source.ref}".encode()
    ).hexdigest()
    return pooch.create(
        path=Path(pooch.os_cache("quel-qal-data")) / identity,
        base_url=source.url(""), registry=registry,
        urls={name: source.url(name) for name in registry},
    )


def _is_pointer(path):
    with open(path, "rb") as stream:
        return stream.read(1024).startswith(b"version https://git-lfs.github.com/spec/v1")


def _valid_file(path, filename):
    try:
        return (path.is_file() and not _is_pointer(path)
                and file_hash(path) == registry[filename])
    except OSError:
        return False


def _downloader(source, filename):
    def download(url, output_file, pooch_object):
        http = pooch.HTTPDownloader(progressbar=True)
        try:
            http(url, output_file, pooch_object)
        except HTTPError as exc:
            if exc.response is None or exc.response.status_code != 404:
                raise
            # Non-LFS blobs are served by raw instead of media. This is the
            # same repository/ref/path, never a fallback to another branch.
            http(source.raw_url(filename), output_file, pooch_object)
        if _is_pointer(output_file):
            raise ValueError("The server returned an unresolved Git LFS pointer")
    return download


def _fetch(filename):
    """Resolve a registered dataset locally, then from its selected remote ref."""
    if filename not in registry:
        raise ValueError(f"'{filename}' is not in the registry.")
    mode = os.environ.get("QAL_DATA_SOURCE", "auto")
    if mode not in ("auto", "remote"):
        raise ValueError("QAL_DATA_SOURCE must be 'auto' or 'remote'.")
    source = resolve_source()
    if mode != "remote":
        roots = []
        if os.environ.get("QAL_DATA_ROOT"):
            roots.append(Path(os.environ["QAL_DATA_ROOT"]).expanduser().resolve())
        if source.checkout:
            roots.append(source.checkout)
        for root in roots:
            path = root / filename
            if _valid_file(path, filename):
                return str(path)

    fetcher = _create_image_fetcher(source=source)
    try:
        # A fresh temporary cache forces an HTTP request even when a normal cache
        # entry exists. Pooch validates hashes before atomically moving files.
        with tempfile.TemporaryDirectory(prefix="qal-remote-") if mode == "remote" else nullcontext(None) as scratch:
            path = pooch.retrieve(
                url=source.url(filename), known_hash=f"sha256:{registry[filename]}",
                path=scratch or fetcher.path, fname=filename,
                downloader=_downloader(source, filename),
            )
            if _is_pointer(path):
                raise ValueError("The server returned an unresolved Git LFS pointer")
            if scratch:
                destination = fetcher.path / filename
                destination.parent.mkdir(parents=True, exist_ok=True)
                # Publish only the validated result; replace atomically even if
                # the temporary directory is on a different filesystem.
                with tempfile.NamedTemporaryFile(dir=destination.parent, delete=False) as temp:
                    staging = Path(temp.name)
                try:
                    shutil.copyfile(path, staging)
                    os.replace(staging, destination)
                finally:
                    staging.unlink(missing_ok=True)
                path = str(destination)
            return path
    except Exception as exc:
        raise RuntimeError(
            f"Cannot fetch '{filename}' from {source.repository} at ref '{source.ref}' "
            f"({source.url(filename)}). Check that the data is pushed and Git LFS "
            f"objects are uploaded. {exc}"
        ) from exc


def get_cache_dir():
    """Return the cache directory for the currently selected repository and ref."""
    return str(_create_image_fetcher().path)


def download_all():
    """Resolve all registered datasets for offline use."""
    for filename in registry:
        print(f"Fetching: {filename}")
        _fetch(filename)


def clear_cache():
    """Remove cached data for the currently selected repository and ref."""
    cache_dir = get_cache_dir()
    if os.path.exists(cache_dir):
        shutil.rmtree(cache_dir)


def _cache_local(path, filename):
    """Keep legacy directory consumers working when resolution returns local data."""
    destination = Path(get_cache_dir()) / filename
    if Path(path) != destination:
        destination.parent.mkdir(parents=True, exist_ok=True)
        # Copy only for consumers that specifically require a cache directory.
        # Do not hardlink: consumers can modify files in the writable cache.
        with tempfile.NamedTemporaryFile(dir=destination.parent, delete=False) as temp:
            staging = Path(temp.name)
        try:
            shutil.copyfile(path, staging)
            os.replace(staging, destination)
        finally:
            staging.unlink(missing_ok=True)
    return str(destination)


def _fetch_directory(directory):
    names = [name for name in registry if str(PurePosixPath(name).parent) == directory]
    paths = [_fetch(name) for name in names]
    parents = {str(Path(path).parent) for path in paths}
    if len(parents) == 1:
        result = parents.pop()
        _SAMPLE_DIRECTORIES[Path(result).resolve()] = tuple(Path(p) for p in paths)
        return result
    for name, path in zip(names, paths):
        _cache_local(path, name)
    result = Path(get_cache_dir()) / directory
    _SAMPLE_DIRECTORIES[result.resolve()] = tuple(result / Path(n).name for n in names)
    return str(result)


def _load(filename):
    """
    Load an image from the data cache by name.

    Parameters
    ----------
    filename : str
        Filename within the registry to load.

    Returns
    -------
    img : ndarray
        Image read from local cache.
    """
    path = _fetch(filename)
    image = load_grayscale_image(path)
    key = id(image)
    _SAMPLE_ARRAYS[key] = (
        weakref.ref(image, lambda ref: _SAMPLE_ARRAYS.pop(key, None)),
        Path(path),
    )
    return image


# --- Concentration samples
def cn_sample_1():
    """
    Return the cn_sample_1.tiff path as an ndarray.
    """
    return _load("concentration_targets/cn_sample_1.tiff")

def cn_sample_2():
    """
    Return the cn_sample_2.tiff path as an ndarray.
    """
    return _load("concentration_targets/cn_sample_2.tiff")

def cn_sample_3():
    """
    Return the cn_sample_3.tiff path as an ndarray.
    """
    return _load("concentration_targets/cn_sample_3.tiff")

def cn_sample_4():
    """
    Return the cn_sample_4.tiff path as an ndarray.
    """
    return _load("concentration_targets/cn_sample_4.tiff")

# --- Radiometric emitter targets
def ret_sample_1_lf():
    """Return sample 1's light frame as a grayscale ndarray."""
    return _load("RET_images/ret_sample_1_lf.tiff")


def ret_sample_1_df():
    """Return the matching dark frame for sample 1 as a grayscale ndarray."""
    return _load("RET_images/ret_sample_1_df.tiff")


def ret_sample_2_lf():
    """Return sample 2's light frame as a grayscale ndarray."""
    return _load("RET_images/ret_sample_2_lf.tiff")


def ret_sample_2_df():
    """Return the matching dark frame for sample 2 as a grayscale ndarray."""
    return _load("RET_images/ret_sample_2_df.tiff")


# --- Depth resolution targets
def dr_sample1():
    """
    Return dr_sample1.tiff as an ndarray.
    """
    return _load("depth_resolution_targets/dr_sample1.tiff")

def dr_sample2():
    """
    Return dr_sample2.tiff as an ndarray.
    """
    return _load("depth_resolution_targets/dr_sample2.tiff")

def dr_sample3():
    """
    Return dr_sample3.tiff as an ndarray.
    """
    return _load("depth_resolution_targets/dr_sample3.tiff")

# --- Depth targets
def depth_sample_1():
    """
    Return depth_sample_1.tiff as an ndarray.
    """
    return _load("depth_targets/depth_sample_1.tiff")

def depth_sample_2():
    """
    Return depth_sample_2.tiff as an ndarray.
    """
    return _load("depth_targets/depth_sample_2.tiff")

# --- Resolution targets
def res_sample_1():
    """
    Return res_sample_1.tiff as an ndarray.
    """
    return _load("resolution_targets/res_sample_1.tiff")

def resolution_target_cropped():
    """
    Return resolution_target_cropped.tiff as an ndarray.
    """
    return _load("resolution_targets/resolution_target_cropped.tiff")

def resolution_template():
    """
    Return res_source.png as an ndarray.
    """
    path = _fetch("USAF1951_template/res_source.png")
    return cv2.imread(path, 0)

# --- RUD targets
def rud_example_1():
    """Return the resolved directory of RUD example 1."""
    return _fetch_directory("rud_targets/example_1")

def rud_example_2():
    """Return the resolved directory of RUD example 2."""
    return _fetch_directory("rud_targets/example_2")

def rud_example_3():
    """Return the resolved directory of RUD example 3."""
    return _fetch_directory("rud_targets/example_3")

def rud_example_4():
    """Return the resolved directory of RUD example 4."""
    return _fetch_directory("rud_targets/example_4")

# --- RUD targets, example_1
def rud_image1_example_1():
    """
    Return rud_image1.tiff from example_1 as an ndarray.
    """
    return _load("rud_targets/example_1/rud_image1.tiff")

def rud_image2_example_1():
    """
    Return rud_image2.tiff from example_1 as an ndarray.
    """
    return _load("rud_targets/example_1/rud_image2.tiff")

def rud_image3_example_1():
    """
    Return rud_image3.tiff from example_1 as an ndarray.
    """
    return _load("rud_targets/example_1/rud_image3.tiff")

def rud_image4_example_1():
    """
    Return rud_image4.tiff from example_1 as an ndarray.
    """
    return _load("rud_targets/example_1/rud_image4.tiff")

# --- RUD targets, example_2
def rud_image1_example_2():
    """
    Return rud_image1.tiff from example_2 as an ndarray.
    """
    return _load("rud_targets/example_2/rud_image1.tiff")

def rud_image2_example_2():
    """
    Return rud_image2.tiff from example_2 as an ndarray.
    """
    return _load("rud_targets/example_2/rud_image2.tiff")

def rud_image3_example_2():
    """
    Return rud_image3.tiff from example_2 as an ndarray.
    """
    return _load("rud_targets/example_2/rud_image3.tiff")

def rud_image4_example_2():
    """
    Return rud_image4.tiff from example_2 as an ndarray.
    """
    return _load("rud_targets/example_2/rud_image4.tiff")

def rud_image5_example_2():
    """
    Return rud_image5.tiff from example_2 as an ndarray.
    """
    return _load("rud_targets/example_2/rud_image5.tiff")

# --- RUD targets, example_3
def rud_center_fov_example_3():
    """
    Return rud_center_fov.tiff from example_3 as an ndarray.
    """
    return _load("rud_targets/example_3/rud_center_fov.tiff")

# --- RUD targets, example_4
def rud_image_example_4():
    """
    Return rud_image.TIF from example_4 as an ndarray.
    """
    return _load("rud_targets/example_4/rud_image.TIF")

# --- Lung test
def lung_test_image():
    """
    Return lung_test_image.tiff as an ndarray.
    """
    return _load("lung_phantom/lung_test_image/lung_test_image.tiff")

def lung_reference_inclusions():
    """
    Return Inclusions image.tiff as an ndarray.
    """
    return _load("lung_phantom/lung_reference_source/Inclusions image.tiff")

def lung_reference_body():
    """
    Return Phantom image.tiff as an ndarray.
    """
    return _load("lung_phantom/lung_reference_source/Phantom image.tiff")

def lung_reference():
    """
    Return Inclusions image.tiff and Phantom image.tiff as ndarrays.
    """
    im_phantom = lung_reference_body()
    im_inclusions = lung_reference_inclusions()
    return im_phantom, im_inclusions

def lung_reference_image():
    """
    Return Lung_reference_image.tiff as an ndarray.
    """
    return _load("lung_phantom/lung_reference/Lung_reference_image.tiff")

def lung_reference_mask():
    """
    Return Lung_reference_mask.png as an ndarray.
    """
    return _load("lung_phantom/lung_reference/Lung_reference_mask.png")

def load_lung_info():
    """
    Return the resolved path to the lung phantom inclusions pickle file
    """
    return _fetch("lung_phantom/lung_reference/Inclusions_info.pkl")

#
# End of _fetchers.py
#
