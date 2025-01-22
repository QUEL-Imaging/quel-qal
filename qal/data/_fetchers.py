import os
import cv2
import hashlib
import shutil
import pooch
from skimage import io

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

def _create_image_fetcher(branch="main"):
    """
    Create a pooch fetcher object that handles downloading/caching data files.

    Parameters
    ----------
    branch : str
        The branch to fetch files from.

    Returns
    -------
    pooch.Pooch
        Configured pooch fetcher.
    """
    base_url = f"https://github.com/QUEL-Imaging/quel-qal/raw/{branch}/qal/data/"
    cache_dir = pooch.os_cache("quel-qal-data")

    # Create the Pooch fetcher
    return pooch.create(
        path=cache_dir,
        base_url=base_url,
        registry=registry,
    )

# Initialize the fetcher
_image_fetcher = _create_image_fetcher(branch='main')

def _fetch(filename):
    """
    Fetch a file by name from the registry, downloading if necessary.

    Parameters
    ----------
    filename : str
        The name of the file to fetch.

    Returns
    -------
    str
        The local, absolute path to the file.

    Raises
    ------
    ValueError:
        If the filename is not in the registry.
    """
    if filename not in registry:
        raise ValueError(f"'{filename}' is not in the registry.")

    # Use pooch.retrieve to download the file and verify the hash
    return pooch.retrieve(
        url=_image_fetcher.get_url(filename),
        known_hash=f"sha256:{registry[filename]}",
        path=_image_fetcher.path,
        fname=filename,
        progressbar=True,  # Show a progress bar during download
    )

def get_cache_dir():
    """
    Get the cache directory used by pooch.

    Returns
    -------
    str
        Path to the cache directory.
    """
    return str(_image_fetcher.path)


def download_all():
    """
    Download all data files for offline use.

    Raises
    ------
    ModuleNotFoundError:
        If `_image_fetcher` is not initialized.
    """
    if _image_fetcher is None:
        raise ModuleNotFoundError("Pooch fetcher is not initialized.")

    for filename in registry:
        print(f"Fetching: {filename}")
        _fetch(filename)

def clear_cache():
    """
    Remove all cached data from the pooch cache directory.
    """
    cache_dir = _image_fetcher.path
    if os.path.exists(cache_dir):
        print(f"Deleting cache at {cache_dir}...")
        shutil.rmtree(cache_dir)
        print("Cache cleared.")
    else:
        print("No cache directory found to delete.")


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
    return io.imread(path)


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
    """
    Load example 1 RUD images and return the local path to the image directory.
    """
    rud_image1_example_1()
    rud_image2_example_1()
    rud_image3_example_1()
    rud_image4_example_1()
    return os.path.join(get_cache_dir(), "rud_targets", "example_1")

def rud_example_2():
    """
    Load example 2 RUD images and return the local path to the image directory.
    """
    rud_image1_example_2()
    rud_image2_example_2()
    rud_image3_example_2()
    rud_image4_example_2()
    rud_image5_example_2()
    return os.path.join(get_cache_dir(), "rud_targets", "example_2")

def rud_example_3():
    """
    Load example 3 RUD image and return the local path to the image directory.
    """
    rud_center_fov_example_3()
    return os.path.join(get_cache_dir(), "rud_targets", "example_3")

def rud_example_4():
    """
    Load example 4 RUD image and return the local path to the image directory.
    """
    rud_image_example_4()
    return os.path.join(get_cache_dir(), "rud_targets", "example_4")

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
    return _load("lung_test_image/lung_test_image.tiff")

def lung_reference_inclusions():
    """
    Return Inclusions image.tiff as an ndarray.
    """
    return _load("lung_reference_source/Inclusions image.tiff")

def lung_reference_body():
    """
    Return Phantom image.tiff as an ndarray.
    """
    return _load("lung_reference_source/Phantom image.tiff")

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
    return _load("lung_reference/Lung_reference_image.tiff")

def lung_reference_mask():
    """
    Return Lung_reference_mask.png as an ndarray.
    """
    return _load("lung_reference/Lung_reference_mask.png")

def load_lung_info():
    """
    Download the pickle file containing information on the lung phantom inclusions
    """
    _fetch("lung_reference/Inclusions_info.pkl")

#
# End of _fetchers.py
#
