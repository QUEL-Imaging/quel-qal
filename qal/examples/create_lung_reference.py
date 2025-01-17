"""
This script creates the reference image and mask for the lung phantom.
"""

from qal.data import lung_reference
from qal import LungPhantom


def main():

    im_phantom, im_inclusions = lung_reference()

    lung_phantom = LungPhantom()
    lung_phantom.create_reference_mask(im_phantom, im_inclusions)


if __name__ == "__main__":
    main()
