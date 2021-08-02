#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# load dependencies
import glob
import argparse
import numpy as np
import matplotlib.image as mpimg
from obj.arg_formatter import arg_metav_formatter
from scipy.ndimage import zoom
from tqdm import tqdm


def makeBin(out, size_factor):
    """
    Resample and convert lfw faces data into numpy array

    Args:
        out (str): name of output numpy array
        size_factor (float): resampling factor
    """
    train_images = [
        zoom(mpimg.imread(file), size_factor, mode="mirror")
        for file in tqdm(glob.glob("./data/lfwcrop_grey/faces/*"))
    ]
    train_images = np.asarray(train_images, dtype="float32")
    train_images = (train_images - 127.5) / 127.5
    np.save("./data/lfwcrop_grey/" + out, train_images)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=arg_metav_formatter)
    parser.add_argument(
        "--size-factor",
        type=float,
        default=0.4375,
        help="factor by which to upsample or downsample images")
    parser.add_argument("--out",
                        type=str,
                        default="lfw.npy",
                        help="output file name")
    args = parser.parse_args()
    makeBin(args.out, args.size_factor)
