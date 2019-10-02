#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# load dependencies
import glob
import argparse
import numpy as np
import matplotlib.image as mpimg
from tqdm import tqdm
from numpy import resize

###############################
# define function
###############################

# save raw images into numpy binary
def makeBin(dim,out):
    train_images = [resize(mpimg.imread(file),(dim,dim)) for file in tqdm(glob.glob("./data/lfwcrop_grey/faces/*"))]
    train_images = np.asarray(train_images,dtype="float32")
    train_images /= 255
    train_images = resize(train_images,(train_images.shape[0],train_images.shape[1]**2,1))
    np.save("./data/"+out,train_images)

###############################
# main command call
###############################

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dim", type=int, default=28,
                        help="dimensions to sample lfw-faces <default:28>")
    parser.add_argument("--out", type=str, default="lfw.npy",
                        help="output file name <default:'lfw.npy'>")
    args = parser.parse_args()
    makeBin(args.dim,args.out)
