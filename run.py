#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import glob
from tqdm import tqdm
import numpy as np
import matplotlib.image as mpimg
from numpy import resize
from RGAN import RGAN

# compile RGAN and load images
# train_images = [resize(mpimg.imread(file),(28,28)) for file in tqdm(glob.glob("./data/lfwcrop_grey/faces/*"))]
# train_images = np.asarray(train_images,dtype="float32")
# train_images /= 255
# train_images = resize(train_images,(train_images.shape[0],train_images.shape[1]**2,1))
train_images = np.load("./data/lfw.npy")
# run model and check outout
test = RGAN()
test.train(train_images,"test")
