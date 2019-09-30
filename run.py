#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import glob
from tqdm import tqdm
import numpy as np
import matplotlib.image as mpimg
from numpy import resize
from RGAN import RGAN

# compile RGAN and load images
train_images = [resize(mpimg.imread(file),(28,28)) for file in tqdm(glob.glob("./data/lfwcrop_grey/faces/*")[:100])]
train_images = np.asarray(train_images,dtype="float32")
train_images /= 255
train_images = resize(train_images,(train_images.shape[0],train_images.shape[1]**2,1))
# run model and check outout
test = RGAN()
test.train(train_images,"test")

# add hp5y pipeline to speed up data transfer
# save better quality images, make more random noise to show transitions
# save proper images in svg and download better image viewer
# make more complex generator
# make deeper and more intracacies to layers
# possibility to add convolutions at early stages and treat meta states as images
# most important is that final stage converges to actual time series
