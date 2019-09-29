#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import glob
from tqdm import tqdm
import numpy as np
import matplotlib.image as mpimg
from numpy import resize
from RGAN import RGAN

# compile RGAN and load images
train_images = [mpimg.imread(file) for file in tqdm(glob.glob("./data/lfwcrop_grey/faces/*"))]
train_images = np.asarray(train_images,dtype="float32")
train_images /= 255
# run model and check outout
test = RGAN()
test.train(train_images,"test")

# make more complex generator
# make deeper and more intracacies to layers
# possibility to add convolutions at early stages and treat meta states as images
# most important is that final stage converges to actual time series
