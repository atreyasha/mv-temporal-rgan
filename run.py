#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import glob
from tqdm import tqdm
import numpy as np
import matplotlib.image as mpimg
from numpy import resize
from RGAN import RGAN

# compile RGAN and load images
train_images = [resize(mpimg.imread(file),(28,28)) for file in tqdm(glob.glob("./data/lfwcrop_grey/faces/*")[:2000])]
train_images = np.asarray(train_images,dtype="float32")
train_images /= 255
# run model and check outout
test = RGAN()
test.train(train_images,"test")
