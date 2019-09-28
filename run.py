#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from RGAN import RGAN
import numpy as np
import matplotlib.image as mpimg

# compile RGAN and load images
test = RGAN()
train_images = [mpimg.imread(file) for file in glob.glob("./data/faces/*")]
train_images = np.asarray(train_images,dtype="float32")
train_images /= 255
test.train(train_images,"test")
