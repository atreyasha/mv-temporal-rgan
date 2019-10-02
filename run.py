#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from obj.RGAN import RGAN

# compile RGAN and load images
train_images = np.load("./data/lfw_28.npy")
# run model and check outout
test = RGAN()
test.train(train_images,"test")
