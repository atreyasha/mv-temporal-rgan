#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from obj.RGAN import RGAN

# compile RGAN and load images
# provide details here to train function for logging
# TODO: add data handling capability here, get data andconvert to string format
# name model as currentDateTime_modeTyle_dataType
train_images = np.load("./data/lfw.npy")
# run model and check outout
# TODO: require RGAN object to return values for logging
# TODO: add saving ability by saving models and main hyperparameters
# TODO: experiment exact model reconstruction in regards to sharing weights and memory
# save weights and parameters and load them back into memory
# needs original model to be removed from memory before reloading
test = RGAN()
test.train(train_images,"test")
