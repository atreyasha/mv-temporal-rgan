#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import h5py

def save_model(model,direct):
    gen_weights = model.generator.get_weights()
    dis_weights = model.discriminator.get_weights()
    comb_opt_weights = model.combined.get_weights()
    dis_opt_weights = model.discriminator.get_weights()
    with h5py.File("./pickles/"+direct+"/model.h5") as h:
        h.create_datasest("generator",gen_weights,compression="gzip")
        h.create_dataset("discriminator",dis_weights,compression="gzip")
        h.create_dataset("comb_opt",comb_opt_weights,compression="gzip")
        h.create_dataset("dis_opt",dis_opt_weights,compression="gzip")
