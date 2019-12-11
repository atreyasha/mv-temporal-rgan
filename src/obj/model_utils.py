#!/usr/bin/env python3
# -*- coding: utf-8 -*-

def save_model(model,direct):
    model.generator.save_weights("./pickles/"+direct+"/gen_weights.h5")
    model.discriminator.save_weights("./pickles/"+direct+"/dis_weights.h5")
    with open("./pickles/"+direct+"/dis_opt_weights.pickle","wb") as f:
        pickle.dump(model.discriminator.optimizer.get_weights(),f)
    with open("./pickles/"+direct+"/comb_opt_weights.pickle","wb") as f:
        pickle.dump(model.combined.optimizer.get_weights(),f)

