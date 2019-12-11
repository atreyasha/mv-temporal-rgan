#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pickle

def save_model(model,direct):
    model.generator.save_weights("./pickles/"+direct+"/gen_weights.h5")
    model.discriminator.save_weights("./pickles/"+direct+"/dis_weights.h5")
    with open("./pickles/"+direct+"/dis_opt_weights.pickle","wb") as f:
        pickle.dump(model.discriminator.optimizer.get_weights(),f)
    with open("./pickles/"+direct+"/comb_opt_weights.pickle","wb") as f:
        pickle.dump(model.combined.optimizer.get_weights(),f)

def restore_model(model,train_set,model_name,
                  directLong,log_dir_pass):
    model.generator.load_weights(directLong+"/gen_weights.h5")
    model.discriminator.load_weights(directLong+"/dis_weights.h5")
    model.combined.layers[-2].set_weights(model.generator.get_weights())
    model.combined.layers[-1].set_weights(model.discriminator.get_weights())
    # hold back model information
    hold_epochs = model.epochs
    hold_batch_size = model.batch_size
    model.epochs = 1
    model.batch_size = 1
    # initialize dummy optimizer weights
    if model_name == "RGAN":
        model.train(train_set[:1],log_dir_pass)
    elif model_name ==  "RCGAN":
        model.train((train_set[0][:1],train_set[1][:1]),log_dir_pass)
    # return model information
    model.epochs = hold_epochs
    model.batch_size = hold_batch_size
    with open(directLong+"/dis_opt_weights.pickle", "rb") as f:
        dis_opt_weights = pickle.load(f)
    with open(directLong+"/comb_opt_weights.pickle", "rb") as f:
        comb_opt_weights = pickle.load(f)
    # load previous optimizer weights
    model.discriminator.optimizer.set_weights(dis_opt_weights)
    model.combined.optimizer.set_weights(comb_opt_weights)
    # clear memory
    del dis_opt_weights, comb_opt_weights
    return model
