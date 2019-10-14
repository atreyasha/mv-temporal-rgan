#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# ignore tensorflow/numpy warnings
import warnings
warnings.filterwarnings('ignore')
# import dependencies
import os
import sys
import re
import pandas as pd
import argparse
import datetime
import numpy as np
from obj.RGAN import RGAN
from keras.datasets import mnist, fashion_mnist

################################
# define key functions
################################

def getCurrentTime():
    return datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")

def loadData(subtype):
    if subtype == "faces":
        return np.load("./data/lfw.npy")
    elif subtype == "mnist":
        (train_images,_), (_,_) = mnist.load_data()
        return np.resize(train_images, (train_images.shape[0],
                                        train_images.shape[1]**2,1))/255
    elif subtype == "fashion":
        (train_images,_), (_,_) = fashion_mnist.load_data()
        return np.resize(train_images, (train_images.shape[0],
                                        train_images.shape[1]**2,1))/255

def singularTrain(subtype,latent_dim,epochs,batch_size,learning_rate,
                  g_factor,droprate,momentum,alpha,model="RGAN"):
    train_images = loadData(subtype)
    im_dim = int(np.sqrt(train_images.shape[1]))
    log_dir = getCurrentTime()+"_"+model+"_"+subtype
    os.makedirs("./pickles/"+log_dir)
    os.makedirs("./pickles/"+log_dir+"/img")
    if model == "RGAN":
        model = RGAN(latent_dim,im_dim,epochs,batch_size,learning_rate,
                     g_factor,droprate,momentum,alpha)
    model.train(train_images,log_dir)

def continueTrain(direct):
    directLong = "./pickles/"+direct
    if not os.path.isdir(directLong):
        sys.exit(directLong +" does not exist")
    # read init.csv and return construction parameters
    meta = pd.read_csv(directLong+"/init.csv")
    subtype = meta.iloc[0]["data"]
    im_dim = meta.iloc[0]["im_dim"]
    latent_dim = meta.iloc[0]["latent_dim"]
    epochs = meta.iloc[0]["epochs"]
    batch_size = meta.iloc[0]["batch_size"]
    learning_rate = meta.iloc[0]["learning_rate"]
    g_factor = meta.iloc[0]["g_factor"]
    droprate = meta.iloc[0]["droprate"]
    momentum = meta.iloc[0]["momentum"]
    alpha = meta.iloc[0]["alpha"]
    train_images = loadData(subtype)
    log_dir = re.sub("RGAN_","RGAN_"+getCurrentTime()+"_",directLong)
    log_dir_pass = re.sub("./pickles/","",log_dir)
    os.makedirs(log_dir)
    os.makedirs(log_dir+"/img")
    rgan = RGAN(latent_dim,im_dim,epochs,batch_size,learning_rate,
                g_factor,droprate,momentum,alpha)
    rgan.generator.load_weights(directLong+"/gen.h5")
    rgan.discriminator.load_weights(directLong+"/dis.h5")
    rgan.combined.load_weights(directLong+"/comb.h5")
    rgan.train(train_images,log_dir_pass)

###############################
# main command call
###############################

if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--subtype", type=str, default="mnist",
                        help="which training data subtype to use; either mnist, fashion or faces")
    parser.add_argument("--latent-dim", type=int, default=100,
                        help="latent dimensionality of GAN generator")
    parser.add_argument("--epochs", type=int, default=100,
                        help="number of training epochs")
    parser.add_argument("--batch-size", type=int, default=256,
                        help="batch size for stochastic gradient descent optimization")
    parser.add_argument("--learning-rate", type=float, default=0.0004,
                        help="learning rate for stochastic gradient descent optimization")
    parser.add_argument("--g-factor", type=float, default=0.25,
                        help="factor by which generator optimizer scales discriminator optimizer")
    parser.add_argument("--droprate", type=float, default=0.25,
                        help="droprate used in GAN discriminator for generalization/robustness")
    parser.add_argument("--momentum", type=float, default=0.8,
                        help="momentum used across GAN batch-normalization")
    parser.add_argument("--alpha", type=float, default=0.2,
                        help="alpha parameter used in discriminator leaky relu")
    parser.add_argument("--continue-train", default=False, action="store_true",
                         help="option to continue training model within log directory; requires --log-dir option to be defined")
    parser.add_argument("--log-dir", required="--continue" in sys.argv,
                        help="log directory whose model should be further trained, only required when --continue-train option is specified")
    args = parser.parse_args()
    assert args.subtype in ["faces","mnist","fashion"]
    if args.continue_train:
        continueTrain(args.log_dir)
    else:
        singularTrain(args.subtype,args.latent_dim,args.epochs,args.batch_size,
                      args.learning_rate,args.g_factor,args.droprate,args.momentum,args.alpha)
