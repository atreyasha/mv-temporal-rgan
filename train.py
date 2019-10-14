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
from keras.models import load_model
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

def continueTrain(direct,arguments):
    if "./pickles/" in direct:
        directLong = direct
        direct = re.sub("./pickles/","",direct)
    else:
        directLong = "./pickles/"+direct
    if not os.path.isdir(directLong):
        sys.exit(directLong +" does not exist")
    # read init.csv and return construction parameters
    meta = pd.read_csv(directLong+"/init.csv")
    toParse = set(meta.columns)-set(arguments.keys())
    # add arguments given as variables in memory
    globals().update(arguments)
    # read remaining variables which must be parsed
    rem = {el:meta.iloc[0][el] for el in toParse}
    # add arguments parsed into memory
    globals().update(rem)
    train_images = loadData(data)
    log_dir = re.sub("RGAN_","RGAN_"+getCurrentTime()+"_",directLong)
    log_dir_pass = re.sub("./pickles/","",log_dir)
    os.makedirs(log_dir)
    os.makedirs(log_dir+"/img")
    rgan = RGAN(latent_dim,im_dim,epochs,batch_size,learning_rate,
                g_factor,droprate,momentum,alpha)
    # load models into memory
    gen = load_model(directLong+"/gen_model.h5")
    dis = load_model(directLong+"/dis_model.h5")
    comb = load_model(directLong+"/comb_model.h5")
    dis_optimizer_weights = dis.optimizer.get_weights()
    comb_optimizer_weights = comb.optimizer.get_weights()
    # load model and optimizer weights into main class
    rgan.generator.set_weights(gen.get_weights())
    rgan.discriminator.set_weights(dis.get_weights())
    rgan.discriminator.optimizer.set_weights([None for el in dis_optimizer_weights])
    rgan.combined.set_weights(comb.get_weights())
    rgan.combined.optimizer.set_weights([None for el in comb_optimizer_weights])
    # clear memory
    del gen, dis, comb
    # rgan.train(train_images,log_dir_pass)

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
    parser.add_argument("--log-dir", required="--continue-train" in sys.argv,
                        help="log directory within ./pickles/ whose model should be further trained, only required when --continue-train option is specified")
    args = parser.parse_args()
    assert args.subtype in ["faces","mnist","fashion"]
    if args.continue_train:
        # parse specified arguments as kwargs to continueTrain
        arguments = [el for el in sys.argv[1:] if el != "--continue-train"]
        for i in range(len(arguments)):
            if arguments[i] == "--log-dir":
                del arguments[i:i+2]
                break
        arguments = [re.sub("-","_",re.sub("--","",arguments[i])) if i%2 == 0 else
                     arguments[i] for i in range(len(arguments))]
        arguments = dict(zip(arguments[::2], arguments[1::2]))
        for key in arguments.keys():
            check = str(type(getattr(args,key)))
            if "int" in check:
                arguments[key] = int(arguments[key])
            elif "str" in check:
                arguments[key] = str(arguments[key])
            elif "float" in check:
                arguments[key] = float(arguments[key])
        continueTrain(args.log_dir,arguments)
    else:
        singularTrain(args.subtype,args.latent_dim,args.epochs,args.batch_size,
                      args.learning_rate,args.g_factor,args.droprate,args.momentum,args.alpha)
