#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# import dependencies
import os
import argparse
import datetime
import numpy as np
from obj.RGAN_dev import RGAN
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
    rgan = RGAN(latent_dim,im_dim,epochs,batch_size,learning_rate,
                g_factor,droprate,momentum,alpha)
    rgan.train(train_images,log_dir)

###############################
# main command call
###############################

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--subtype", type=str, default="mnist",
                        help="which training data subtype to use; either 'mnist', 'fashion' or 'faces' <default:'mnist'>")
    parser.add_argument("--latent-dim", type=int, default=20,
                        help="latent dimensionality of GAN generator <default:20>")
    parser.add_argument("--epochs", type=int, default=100,
                        help="number of training epochs <default:100>")
    parser.add_argument("--batch-size", type=int, default=256,
                        help="batch size for stochastic gradient descent optimization <default:256>")
    parser.add_argument("--learning-rate", type=float, default=0.01,
                        help="learning rate for stochastic gradient descent optimization <default:0.01>")
    parser.add_argument("--g-factor", type=float, default=1.2,
                        help="multiplicity factor by which generator learning rate scales to that of discriminator <default:1.2>")
    parser.add_argument("--droprate", type=float, default=0.25,
                        help="droprate used across GAN model for generalization/robustness <default:0.25>")
    parser.add_argument("--momentum", type=float, default=0.8,
                        help="momentum used in discriminator batch-normalization <default:0.8>")
    parser.add_argument("--alpha", type=float, default=0.2,
                        help="alpha paramter used in discriminator leaky relu <default:0.2>")
    args = parser.parse_args()
    assert args.subtype in ["faces","mnist","fashion"]
    singularTrain(args.subtype,args.latent_dim,args.epochs,args.batch_size,
                  args.learning_rate,args.g_factor,args.droprate,args.momentum,
                  args.alpha)
