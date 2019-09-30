#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# get all dependencies
import numpy as np
import matplotlib.pyplot as plt
from keras import backend
from keras.models import Model
from keras.constraints import max_norm
from keras.optimizers import Adam
from keras.layers import Dense, Activation, Reshape
from keras.layers import LSTM, CuDNNLSTM, Input, TimeDistributed
from keras.backend.tensorflow_backend import clear_session

################################
# define class and functions
################################

class RGAN():
    def __init__(self,latent_dim=12,im_dim=28,epochs=100,batch_size=64,learning_rate=0.0001,
                 g_factor=1.5,droprate=0.2):
        # define and store local variables
        clear_session()
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.g_factor = g_factor
        self.optimizer_d = Adam(self.learning_rate)
        self.optimizer_g = Adam(self.learning_rate*self.g_factor)
        self.latent_dim = latent_dim
        self.im_dim = im_dim
        self.droprate = droprate
        # define and compile discriminator
        self.discriminator = self.getDiscriminator(self.im_dim,self.droprate)
        self.discriminator.compile(loss=['binary_crossentropy'], optimizer=self.optimizer_d,
            metrics=['accuracy'])
        # define generator
        self.generator = self.getGenerator(self.im_dim,self.latent_dim,self.droprate)
        self.discriminator.trainable = False
        # define combined network with partial gradient application
        z = Input(shape=(self.im_dim*self.latent_dim,1))
        img = self.generator(z)
        validity = self.discriminator(img)
        self.combined = Model(z, validity)
        self.combined.compile(loss=['binary_crossentropy'], optimizer=self.optimizer_g,
                              metrics=['accuracy'])

    def getGenerator(self,im_dim,latent_dim,droprate):
        in_data = Input(shape=(im_dim*latent_dim,1))
        # possible dense layer to reduce dimensions and noise
        # out = TimeDistributed(Dense(20))(in_data)
        # out = TimeDistributed(Dense(28))(in_data)
        # out = Activation("relu")(out)
        if len(backend.tensorflow_backend._get_available_gpus()) > 0:
            out = CuDNNLSTM(im_dim,
                    kernel_constraint=max_norm(3), recurrent_constraint=max_norm(3),
                    bias_constraint=max_norm(3))(in_data)
        else:
            out = LSTM(im_dim,recurrent_dropout=droprate,
                    kernel_constraint=max_norm(3), recurrent_constraint=max_norm(3),
                    bias_constraint=max_norm(3))(in_data)
        out = Dense(im_dim**2)(out)
        out = Reshape((im_dim**2,1))(out)
        return Model(inputs=in_data,outputs=out)

    def getDiscriminator(self,im_dim,droprate):
        in_data = Input(shape=(im_dim**2,1))
        if len(backend.tensorflow_backend._get_available_gpus()) > 0:
            out = CuDNNLSTM(28,kernel_constraint=max_norm(3),recurrent_constraint
                   =max_norm(3),bias_constraint=max_norm(3))(in_data)
        else:
            out = LSTM(28,recurrent_dropout=droprate,kernel_constraint=max_norm(3),recurrent_constraint
                   =max_norm(3),bias_constraint=max_norm(3))(in_data)
        out = Dense(1)(out)
        out = Activation("sigmoid")(out)
        return Model(inputs=in_data,outputs=out)

    def train(self,data,direct):
        np.random.seed(42)
        constant_noise = np.random.normal(size=(1,self.im_dim*self.latent_dim,1))
        real_labels = np.ones((self.batch_size,1))
        fake_labels = np.zeros((self.batch_size,1))
        runs = int(np.ceil(data.shape[0]/128))
        for epoch in range(self.epochs):
            for batch in range(runs):
                # randomize data and generate noise
                idx = np.random.randint(0, data.shape[0],self.batch_size)
                real = data[idx]
                noise = np.random.normal(size=(self.batch_size,self.im_dim*self.latent_dim,1))
                # generate fake data
                fake = self.generator.predict(noise)
                # train the discriminator
                d_loss_real = self.discriminator.train_on_batch(real, real_labels)
                d_loss_fake = self.discriminator.train_on_batch(fake, fake_labels)
                d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
                # generate new set of noise
                noise = np.random.normal(size=(self.batch_size,self.im_dim*self.latent_dim,1))
                # train generator while freezing discriminator
                g_loss = self.combined.train_on_batch(noise, real_labels)
                # plot the progress
                if (batch+1) % 20 == 0:
                    print("epoch: %d [batch: %d] [D loss: %f, acc.: %.2f%%] [G loss: %f, acc.: %.2f%%]" % (epoch+1,batch+1,d_loss[0],100*d_loss[1],g_loss[0],100*g_loss[1]))
            # at every epoch, generate an image for reference
            test_img = np.resize(self.generator.predict(constant_noise)[0],(self.im_dim,self.im_dim))
            plt.imsave("./pickles/"+direct+"/img/epoch"+str(epoch+1)+".png",test_img)

################################
# comments/to-dos
################################

# convert faces to newer and better format hp5
# reduce cpu variable conversion in colab

# add function for checking generation process and constraints
# save models at each epoch and decide quality by factor of gen vs dis
# add gradient checks to early stopping mechanism
# add some steps to configure g-factor while training

# make mechanism for early stopping within training
# try to use early checkpoint method with some modification
# add sample images to keep track of training progress
# add sample generation layer and saving model function

# clean code, try running on google cpu
# make network deeper and emulate cross relationships similar to pixelnet
# use convolutions in hidden layers
# save images as tif for best preview

# try deforming multivariate time series to single series
# then model via lstm and possibly cnn
# see if basic reproduction is possible
# try to model via MNIST in worst case scenario

# TODO:
# grid-search:
# set up git repository and add GPU support
# let grid-search run on colab for at least 100 epochs
# apply some basic filtering such as limits of loss ratios
# make some early stopping mechanisms and save models to check for convergence

# networks:
# consider changing LSTM's to bidirectional
# consider possibly downsampling, but try with high dimensions to check viability
# consider adding convolutions in both generator and discriminator for locality
# extend to RCGAN with realistic conditionings such as gender/smile/expression
# consider adding convolutions where this might be useful
# implement single run and then grid-search

# masking varied features:
# come up with mask to create or ignore feature differences
# can be included within images

# input images:
# consider downsampling to save memory and computational power
# consider normalizing in a different way, via local max or possible integration

# code-health:
# fix unused imports and sort with python tools
