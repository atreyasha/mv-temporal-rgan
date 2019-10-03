#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# get all dependencies
import pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from keras import backend
from keras.models import Model
from keras.constraints import max_norm
from keras.optimizers import Adam
from keras.layers import Dense, Activation, Reshape
from keras.layers import LSTM, CuDNNLSTM, Input
from keras.backend.tensorflow_backend import clear_session

################################
# define class and functions
################################

class RGAN():
    def __init__(self,latent_dim=28,im_dim=28,epochs=100,batch_size=128,learning_rate=0.01,
                 g_factor=0.7,droprate=0.2):
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
        if len(backend.tensorflow_backend._get_available_gpus()) > 0:
            out = CuDNNLSTM(im_dim**2,
                    kernel_constraint=max_norm(3), recurrent_constraint=max_norm(3),
                    bias_constraint=max_norm(3))(in_data)
        else:
            out = LSTM(im_dim**2,recurrent_dropout=droprate,
                    kernel_constraint=max_norm(3), recurrent_constraint=max_norm(3),
                    bias_constraint=max_norm(3))(in_data)
        out = Reshape((im_dim**2,1))(out)
        return Model(inputs=in_data,outputs=out)

    def getDiscriminator(self,im_dim,droprate):
        in_data = Input(shape=(im_dim**2,1))
        if len(backend.tensorflow_backend._get_available_gpus()) > 0:
            out = CuDNNLSTM(im_dim*10,kernel_constraint=max_norm(3),recurrent_constraint
                   =max_norm(3),bias_constraint=max_norm(3))(in_data)
        else:
            out = LSTM(im_dim*10,recurrent_dropout=droprate,kernel_constraint=max_norm(3),recurrent_constraint
                   =max_norm(3),bias_constraint=max_norm(3))(in_data)
        out = Dense(100)(out)
        out = Activation("relu")(out)
        out = Dense(10)(out)
        out = Activation("relu")(out)
        out = Dense(1)(out)
        out = Activation("sigmoid")(out)
        return Model(inputs=in_data,outputs=out)

    def train(self,data,direct,plot_samples=10):
        np.random.seed(42)
        constant_noise = np.random.normal(size=(plot_samples,self.im_dim*self.latent_dim,1))
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
            test_img = np.resize(self.generator.predict(constant_noise),(plot_samples,self.im_dim,self.im_dim))
            test_img = np.hstack(test_img)
            fig, ax = plt.subplots()
            plt.imshow(test_img)
            fig.savefig("./pickles/"+direct+"/img/epoch"+str(epoch+1)+".png", format='png', dpi=500)
