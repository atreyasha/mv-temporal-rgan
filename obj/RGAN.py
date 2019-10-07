#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# get all dependencies
import re
import csv
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
    def __init__(self,latent_dim=28,im_dim=32,epochs=100,batch_size=128,learning_rate=0.01,
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

    def _plot_figures(self,figures,direct,epoch,dim=1):
        """Plot a dictionary of figures.
        adapted from https://stackoverflow.com/questions/11159436/multiple-figures-in-a-single-window
        Parameters
        ----------
        figures : <title, figure> dictionary
        ncols : number of columns of subplots wanted in the display
        nrows : number of rows of subplots wanted in the figure
        """
        fig, axeslist = plt.subplots(ncols=dim, nrows=dim)
        for ind,title in enumerate(figures):
            axeslist.ravel()[ind].imshow(figures[title], cmap=plt.gray())
            axeslist.ravel()[ind].set_title(title)
            axeslist.ravel()[ind].set_axis_off()
        plt.tight_layout()
        fig.savefig("./pickles/"+direct+"/img/epoch"+str(epoch+1)+".png", format='png', dpi=500)

    def train(self,data,direct,sq_dim=4):
        plot_samples=sq_dim**2
        data_type = re.sub(r".*\\_","",direct)
        # write init.csv to file for future class reconstruction
        with open("pickles/"+direct+"/init.csv", "w") as csvfile:
            fieldnames = ["data", "im_dim", "latent_dim", "epochs", "batch_size", "learning_rate", "droprate", "g_factor"]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerow({"data":data_type, "im_dim":str(self.im_dim), "latent_dim":str(self.latent_dim),
                             "epochs":str(self.epochs), "batch_size":str(self.batch_size), "learning_rate":str(self.learning_rate),
                             "droprate":str(self.droprate), "g_factor":str(self.g_factor)})
        csvfile = open("pickles/"+direct+"/log.csv", "w")
        fieldnames = ["epoch", "batch", "d_loss", "d_acc", "g_loss", "g_acc"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        csvfile.flush()
        # generate constant noise vector for model comparisons
        np.random.seed(42)
        constant_noise = np.random.normal(size=(plot_samples,self.im_dim*self.latent_dim,1))
        np.random.seed(None)
        real_labels = np.ones((self.batch_size,1))
        fake_labels = np.zeros((self.batch_size,1))
        runs = int(np.ceil(data.shape[0]/self.batch_size))
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
                    writer.writerow({"epoch":str(epoch+1), "batch":str(batch+1), "d_loss":str(d_loss[0]),
                             "d_acc":str(d_loss[1]), "g_loss":str(g_loss[0]), "g_acc":str(g_loss[1])})
                    csvfile.flush()
            # at every epoch, generate 16 images for reference
            test_img = np.resize(self.generator.predict(constant_noise),(plot_samples,self.im_dim,self.im_dim))
            test_img = {str(i+1):test_img[i] for i in range(test_img.shape[0])}
            self._plot_figures(test_img,direct,epoch,sq_dim)
        # save model weights at end of training
        self.generator.save_weights("./pickles/"+direct+"/gen.h5")
        self.discriminator.save_weights("./pickles/"+direct+"/dis.h5")
        self.combined.save_weights("./pickles/"+direct+"/comb.h5")
