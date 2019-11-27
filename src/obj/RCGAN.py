#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# import all dependencies
import re
import pickle
import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from keras import backend
from keras.models import Model
from keras.optimizers import Adam
from keras.constraints import max_norm
from keras.layers import Dense, Activation, Reshape, Flatten, Embedding
from keras.layers import LSTM, CuDNNLSTM, Input, Bidirectional, Conv2D, Multiply
from keras.layers import BatchNormalization, LeakyReLU, Dropout, UpSampling2D
from .spec_norm.SpectralNormalizationKeras import ConvSN2D, DenseSN
from keras.backend.tensorflow_backend import clear_session

################################
# define class and functions
################################

# TODO: add conditional workflow
# TODO: figure out how to deal with num_classes variable
# TODO: modify plotting pipeline to include all indices to plot
# TODO: modify train and other logging functions to enable labels
# make test runs to ensure correct logging procedures
# create labels for faces if possible, perhaps for basic facial indicators (can also re-publish)
# update readme with relevant changes

class RCGAN():
    def __init__(self,num_classes,latent_dim=100,im_dim=28,epochs=100,batch_size=256,
                 learning_rate=0.0004,g_factor=0.25,droprate=0.25,momentum=0.8,alpha=0.2,saving_rate=10):
        # define and store local variables
        clear_session()
        self.num_classes = num_classes
        self.latent_dim = latent_dim
        self.im_dim = im_dim
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.g_factor = g_factor
        self.optimizer_d = Adam(self.learning_rate)
        self.optimizer_g = Adam(self.learning_rate*self.g_factor)
        self.droprate = droprate
        self.momentum = momentum
        self.alpha = alpha
        self.saving_rate = saving_rate
        # define and compile discriminator
        self.discriminator = self.getDiscriminator(self.im_dim,self.droprate,self.momentum,
                                                   self.alpha,self.num_classes)
        self.discriminator.compile(loss=['binary_crossentropy'], optimizer=self.optimizer_d)
        # define generator
        self.generator = self.getGenerator(self.latent_dim,self.momentum,
                                           self.alpha,self.num_classes)
        self.discriminator.trainable = False
        # define combined network with partial gradient application
        noise = Input(shape=(self.latent_dim,))
        label = Input(shape=(1,),dtype="int32")
        img = self.generator([noise, label])
        validity = self.discriminator([img,label])
        self.combined = Model([noise,label], validity)
        self.combined.compile(loss=['binary_crossentropy'], optimizer=self.optimizer_g)

    def getGenerator(self,latent_dim,momentum,alpha,num_classes):
        # generate conditional noise vectors
        noise = Input(shape=(latent_dim,))
        label = Input(shape=(1,), dtype='int32')
        label_embedding = Flatten()(Embedding(num_classes,latent_dim)(label))
        in_data = Multiply()([noise,label_embedding])
        # block 1: upsampling using dense layers
        out = DenseSN(128*49)(in_data)
        out = LeakyReLU(alpha=alpha)(out)
        out = Reshape((7,7,128))(out)
        # block 2: convolution
        out = ConvSN2D(256, kernel_size=3, padding="same")(out)
        out = BatchNormalization(momentum=momentum)(out)
        out = LeakyReLU(alpha=alpha)(out)
        # block 3: upsampling and convolution
        out = UpSampling2D()(out)
        out = ConvSN2D(128, kernel_size=3, padding="same")(out)
        out = BatchNormalization(momentum=momentum)(out)
        out = LeakyReLU(alpha=alpha)(out)
        # block 4: upsampling and convolution
        out = UpSampling2D()(out)
        out = ConvSN2D(64, kernel_size=4, padding="same")(out)
        out = BatchNormalization(momentum=momentum)(out)
        out = LeakyReLU(alpha=alpha)(out)
        # block 5: flatten and enrich string features using LSTM
        out = Reshape((28*28,64))(out)
        if len(backend.tensorflow_backend._get_available_gpus()) > 0:
            out = CuDNNLSTM(32,return_sequences=True,
                       kernel_constraint=max_norm(3),
                       recurrent_constraint=max_norm(3),bias_constraint=max_norm(3))(out)
        else:
            out = LSTM(32,return_sequences=True,
                       kernel_constraint=max_norm(3),
                       recurrent_constraint=max_norm(3),bias_constraint=max_norm(3))(out)
        out = Reshape((28,28,32))(out)
        # block 6: continuous convolutions for smoother features
        out = ConvSN2D(32, kernel_size=3, padding="same")(out)
        out = BatchNormalization(momentum=momentum)(out)
        out = ConvSN2D(32, kernel_size=3, padding="same")(out)
        out = BatchNormalization(momentum=momentum)(out)
        out = ConvSN2D(1, kernel_size=3, padding="same")(out)
        out = BatchNormalization(momentum=momentum)(out)
        out = LeakyReLU(alpha=alpha)(out)
        out = Reshape((28,28))(out)
        return Model(inputs=[noise,label],outputs=out)

    def getDiscriminator(self,im_dim,droprate,momentum,alpha,num_classes):
        # reprocess image with provided label
        img = Input(shape=(im_dim,im_dim))
        label = Input(shape=(1,), dtype='int32')
        label_embedding = Flatten()(Embedding(num_classes, im_dim**2)(label))
        flat_img = Flatten()(img)
        in_data = Multiply()([flat_img, label_embedding])
        # initial convolution to prevent artifacts
        out = Reshape((im_dim,im_dim,1))(in_data)
        out = ConvSN2D(1, kernel_size=3, padding="same")(out)
        out = BatchNormalization(momentum=momentum)(out)
        out = LeakyReLU(alpha=alpha)(out)
        out = Dropout(droprate)(out)
        # block 1: flatten and check sequence using LSTM
        out = Reshape((im_dim**2,1))(out)
        if len(backend.tensorflow_backend._get_available_gpus()) > 0:
            out = CuDNNLSTM(1,return_sequences=True,
                       kernel_constraint=max_norm(3),
                       recurrent_constraint=max_norm(3),bias_constraint=max_norm(3))(out)
        else:
            out = LSTM(1,return_sequences=True,
                       kernel_constraint=max_norm(3),
                       recurrent_constraint=max_norm(3),bias_constraint=max_norm(3))(out)
        out = Reshape((im_dim,im_dim,1))(out)
        # block 2: convolution with dropout
        out = ConvSN2D(256, kernel_size=3, strides=2)(out)
        out = BatchNormalization(momentum=momentum)(out)
        out = LeakyReLU(alpha=alpha)(out)
        out = Dropout(droprate)(out)
        # block 3: convolution with dropout
        out = ConvSN2D(128, kernel_size=3, strides=2)(out)
        out = BatchNormalization(momentum=momentum)(out)
        out = LeakyReLU(alpha=alpha)(out)
        out = Dropout(droprate)(out)
        # block 4: convolution with dropout
        out = ConvSN2D(64, kernel_size=3)(out)
        out = BatchNormalization(momentum=momentum)(out)
        out = LeakyReLU(alpha=alpha)(out)
        out = Dropout(droprate)(out)
        # block 5: flatten and detect final features using bi-LSTM
        out = Reshape((4*4,64))(out)
        if len(backend.tensorflow_backend._get_available_gpus()) > 0:
            out = Bidirectional(CuDNNLSTM(8,
                       kernel_constraint=max_norm(3),
                       recurrent_constraint=max_norm(3),bias_constraint=max_norm(3)))(out)
        else:
            out = Bidirectional(LSTM(8,
                       kernel_constraint=max_norm(3),
                       recurrent_constraint=max_norm(3),bias_constraint=max_norm(3)))(out)
        # block 6: map final features to dense output
        out = Dense(1)(out)
        out = Activation("sigmoid")(out)
        return Model(inputs=[img,label],outputs=out)

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
        fig.clear()
        plt.close("all")

    def train(self,data,direct,sq_dim=4):
        plot_samples=sq_dim**2
        data_type = re.sub(r".*_","",direct)
        dict_field = {"data":data_type}
        dict_field.update({el[0]:el[1] for el in self.__dict__.items()
                           if type(el[1]) in [int,str,float,np.int64,np.float64]})
        fieldnames = list(dict_field.keys())
        # write init.csv to file for future class reconstruction
        with open("./pickles/"+direct+"/init.csv", "w") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerow(dict_field)
        fieldnames = ["epoch", "batch", "d_loss", "g_loss"]
        with open("./pickles/"+direct+"/log.csv", "w") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
        # generate constant noise vector for model comparisons
        np.random.seed(42)
        constant_noise = np.random.normal(size=(plot_samples,self.latent_dim,))
        np.random.seed(None)
        # label smoothing by using less-than-one value
        fake_labels = np.zeros((self.batch_size,1))
        runs = int(np.ceil(data[0].shape[0]/self.batch_size))
        for epoch in range(self.epochs):
            # make noisy labels per epoch
            real_labels = np.clip(np.random.normal(loc=0.90,
                                                   scale=0.005,size=(self.batch_size,1)),None,1)
            for batch in range(runs):
                # randomize data and generate noise
                idx = np.random.randint(0,data[0].shape[0],self.batch_size)
                real_imgs, img_labels = data[0][idx], data[1][idx]
                noise = np.random.normal(size=(self.batch_size,self.latent_dim,))
                # generate fake data
                fake_imgs = self.generator.predict([noise,labels])
                # train the discriminator
                d_loss_real = self.discriminator.train_on_batch([real_imgs,img_labels], real_labels)
                d_loss_fake = self.discriminator.train_on_batch([fake_imgs,img_labels], fake_labels)
                d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
                # generate new set of noise and sampled labels and sampled labels
                noise = np.random.normal(size=(self.batch_size,self.latent_dim,))
                sampled_img_labels = np.random.randint(0, self.num_classes+1, batch_size)
                # train generator while freezing discriminator
                g_loss = self.combined.train_on_batch([noise,sampled_img_labels], real_labels)
                # plot the progress
                if (batch+1) % 20 == 0:
                    print("epoch: %d [batch: %d] [D loss: %f] [G loss: %f]" %
                          (epoch+1,batch+1,d_loss,g_loss))
                    with open("./pickles/"+direct+"/log.csv", "a") as csvfile:
                        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                        writer.writerow({"epoch":str(epoch+1), "batch":str(batch+1),
                                         "d_loss":str(d_loss), "g_loss":str(g_loss)})
            # at every epoch, generate images for reference
            test_img = self.generator.predict(constant_noise)
            test_img = {str(i+1):test_img[i] for i in range(test_img.shape[0])}
            self._plot_figures(test_img,direct,epoch,sq_dim)
            if (epoch+1) % self.saving_rate == 0 or (epoch+1) == self.epochs:
                # save models with defined periodicity
                self.generator.save_weights("./pickles/"+direct+"/gen_weights.h5")
                self.discriminator.save_weights("./pickles/"+direct+"/dis_weights.h5")
                self.combined.save_weights("./pickles/"+direct+"/comb_weights.h5")
                with open("./pickles/"+direct+"/dis_opt_weights.pickle","wb") as f:
                    pickle.dump(self.discriminator.optimizer.get_weights(),f)
                with open("./pickles/"+direct+"/comb_opt_weights.pickle","wb") as f:
                    pickle.dump(self.combined.optimizer.get_weights(),f)