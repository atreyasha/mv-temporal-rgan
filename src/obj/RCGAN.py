#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import re
import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from keras import backend
from .model_utils import save_model
from keras.models import Model
from keras.optimizers import Adam
from keras.constraints import max_norm
from keras.layers import Dense, Activation, Reshape, Flatten, Embedding
from keras.layers import LSTM, CuDNNLSTM, Input, Bidirectional, Multiply
from keras.layers import BatchNormalization, LeakyReLU, Dropout, UpSampling2D
from .spec_norm.SpectralNormalizationKeras import ConvSN2D, DenseSN
from keras.backend.tensorflow_backend import clear_session

class RCGAN():
    """ Class definition for RCGAN """
    def __init__(self,num_classes,latent_dim=100,im_dim=28,epochs=100,
                 batch_size=256,learning_rate=0.0004,g_factor=0.25,
                 droprate=0.25,momentum=0.8,alpha=0.2,saving_rate=10):
        """
        Initialize RCGAN with model parameters

        Args:
            num_classes (int): number of unique classes
            latent_dim (int): latent dimensions of generator
            im_dim (int): square dimensionality of images
            epochs (int): maximum number of training epochs
            batch_size (int): batch size for stochastic gradient descent
            learning_rate (float): learning rate for stochastic gradient descent,
            particularly for the discriminator
            g_factor (float): learning rate for generator =
            g_factor*learning_rate, which is defined above
            droprate (float): dropout-rate used within the model
            momentum (float): momentum used in batch normalization
            alpha (float): alpha used in leaky relu
            saving_rate (int): epoch interval when model is saved
        """
        # define and store local variables
        clear_session()
        self.num_classes = num_classes
        self.latent_dim = latent_dim
        self.im_dim = im_dim
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.g_factor = g_factor
        self.optimizer_d = Adam(self.learning_rate,0.5)
        self.optimizer_g = Adam(self.learning_rate*self.g_factor,0.5)
        self.droprate = droprate
        self.momentum = momentum
        self.alpha = alpha
        self.saving_rate = saving_rate
        losses = ["binary_crossentropy","sparse_categorical_crossentropy"]
        # define and compile discriminator
        self.discriminator = self.getDiscriminator(self.im_dim,self.droprate,
                                                   self.momentum,self.alpha,
                                                   self.num_classes)
        self.discriminator.compile(loss=losses, optimizer=self.optimizer_d)
        # define generator
        self.generator = self.getGenerator(self.latent_dim,self.momentum,
                                           self.alpha,self.num_classes)
        self.discriminator.trainable = False
        # define combined network with partial gradient application
        noise = Input(shape=(self.latent_dim,))
        label = Input(shape=(1,),dtype="int32")
        img = self.generator([noise, label])
        validity,target_label = self.discriminator(img)
        self.combined = Model([noise,label],[validity,target_label])
        self.combined.compile(loss=losses, optimizer=self.optimizer_g)

    def getGenerator(self,latent_dim,momentum,alpha,num_classes):
        """
        Initialize generator model

        Args:
            latent_dim (int): latent dimensions of generator
            momentum (float): momentum used in batch normalization
            alpha (float): alpha used in leaky relu
            num_classes (int): number of unique classes

        Returns:
            (keras.models.Model): keras model for generator
        """
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
                       recurrent_constraint=max_norm(3),
                            bias_constraint=max_norm(3))(out)
        else:
            out = LSTM(32,return_sequences=True,
                       kernel_constraint=max_norm(3),
                       recurrent_constraint=max_norm(3),
                       bias_constraint=max_norm(3))(out)
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
        """
        Initialize discriminator model

        Args:
            im_dim (int): square dimensionality of images
            droprate (float): dropout-rate used within the model
            momentum (float): momentum used in batch normalization
            alpha (float): alpha used in leaky relu
            num_classes (int): number of unique classes

        Returns:
            (keras.models.Model): keras model for discriminator
        """
        # reprocess image with provided label
        in_data = Input(shape=(im_dim,im_dim))
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
                       recurrent_constraint=max_norm(3),
                            bias_constraint=max_norm(3))(out)
        else:
            out = LSTM(1,return_sequences=True,
                       kernel_constraint=max_norm(3),
                       recurrent_constraint=max_norm(3),
                       bias_constraint=max_norm(3))(out)
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
            out = Bidirectional(CuDNNLSTM(32,
                       kernel_constraint=max_norm(3),
                       recurrent_constraint=max_norm(3),
                                          bias_constraint=max_norm(3)))(out)
        else:
            out = Bidirectional(LSTM(32,
                       kernel_constraint=max_norm(3),
                       recurrent_constraint=max_norm(3),
                                     bias_constraint=max_norm(3)))(out)
        # block 6: map final features to dense output
        validity = DenseSN(16)(out)
        validity = BatchNormalization(momentum=momentum)(validity)
        validity = LeakyReLU(alpha=alpha)(validity)
        validity = Dropout(droprate)(validity)
        validity = Dense(1)(validity)
        validity = Activation("sigmoid")(validity)
        # classify to actual class
        label = DenseSN(num_classes)(out)
        label = Activation("softmax")(label)
        return Model(inputs=in_data,outputs=[validity,label])

    def _plot_figures(self,gen_imgs,direct,epoch,plot_samples,
                      num_classes,constant_labels):
        """
        Plot images produced from model

        Args:
            gen_imgs (numpy.ndarray): interim images produced by model
            direct (str): log directory to save plot
            epoch (int): current epoch for plot
            plot_samples (int): number of samples per class
            num_classes (int): number of unique classes
            constant_labels (int): indices of unique classes
        """
        # plotting function
        gen_imgs = 0.5 * gen_imgs + 0.5
        fig, axs = plt.subplots(ncols=num_classes,nrows=plot_samples)
        cnt = 0
        for j in range(num_classes):
            for i in range(plot_samples):
                axs[i,j].imshow(gen_imgs[cnt], cmap='gray')
                if i == 0:
                    axs[i,j].set_title("%d" % constant_labels[cnt])
                axs[i,j].axis('off')
                cnt += 1
        plt.tight_layout()
        fig.savefig("./pickles/"+direct+"/img/epoch"+str(epoch+1)+".png",
                    format='png', dpi=500)
        fig.clear()
        plt.close("all")

    def train(self,data,direct,plot_samples=5,check_rate=20):
        """
        Train RCGAN model

        Args:
            data (numpy.ndarray): numpy array of training data
            direct (str): log-directory to store model
            plot_samples (int): number of samples per class
            check_rate (int): epoch interval to log performance
        """
        data_type = re.sub(r".*_","",direct)
        dict_field = {"data":data_type}
        dict_field.update({el[0]:el[1] for el in self.__dict__.items()
                           if type(el[1]) in
                           [int,str,float,np.int64,np.float64]})
        fieldnames = list(dict_field.keys())
        # write init.csv to file for future class reconstruction
        with open("./pickles/"+direct+"/init.csv", "w") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerow(dict_field)
        fieldnames = ["epoch", "batch", "d_loss", "d_a_loss", "g_loss",
                      "g_a_loss"]
        with open("./pickles/"+direct+"/log.csv", "w") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
        # generate constant noise vector for model comparisons
        np.random.seed(42)
        constant_noise = np.random.normal(size=(plot_samples,self.latent_dim))
        constant_noise = np.concatenate(np.repeat(constant_noise[None, :],
                                                  self.num_classes,
                                                  axis=0),axis=0)
        constant_labels = np.repeat(np.arange(0,self.num_classes),plot_samples)
        np.random.seed(None)
        # generate target labels
        fake_labels = np.zeros((self.batch_size,1))
        runs = int(np.ceil(data[0].shape[0]/self.batch_size))
        for epoch in range(self.epochs):
            # make noisy real labels per epoch
            real_labels = np.clip(np.random.normal(loc=0.90,
                                                   scale=0.005,size=
                                                   (self.batch_size,1)),None,1)
            for batch in range(runs):
                # randomize data and generate noise
                idx = np.random.randint(0,data[0].shape[0],self.batch_size)
                real_imgs, img_labels = data[0][idx], data[1][idx]
                noise = np.random.normal(size=(self.batch_size,
                                               self.latent_dim,))
                sampled_img_labels = np.random.randint(0, self.num_classes,
                                                       self.batch_size)
                # generate fake data
                fake_imgs = self.generator.predict([noise,sampled_img_labels])
                # train the discriminator
                d_loss_real = self.discriminator.train_on_batch(real_imgs,
                                                                [real_labels,
                                                                 img_labels])
                d_loss_fake = self.discriminator.train_on_batch(fake_imgs,
                                                                [fake_labels,
                                                                 sampled_img_labels])
                d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
                # train generator while freezing discriminator
                g_loss = self.combined.train_on_batch([noise,
                                                       sampled_img_labels],
                                                      [real_labels,
                                                       sampled_img_labels])
                # plot the progress
                if (batch+1) % check_rate == 0:
                    print("epoch: %d [batch: %d] [D loss: %f, D.A Loss: %f] [G loss: %f, G.A Loss: %f]"
                          % (epoch+1,batch+1,d_loss[0],d_loss[1],
                             g_loss[0],g_loss[1]))
                    with open("./pickles/"+direct+"/log.csv", "a") as csvfile:
                        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                        writer.writerow({"epoch":str(epoch+1),
                                         "batch":str(batch+1),
                                         "d_loss":str(d_loss[0]),
                                         "g_loss":str(g_loss[0]),
                                         "d_a_loss":str(d_loss[1]),
                                         "g_a_loss":str(g_loss[1])})
            # at every epoch, generate images for reference
            test_img = self.generator.predict([constant_noise,constant_labels])
            self._plot_figures(test_img,direct,epoch,plot_samples,
                               self.num_classes,constant_labels)
            if (epoch+1) % self.saving_rate == 0 or (epoch+1) == self.epochs:
                # save models with defined periodicity
                save_model(self,direct)
