from __future__ import print_function, division
import numpy as np
import os
import cv2
from PIL import Image
import random
from functools import partial

import tensorflow as tf
from keras.models import Model, Sequential, load_model
from keras.layers.merge import _Merge
from keras.layers import Input, Conv2D, MaxPooling2D, ZeroPadding2D, Conv2D, BatchNormalization, UpSampling2D, Activation
from keras.layers import Reshape, Dropout, Concatenate, Lambda, Multiply, Add, Flatten, Dense
from keras_contrib.layers.normalization.instancenormalization import InstanceNormalization
from keras.layers.advanced_activations import LeakyReLU, PReLU
from keras.optimizers import Adam
from keras import backend as K
import keras
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import cv2
from sklearn.utils import shuffle
import random
import datetime
from keras.applications.vgg16 import VGG16
from keras.applications.vgg19 import VGG19
from keras.applications.resnet50 import ResNet50
import math
from skimage.measure import compare_psnr, compare_ssim
from keras.utils import multi_gpu_model
from scipy.stats import pearsonr

def load_confocal(input_shape=None, set=None, z_depth=None):
    1

class GAN(object):
    def __init__(self):

        self.channels = 3
        self.hr_height = 512
        self.hr_width = 512
        self.hr_shape = (self.hr_height, self.hr_width, self.channels)

        self.n_residual_blocks = 9

        optimizer = Adam(0.0001, 0.5, 0.99)

        self.vgg_hq = self.build_vgg_hr(name='vgg_hq')
        self.vgg_hq.trainable = False
        self.vgg_hq_m = multi_gpu_model(self.vgg_hq, gpus=4)
        self.vgg_hq_m.compile(loss='mse',
                         optimizer=optimizer,
                         metrics=['accuracy'])
        self.vgg_lq = self.build_vgg_hr(name='vgg_lq')
        self.vgg_lq.trainable = False
        self.vgg_lq_m = multi_gpu_model(self.vgg_lq, gpus=4)
        self.vgg_lq_m.compile(loss='mse',
                         optimizer=optimizer,
                         metrics=['accuracy'])

        patch_hr_h = int(self.hr_height / 2 ** 4)
        patch_hr_w = int(self.hr_width / 2 ** 4)
        self.disc_patch_hr = (patch_hr_h, patch_hr_w, 1)

        self.gf = 64
        self.df = 64

        self.discriminator_hq = self.build_discriminator(name='dis_hq')
        self.discriminator_hq_m = multi_gpu_model(self.discriminator_hq, gpus=4)
        self.discriminator_hq_m.compile(loss='mse',
                                   optimizer=optimizer,
                                   metrics=['accuracy'])
        self.discriminator_lq = self.build_discriminator(name='dis_lq')
        self.discriminator_lq_m = multi_gpu_model(self.discriminator_lq, gpus=4)
        self.discriminator_lq_m.compile(loss='mse',
                                   optimizer=optimizer,
                                   metrics=['accuracy'])

        self.generator_lq2hq = self.build_generator(name='gen_lq2hq')
        self.generator_hq2lq = self.build_generator(name='gen_hq2lq')

        img_lq = Input(shape=self.hr_shape)
        img_hq = Input(shape=self.hr_shape)

        fake_hq = self.generator_lq2hq(img_lq)
        fake_lq = self.generator_hq2lq(img_hq)

        reconstr_lq = self.generator_hq2lq(fake_hq)
        reconstr_hq = self.generator_lq2hq(fake_lq)
        
        img_lq_id = self.generator_hq2lq(img_lq)
        img_hq_id = self.generator_lq2hq(img_hq)
        
        fake_hq_features = self.vgg_hq(fake_hq)
        fake_lq_features = self.vgg_lq(fake_lq)

        reconstr_hq_features = self.vgg_hq(reconstr_hq)
        reconstr_lq_features = self.vgg_lq(reconstr_lq)

        self.discriminator_hq.trainable = False
        self.discriminator_lq.trainable = False

        validity_hq = self.discriminator_hq(fake_hq)
        validity_lq = self.discriminator_lq(fake_lq)

        validity_reconstr_hq = self.discriminator_hq(reconstr_hq)
        validity_reconstr_lq = self.discriminator_lq(reconstr_lq)

        self.combined_hq = Model([img_lq, img_hq], [validity_hq, validity_reconstr_lq,
                                                 fake_hq_features, reconstr_lq_features, img_lq_id])
        self.combined_hq_m = multi_gpu_model(self.combined_hq, gpus=4)
        self.combined_hq_m.compile(loss=['mse', 'mse', 'mse', 'mse', 'mse'],
                              loss_weights=[1e-3, 1e-3, 1, 1, 1],
                              optimizer=optimizer)
        self.combined_lq = Model([img_lq, img_hq], [validity_lq, validity_reconstr_hq,
                                                 fake_lq_features, reconstr_hq_features, img_hq_id])
        self.combined_lq_m = multi_gpu_model(self.combined_lq, gpus=4)
        self.combined_lq_m.compile(loss=['mse', 'mse', 'mse', 'mse', 'mse'],
                              loss_weights=[1e-3, 1e-3, 1, 1, 1],
                              optimizer=optimizer)
    def build_vgg_hr(self, name=None):
        vgg = VGG19(include_top=False)
        vgg.outputs = [vgg.layers[9].output]
        img = Input(shape=self.hr_shape)

        img_features = vgg(img)
        model = Model(img, img_features, name=name)
        model.summary()
        return model


    def build_generator(self, name=None):
        def residual_block(layer_input, filters):
            d = Conv2D(filters, kernel_size=3, strides=1, padding='same')(layer_input)
            d = InstanceNormalization()(d)
            d = Activation('relu')(d)
            d = Conv2D(filters, kernel_size=3, strides=1, padding='same')(d)
            d = InstanceNormalization()(d)
            d = Add()([d, layer_input])
            return d

        img_lr = Input(shape=self.hr_shape)
        c1 = Conv2D(64, kernel_size=7, strides=1, padding='same')(img_lr)
        c1 = InstanceNormalization()(c1)
        c1 = Activation('relu')(c1)

        n_downsampling = 2
        for i in range(n_downsampling):
            mult = 2 ** i
            c1 = Conv2D(filters=64 * mult * 2, kernel_size=(3, 3), strides=2, padding='same')(c1)
            c1 = InstanceNormalization()(c1)
            c1 = Activation('relu')(c1)

        r = residual_block(c1, self.gf * (n_downsampling ** 2))
        for _ in range(8):
            r = residual_block(r, self.gf * (n_downsampling ** 2))

        for i in range(n_downsampling):
            mult = 2 ** (n_downsampling - i)
            r = UpSampling2D()(r)
            r = Conv2D(filters=int(64 * mult / 2), kernel_size=(3, 3), padding='same')(r)
            r = InstanceNormalization()(r)
            r = Activation('relu')(r)

        c2 = Conv2D(self.channels, kernel_size=7, strides=1, padding='same')(r)
        c2 = Activation('tanh')(c2)
        c2 = Add()([c2, img_lr])
        model = Model(img_lr, [c2], name=name)

        return model

    def build_discriminator(self, name=None):
        n_layers, use_sigmoid = 3, False
        inputs = Input(shape=self.hr_shape)
        ndf=64
        x = Conv2D(filters=ndf, kernel_size=(4, 4), strides=2, padding='same')(inputs)
        x = LeakyReLU(0.2)(x)

        nf_mult, nf_mult_prev = 1, 1
        for n in range(n_layers):
            nf_mult_prev, nf_mult = nf_mult, min(2 ** n, 8)
            x = Conv2D(filters=ndf * nf_mult, kernel_size=(4, 4), strides=2, padding='same')(x)
            x = BatchNormalization()(x)
            x = LeakyReLU(0.2)(x)

        nf_mult_prev, nf_mult = nf_mult, min(2 ** n_layers, 8)
        x = Conv2D(filters=ndf * nf_mult, kernel_size=(4, 4), strides=1, padding='same')(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(0.2)(x)

        x = Conv2D(filters=1, kernel_size=(4, 4), strides=1, padding='same')(x)
        if use_sigmoid:
            x = Activation('sigmoid')(x)

        x = Dense(1024, activation='tanh')(x)
        x = Dense(1, activation='sigmoid')(x)

        model = Model(inputs=inputs, outputs=x, name=name)

        return model

    def train(self, model, epochs, batch_size, set=None, z_depth=None):
        input_shape = (128, 128, 3)
        start_time = datetime.datetime.now()
        weigths_dir = model + '_weights'
        img_dir = model + '_img'
        log_dir = model + '_logs/' + datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
        if not os.path.exists(weigths_dir):
            os.makedirs(weigths_dir)
        if not os.path.exists(img_dir):
            os.makedirs(img_dir)
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        hrhq_train, hrhq_test, lrhq_train, lrhq_test, hrlq_train, hrlq_test, lrlq_train, lrlq_test = load_confocal(
            input_shape=input_shape,
            set=set,z_depth=z_depth)

        for epoch in range(epochs):
            idx = np.random.randint(0, hrhq_train.shape[0], batch_size)
            imgs_hrhq_train, imgs_lrhq_train, imgs_hrlq_train, imgs_lrlq_train = \
                hrhq_train[idx], lrhq_train[idx], hrlq_train[idx], lrlq_train[idx]

            if epoch > int(epochs/2):
                K.set_value(self.discriminator_hq_m.optimizer.lr,
                            1e-4 * (epochs - epoch) / (epochs - int(epochs/2)))
                K.set_value(self.discriminator_lq_m.optimizer.lr,
                            1e-4 * (epochs - epoch) / (epochs - int(epochs/2)))
                K.set_value(self.combined_hq_m.optimizer.lr,
                            1e-4 * (epochs - epoch) / (epochs - int(epochs / 2)))
                K.set_value(self.combined_lq_m.optimizer.lr,
                            1e-4 * (epochs - epoch) / (epochs - int(epochs / 2)))

            fake_hrhq = self.generator_lq2hq.predict(imgs_hrlq_train, batch_size=1)
            fake_hrlq = self.generator_hq2lq.predict(imgs_hrhq_train, batch_size=1)

            valid_hr = np.ones((batch_size,) + self.disc_patch_hr)
            fake_hr = np.zeros((batch_size,) + self.disc_patch_hr)

            d_loss_real_hr = self.discriminator_hq_m.train_on_batch(imgs_hrhq_train, valid_hr)
            d_loss_fake_hr = self.discriminator_hq_m.train_on_batch(fake_hrhq, fake_hr)
            d_loss_hr = 0.5 * np.add(d_loss_real_hr, d_loss_fake_hr)
            d_loss_real_lr = self.discriminator_lq_m.train_on_batch(imgs_hrlq_train, valid_hr)
            d_loss_fake_lr = self.discriminator_lq_m.train_on_batch(fake_hrlq, fake_hr)
            d_loss_lr = 0.5 * np.add(d_loss_real_lr, d_loss_fake_lr)

            image_features_hq = self.vgg_hq.predict(imgs_hrhq_train)
            image_features_lq = self.vgg_lq.predict(imgs_hrlq_train)

            g_loss_hq = self.combined_hq_m.train_on_batch([imgs_hrlq_train, imgs_hrhq_train],
                                                  [valid_hr, valid_hr,
                                                   image_features_hq,
                                                   image_features_lq, imgs_hrlq_train])
            g_loss_lq = self.combined_lq_m.train_on_batch([imgs_hrlq_train, imgs_hrhq_train],
                                                  [valid_hr, valid_hr,
                                                  image_features_lq,
                                                  image_features_hq, imgs_hrhq_train])

            elapsed_time = datetime.datetime.now() - start_time
            if (epoch + 1) % 100 == 0:
                print('Iteration : ' + str(epoch + 1) + '/' + str(epochs))
                print('time : ' + str(elapsed_time))
                print('D/loss_hq : ' + str(d_loss_hr[0]) + ' D/acc_hq : ' + str(d_loss_hr[1]))
                print('D/loss_lq : ' + str(d_loss_lr[0]) + ' D/acc_lq : ' + str(d_loss_lr[1]))

            if epoch % (sample_interval*10) == 0:
                self.generator_lq2hq.save(weigths_dir + "/generator_l2h.h5")
                self.generator_hq2lq.save(weigths_dir + "/generator_h2l.h5")
                self.discriminator_lq.save(weigths_dir + "/discriminator_l.h5")
                self.discriminator_hq.save(weigths_dir + "/discriminator_h.h5")
                # self.generator_lq2hq.save_weights(weigths_dir + "/generator_l2h_weights.h5")
                # self.generator_hq2lq.save_weights(weigths_dir + "/generator_h2l_weights.h5")
                # self.discriminator_lq.save_weights(weigths_dir + "/discriminator_l_weights.h5")
                # self.discriminator_hq.save_weights(weigths_dir + "/discriminator_h_weights.h5")

if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1, 2, 3"

    gan = GAN()
    epoch = 25000
    set = 'C0depth'
    z_depth = 'Z005'
    model = 'gan' + '_' + set + '_' + z_depth
    batch_size = 4
    gan.train(model=model, epochs=epoch, batch_size=batch_size, set=set, z_depth=z_depth)
