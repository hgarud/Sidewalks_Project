#!/usr/bin/python
# -*- coding: UTF-8 -*-

import numpy as np
from keras.models import *
from keras.layers import Input, Conv2D, Conv2DTranspose, MaxPooling2D, Dropout, concatenate
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as keras


def unet(input_shape=None, n_labels=12, output_mode = "softmax"):
    inputs = Input(shape=input_shape)

    conv1 = Conv2D(64, 3, padding='same', kernel_initializer='he_normal')(inputs)
    conv1 = BatchNormalization()(conv1)
    conv1 = Activation("relu")(conv1)
    # print "conv1 shape:", conv1.shape
    conv1 = Conv2D(64, 3, padding='same', kernel_initializer='he_normal')(conv1)
    conv1 = BatchNormalization()(conv1)
    conv1 = Activation("relu")(conv1)
    # print "conv1 shape:", conv1.shape
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    # print "pool1 shape:", pool1.shape

    conv2 = Conv2D(128, 3, padding='same', kernel_initializer='he_normal')(pool1)
    conv2 = BatchNormalization()(conv2)
    conv2 = Activation("relu")(conv2)
    # print "conv2 shape:", conv2.shape
    conv2 = Conv2D(128, 3, padding='same', kernel_initializer='he_normal')(conv2)
    conv2 = BatchNormalization()(conv2)
    conv2 = Activation("relu")(conv2)
    # print "conv2 shape:", conv2.shape
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    # print "pool2 shape:", pool2.shape

    conv3 = Conv2D(256, 3, padding='same', kernel_initializer='he_normal')(pool2)
    conv3 = BatchNormalization()(conv3)
    conv3 = Activation("relu")(conv3)
    # print "conv3 shape:", conv3.shape
    conv3 = Conv2D(256, 3, padding='same', kernel_initializer='he_normal')(conv3)
    conv3 = BatchNormalization()(conv3)
    conv3 = Activation("relu")(conv3)
    # print "conv3 shape:", conv3.shape
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    # print "pool3 shape:", pool3.shape

    conv4 = Conv2D(512, 3, padding='same', kernel_initializer='he_normal')(pool3)
    conv4 = BatchNormalization()(conv4)
    conv4 = Activation("relu")(conv4)
    conv4 = Conv2D(512, 3, padding='same', kernel_initializer='he_normal')(conv4)
    conv4 = BatchNormalization()(conv4)
    conv4 = Activation("relu")(conv4)
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

    conv5 = Conv2D(1024, 3, padding='same', kernel_initializer='he_normal')(pool4)
    conv5 = BatchNormalization()(conv5)
    conv5 = Activation("relu")(conv5)
    conv5 = Conv2D(1024, 3, padding='same', kernel_initializer='he_normal')(conv5)
    conv5 = BatchNormalization()(conv5)
    conv5 = Activation("relu")(conv5)
    drop5 = Dropout(0.5)(conv5)

    up6 = Conv2DTranspose(512, (2,2),strides=(2,2), padding = 'same', kernel_initializer = 'he_normal')(drop5)
    up6 = BatchNormalization()(up6)
    up6 = Activation("relu")(up6)
    merge6 = concatenate([drop4,up6],axis=3)
    conv6 = Conv2D(512, 3, padding = 'same', kernel_initializer = 'he_normal')(merge6)
    conv6 = BatchNormalization()(conv6)
    conv6 = Activation("relu")(conv6)
    conv6 = Conv2D(512, 3, padding = 'same', kernel_initializer = 'he_normal')(conv6)
    conv6 = BatchNormalization()(conv6)
    conv6 = Activation("relu")(conv6)

    up7 = Conv2DTranspose(256,(2,2),strides=(2,2), padding = 'same', kernel_initializer = 'he_normal')(conv6)
    up7 = BatchNormalization()(up7)
    up7 = Activation("relu")(up7)
    merge7 =concatenate([conv3,up7],axis=3)
    conv7 = Conv2D(256, 3, padding = 'same', kernel_initializer = 'he_normal')(merge7)
    conv7 = BatchNormalization()(conv7)
    conv7 = Activation("relu")(conv7)
    conv7 = Conv2D(256, 3, padding = 'same', kernel_initializer = 'he_normal')(conv7)
    conv7 = BatchNormalization()(conv7)
    conv7 = Activation("relu")(conv7)
    
    up8 = Conv2DTranspose(128,(2,2),strides=(2,2), padding = 'same', kernel_initializer = 'he_normal')(conv7)
    up8 = BatchNormalization()(up8)
    up8 = Activation("relu")(up8)
    merge8 =concatenate([conv2,up8],axis=3)
    conv8 = Conv2D(128, 3, padding = 'same', kernel_initializer = 'he_normal')(merge8)
    conv8 = BatchNormalization()(conv8)
    conv8 = Activation("relu")(conv8)
    conv8 = Conv2D(128, 3, padding = 'same', kernel_initializer = 'he_normal')(conv8)
    conv8 = BatchNormalization()(conv8)
    conv8 = Activation("relu")(conv8)

    up9 = Conv2DTranspose(64,(2,2),strides=(2,2), padding = 'same', kernel_initializer = 'he_normal')(conv8)
    up9 = BatchNormalization()(up9)
    up9 = Activation("relu")(up9)
    merge9 =concatenate([conv1,up9],axis=3)
    conv9 = Conv2D(64, 3, padding = 'same', kernel_initializer = 'he_normal')(merge9)
    conv9 = BatchNormalization()(conv9)
    conv9 = Activation("relu")(conv9)
    conv9 = Conv2D(64, 3, padding = 'same', kernel_initializer = 'he_normal')(conv9)
    conv9 = BatchNormalization()(conv9)
    conv9 = Activation("relu")(conv9)
    
    conv10 = Conv2D(n_labels, 1, padding = 'same', kernel_initializer = 'he_normal')(conv9)
    conv_10 = BatchNormalization()(conv_10)
    conv_10 = Reshape((input_shape[0] * input_shape[1], n_labels), input_shape=(input_shape[0], input_shape[1], n_labels))(conv_10)

    outputs = Activation(output_mode)(conv_10)

    model = Model(input = inputs, output = outputs)

    return model

if __name__ == '__main__':
    input_shape = (224, 224, 3)
    n_labels = 12
    model = unet(input_shape=input_shape, n_labels=n_labels)
    model.summary()
