import numpy as np
import cv2
import os
from keras.models import Model
from keras.layers import Input
from keras.layers.core import Dense, Dropout, Activation, Flatten, Reshape, Permute
from keras.layers.convolutional import Convolution2D, MaxPooling2D, UpSampling2D, ZeroPadding2D
from keras.layers.normalization import BatchNormalization
from keras.layers.merge import Multiply, Concatenate
from keras.utils import np_utils
from CustomLayers import MaxPoolingWithArgmax2D, MaxUnpooling2D
from keras import regularizers, losses, optimizers

#from keras import backend as K
#K.set_floatx('float16')
os.environ["CUDA_VISIBLE_DEVICES"]="1"


class SegNet(object):
    
    def __init__(self):
        print("Building the SegNet architecture")

    def CreateSegNet(self, input_shape, n_labels, kernel=3, pool_size=(2, 2), output_mode="softmax"):
        # encoder
        inputs = Input(shape=input_shape)

        conv_1 = Convolution2D(64, (kernel, kernel), padding="same", kernel_regularizer = regularizers.l2(0.01))(inputs)
        conv_1 = BatchNormalization()(conv_1)
        conv_1 = Activation("relu")(conv_1)
        conv_2 = Convolution2D(64, (kernel, kernel), padding="same", kernel_regularizer = regularizers.l2(0.01))(conv_1)
        conv_2 = BatchNormalization()(conv_2)
        conv_2 = Activation("relu")(conv_2)

        pool_1, mask_1 = MaxPoolingWithArgmax2D(pool_size)(conv_2)
        
        conv_3 = Convolution2D(128, (kernel, kernel), padding="same", kernel_regularizer = regularizers.l2(0.01))(pool_1)
        conv_3 = BatchNormalization()(conv_3)
        conv_3 = Activation("relu")(conv_3)
        conv_4 = Convolution2D(128, (kernel, kernel), padding="same", kernel_regularizer = regularizers.l2(0.01))(conv_3)
        conv_4 = BatchNormalization()(conv_4)
        conv_4 = Activation("relu")(conv_4)

        pool_2, mask_2 = MaxPoolingWithArgmax2D(pool_size)(conv_4)

        conv_5 = Convolution2D(256, (kernel, kernel), padding="same", kernel_regularizer = regularizers.l2(0.01))(pool_2)
        conv_5 = BatchNormalization()(conv_5)
        conv_5 = Activation("relu")(conv_5)
        conv_6 = Convolution2D(256, (kernel, kernel), padding="same", kernel_regularizer = regularizers.l2(0.01))(conv_5)
        conv_6 = BatchNormalization()(conv_6)
        conv_6 = Activation("relu")(conv_6)
        conv_7 = Convolution2D(256, (kernel, kernel), padding="same", kernel_regularizer = regularizers.l2(0.01))(conv_6)
        conv_7 = BatchNormalization()(conv_7)
        conv_7 = Activation("relu")(conv_7)
        
        pool_3, mask_3 = MaxPoolingWithArgmax2D(pool_size)(conv_7)
        
        conv_8 = Convolution2D(512, (kernel, kernel), padding="same", kernel_regularizer = regularizers.l2(0.01))(pool_3)
        conv_8 = BatchNormalization()(conv_8)
        conv_8 = Activation("relu")(conv_8)
        conv_9 = Convolution2D(512, (kernel, kernel), padding="same", kernel_regularizer = regularizers.l2(0.01))(conv_8)
        conv_9 = BatchNormalization()(conv_9)
        conv_9 = Activation("relu")(conv_9)
        conv_10 = Convolution2D(512, (kernel, kernel), padding="same", kernel_regularizer = regularizers.l2(0.01))(conv_9)
        conv_10 = BatchNormalization()(conv_10)
        conv_10 = Activation("relu")(conv_10)
        
        pool_4, mask_4 = MaxPoolingWithArgmax2D(pool_size)(conv_10)
        
        conv_11 = Convolution2D(512, (kernel, kernel), padding="same", kernel_regularizer = regularizers.l2(0.01))(pool_4)
        conv_11 = BatchNormalization()(conv_11)
        conv_11 = Activation("relu")(conv_11)
        conv_12 = Convolution2D(512, (kernel, kernel), padding="same", kernel_regularizer = regularizers.l2(0.01))(conv_11)
        conv_12 = BatchNormalization()(conv_12)
        conv_12 = Activation("relu")(conv_12)
        conv_13 = Convolution2D(512, (kernel, kernel), padding="same", kernel_regularizer = regularizers.l2(0.01))(conv_12)
        conv_13 = BatchNormalization()(conv_13)
        conv_13 = Activation("relu")(conv_13)
        
        pool_5, mask_5 = MaxPoolingWithArgmax2D(pool_size)(conv_13)
        print("Building encoder done..")
        
        # decoder
        unpool_1 = MaxUnpooling2D(pool_size)([pool_5, mask_5])
        
        conv_14 = Convolution2D(512, (kernel, kernel), padding="same", kernel_regularizer = regularizers.l2(0.01))(unpool_1)
        conv_14 = BatchNormalization()(conv_14)
        conv_14 = Activation("relu")(conv_14)
        conv_15 = Convolution2D(512, (kernel, kernel), padding="same", kernel_regularizer = regularizers.l2(0.01))(conv_14)
        conv_15 = BatchNormalization()(conv_15)
        conv_15 = Activation("relu")(conv_15)
        conv_16 = Convolution2D(512, (kernel, kernel), padding="same", kernel_regularizer = regularizers.l2(0.01))(conv_15)
        conv_16 = BatchNormalization()(conv_16)
        conv_16 = Activation("relu")(conv_16)
        
        unpool_2 = MaxUnpooling2D(pool_size)([conv_16, mask_4])

        conv_17 = Convolution2D(512, (kernel, kernel), padding="same", kernel_regularizer = regularizers.l2(0.01))(unpool_2)
        conv_17 = BatchNormalization()(conv_17)
        conv_17 = Activation("relu")(conv_17)
        conv_18 = Convolution2D(512, (kernel, kernel), padding="same", kernel_regularizer = regularizers.l2(0.01))(conv_17)
        conv_18 = BatchNormalization()(conv_18)
        conv_18 = Activation("relu")(conv_18)
        conv_19 = Convolution2D(256, (kernel, kernel), padding="same", kernel_regularizer = regularizers.l2(0.01))(conv_18)
        conv_19 = BatchNormalization()(conv_19)
        conv_19 = Activation("relu")(conv_19)
        
        unpool_3 = MaxUnpooling2D(pool_size)([conv_19, mask_3])
        
        conv_20 = Convolution2D(256, (kernel, kernel), padding="same", kernel_regularizer = regularizers.l2(0.01))(unpool_3)
        conv_20 = BatchNormalization()(conv_20)
        conv_20 = Activation("relu")(conv_20)
        conv_21 = Convolution2D(256, (kernel, kernel), padding="same", kernel_regularizer = regularizers.l2(0.01))(conv_20)
        conv_21 = BatchNormalization()(conv_21)
        conv_21 = Activation("relu")(conv_21)
        conv_22 = Convolution2D(128, (kernel, kernel), padding="same", kernel_regularizer = regularizers.l2(0.01))(conv_21)
        conv_22 = BatchNormalization()(conv_22)
        conv_22 = Activation("relu")(conv_22)
        
        unpool_4 = MaxUnpooling2D(pool_size)([conv_22, mask_2])
        
        conv_23 = Convolution2D(128, (kernel, kernel), padding="same", kernel_regularizer = regularizers.l2(0.01))(unpool_4)
        conv_23 = BatchNormalization()(conv_23)
        conv_23 = Activation("relu")(conv_23)
        conv_24 = Convolution2D(64, (kernel, kernel), padding="same", kernel_regularizer = regularizers.l2(0.01))(conv_23)
        conv_24 = BatchNormalization()(conv_24)
        conv_24 = Activation("relu")(conv_24)
        
        unpool_5 = MaxUnpooling2D(pool_size)([conv_24, mask_1])
        
        conv_25 = Convolution2D(64, (kernel, kernel), padding="same", kernel_regularizer = regularizers.l2(0.01), name = 'conv25')(unpool_5)
        conv_25 = BatchNormalization()(conv_25)
        conv_25 = Activation("relu", name = 'conv25_activation')(conv_25)
        
        conv_26 = Convolution2D(n_labels, (1, 1), padding="valid", kernel_regularizer = regularizers.l2(0.01))(conv_25)
        conv_26 = BatchNormalization()(conv_26)
        conv_26 = Reshape((input_shape[0] * input_shape[1], n_labels), input_shape=(input_shape[0], input_shape[1], n_labels))(conv_26)
        #conv_26 = Reshape((input_shape[0] * input_shape[1],), input_shape=(input_shape[0], input_shape[1],))(conv_26)

        outputs = Activation(output_mode, name = 'output_activation')(conv_26)
        print("Building decoder done..")
        
        segnet = Model(inputs=inputs, outputs=outputs, name="SegNet")
        
        return segnet
        
class ResNet(object):
    def __init__(self):
        print("Building the ResNet architecture")
        
    def CreateResNet(self, input_shape):
        from resnet50 import ResNet50
        resnet = ResNet50(include_top = False, weights='imagenet', input_shape = input_shape)
        
        return resnet

class EnsembleModelAddOns(object):
    def __init__(self):
        print("Adding Model AddOns")
        
    def MLPClassifier(self):
        # from sklearn.neural_network import MLPClassifier
        clf = MLPClassifier(solver='lbfgs', alpha=1e-5,
                    hidden_layer_sizes=(5, 2), random_state=1)
        return clf
        
    def MultinomialNaiveBayes(self):
        # from sklearn.naive_bayes import MultinomialNB
        clf = MultinomialNB(alpha = 0.001)
        return clf
        
    def NNClassifier(self, input_shape):
        from keras.models import Sequential
        model = Sequential()
        model.add(Dense(64, input_shape = (input_shape,), use_bias=True, activation='relu', kernel_regularizer=regularizers.l1(0.01)))
        model.add(Dropout(0.3))
        model.add(Dense(128, activation='relu', kernel_regularizer=regularizers.l1(0.01)))
        model.add(Dropout(0.3))
        model.add(Dense(256, activation='relu', kernel_regularizer=regularizers.l1(0.01)))
        model.add(Dropout(0.3))
        model.add(Dense(2, activation = 'softmax'))
        return model
        

if __name__ == '__main__':
    model = SegNet()
    segnet = model.CreateSegNet(input_shape = (1944, 2592, 3), n_labels = 12)
    segnet.compile(loss = losses.categorical_crossentropy,
                    optimizer = optimizers.Adam(),
                    metrics = ['accuracy'])
    print(segnet.summary())

