from keras import losses, optimizers, utils, models
from keras.callbacks import ModelCheckpoint, Callback
from Data import MapillaryData, MyGenerator
from Model import SegNet, ResNet
import cv2
import numpy as np
import argparse
import os
from unet import unet
from keras import backend as K
from itertools import product
#K.set_floatx('float16')

def weighted_categorical_crossentropy(weights):
    """
    A weighted version of keras.objectives.categorical_crossentropy
    
    Variables:
        weights: numpy array of shape (C,) where C is the number of classes
    
    Usage:
        weights = np.array([0.5,2,10]) # Class one at 0.5, class 2 twice the normal weights, class 3 10x.
        loss = weighted_categorical_crossentropy(weights)
        model.compile(loss=loss,optimizer='adam')
    """
    
    weights = K.variable(weights)
        
    def loss(y_true, y_pred):
        # scale predictions so that the class probas of each sample sum to 1
        #y_pred /= K.sum(y_pred, axis=-1, keepdims=True)
        # clip to prevent NaN's and Inf's
        y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
        # calc
        loss = y_true * K.log(y_pred) * weights
        loss = -K.sum(loss, -1)
        return loss
    
    return loss

def main(args):
    # Create Data Injector 
    print("Building the data generators...")

#    data = MapillaryData(base_dir = args.data_dir, batch_size = args.batch_size)
#    train_generator = data.get_batch("training")
#    val_generator = data.get_batch("validation")


    data = MyGenerator(base_dir = args.data_dir, n_labels = args.n_labels, batch_size = args.batch_size)
    train_generator = data.get_batch_generator(subset = "training", size = (args.input_shape[0], args.input_shape[1]))
    val_generator = data.get_batch_generator(subset = "validation", size = (args.input_shape[0], args.input_shape[1]))
    
    print("Done.")
    
    # Create Callbacks for Accuracy and Saving Checkpoints
    class AccuracyHistory(Callback):
        def on_train_begin(self, logs={}):
            self.acc = []

        def on_epoch_end(self, batch, logs={}):
            self.acc.append(logs.get('acc'))

    history = AccuracyHistory()
    filepath = args.model+"_weigths-improvement-{epoch:02d}-{val_acc:.2f}_"+str(args.input_shape[0])+"_BS"+str(args.batch_size)+".hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    callbacks_list = [checkpoint, history]
    
    # Custom class-weighted cross-entropy loss function to tackle the high class bias
    from functools import partial, update_wrapper
    weights = np.array([1,10])
    custom_loss = weighted_categorical_crossentropy(weights)
    
    if args.model == "segnet":
        # Create SegNet 
        model = SegNet()
    #    input_shape = (args.batch_size, args.input_shape[0], args.input_shape[1], args.input_shape[2])
        segnet = model.CreateSegNet(input_shape = args.input_shape, n_labels = args.n_labels)
        #segnet.load_weights('/share/ece592f17/RA/codebase/weigths-improvement-05-0.97_512_BS2.hdf5')
        segnet.compile(loss = ,
                    optimizer = optimizers.Adam(),
                    metrics = ['accuracy'])
        segnet.fit_generator(generator = train_generator,
                        steps_per_epoch = int(18000/args.batch_size),
                        epochs = args.n_epochs,
                        verbose = 1, callbacks = callbacks_list,
                        validation_data = val_generator,
                        validation_steps = int(2000/args.batch_size))
    elif args.model == "resnet":
        # Create ResNet
        model = ResNet()
        resnet = model.CreateResNet(input_shape = args.input_shape)
        resnet.compile(loss = losses.categorical_crossentropy,
                        optimizer = optimizers.Adam(),
                        metrics = ['accuracy'])
        resnet.fit_generator(generator = train_generator,
                        steps_per_epoch = int(18000/args.batch_size),
                        epochs = args.n_epochs,
                        verbose = 1, callbacks = callbacks_list,
                        validation_data = val_generator,
                        validation_steps = int(2000/args.batch_size))
    elif args.model == "unet":
        # Create UNet
        print("Creating Unet")
        Unet = unet(input_shape = args.input_shape, n_labels = 2)
        Unet.compile(loss = losses.categorical_crossentropy,
                        optimizer = optimizers.Adam(),
                        metrics = ['accuracy'])
        Unet.load_weights('/share/ece592f17/RA/codebase/unet_weigths-improvement-04-0.97_512_BS2.hdf5')                
        Unet.fit_generator(generator = train_generator,
                            steps_per_epoch = int(18000/args.batch_size),
                            epochs = args.n_epochs,
                            verbose = 1, callbacks = callbacks_list,
                            validation_data = val_generator,
                            validation_steps = int(2000/args.batch_size))
    else:
        print("Enter valid model")

if __name__ == "__main__":
    # command line argments
    parser = argparse.ArgumentParser(description="SegNet Mapillary dataset")

    parser.add_argument("--model",
            type=str,
            help="Model Architecture to be used",
            required = True)    
    
    parser.add_argument("--data_dir",
            type=str,
            help="Base Data directory",
            required = True)
            
    parser.add_argument("--batch_size",
            default=64,
            type=int,
            help="Batch Size")

    parser.add_argument("--n_epochs",
            default=10,
            type=int,
            help="Number of Epochs")
            
    parser.add_argument("--n_labels",
            default=2,
            type=int,
            help="Number of Output Labels")
            
    parser.add_argument("--input_shape",
            default="512 512 3",
            type=int,
            nargs="*",
            help="Input Image Shape")
            
    parser.add_argument("--gpu_id",
            default="0",
            type=str,
            help="GPU Id")

    args = parser.parse_args()
    
    os.environ["CUDA_VISIBLE_DEVICES"]= args.gpu_id
    main(args)
