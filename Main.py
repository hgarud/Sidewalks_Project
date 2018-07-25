from keras import losses, optimizers, utils, models
from keras.callbacks import ModelCheckpoint, Callback
from Data import MapillaryData, MyGenerator
from Model import SegNet
import cv2
import numpy
import argparse
import os
#from keras import backend as K
#K.set_floatx('float16')

os.environ["CUDA_VISIBLE_DEVICES"]="1"

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

    # Create SegNet 
    model = SegNet()
#    input_shape = (args.batch_size, args.input_shape[0], args.input_shape[1], args.input_shape[2])
    segnet = model.CreateSegNet(input_shape = args.input_shape, n_labels = args.n_labels)
#    segnet.load_weights('/share/ece592f17/RA/codebase/weigths-improvement-01-0.97.hdf5')
#    print(segnet.summary())

#    with open('model_5l.json', 'r') as model_file:
#        segnet = models.model_from_json(model_file.read())
    
#    segnet.load_weights('model_5l_weight_ep50.hdf5')
    
    # Create Callbacks for Accuracy and Saving Checkpoints
    class AccuracyHistory(Callback):
        def on_train_begin(self, logs={}):
            self.acc = []

        def on_epoch_end(self, batch, logs={}):
            self.acc.append(logs.get('acc'))

    history = AccuracyHistory()
    filepath = "weigths-improvement-{epoch:02d}-{val_acc:.2f}.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    callbacks_list = [checkpoint, history]
    
    segnet.compile(loss = losses.categorical_crossentropy,
                    optimizer = optimizers.Adam(),
                    metrics = ['accuracy'])
                    
    
                    
    segnet.fit_generator(generator = train_generator,
                            steps_per_epoch = int(18000/args.batch_size),
                            epochs = args.n_epochs,
                            verbose = 1, callbacks = callbacks_list,
                            validation_data = val_generator,
                            validation_steps = int(2000/args.batch_size))
    

if __name__ == "__main__":
    # command line argments
    parser = argparse.ArgumentParser(description="SegNet Mapillary dataset")

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
            default="512,512,3",
            type=int,
            nargs="*",
            help="Input Image Shape")

    args = parser.parse_args()

    main(args)