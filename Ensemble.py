from Model import SegNet, EnsembleModelAddOns
import cv2
import numpy as np
import os
import argparse
import itertools
from keras import losses, optimizers, utils, models
from keras.models import Model
from keras.callbacks import ModelCheckpoint, Callback

class MyEnsembleGenerator(object):
    def __init__(self, base_dir, batch_size = 64):
        assert base_dir[-1] == '/'
        self.base_dir = base_dir
        self.batch_size = batch_size
        
    def getImage(self, path, size):
        try:
            img = cv2.imread(path, 1)
            if img is not None:
                img = cv2.resize(img, (size[1], size[0]))
                img = img.astype(np.float32)
    #            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #            img = img/255.0
                return img
            else:
                print("Image {} not loaded".format(path))
        except Exception as e:
            print(path, e)
            img = np.zeros((size[0], size[1], 3))
            return img
            
    def get_image_batch_generator(self, subset, size):
        self._image_path = self.base_dir + subset + "/images/all_images/"
#        self._image_path = self.base_dir + subset + "/images/"
        images = sorted(os.listdir(self._image_path))

        zipped = itertools.cycle(images)
        while True:
            X = []
            for _ in range(self.batch_size):
                im = next(zipped)
                X.append(self.getImage(self._image_path+im, size))

            yield np.array(X)

    def getLabel(self, path, size, n_labels = 2, grayscale = True):
        labels = np.zeros((size[0], size[1], n_labels))
        try:
            img = cv2.imread(path, 1)
            if img is not None:
                img = cv2.resize(img, (size[1], size[0]))
                img = img.astype(np.float32)                
                if grayscale:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                img[img != 119.949] = 0
                img[img == 119.949] = 255
#                where = np.where(img == 119.949)
#                img[where] = 255.0
#                for c, label in enumerate(np.unique(img)):
#                    labels[:, :, c] = (img == label)
                labels[:, :, 0] = 255 - img
                labels[:, :, 1] = img

        except Exception as e:
            print (path, e)
            
        labels =  np.reshape(labels, (size[0]*size[1], n_labels))
        return labels           

    def get_label_batch_generator(self, subset, size):
        self._label_path = self.base_dir + subset + "/labels/all_images/"
        images = sorted(os.listdir(self._label_path))

        zipped = itertools.cycle(images)
        while True:
            Y = []
            for _ in range(self.batch_size):
                im = next(zipped)
                Y.append(self.getLabel(self._label_path+im, size))

            yield np.array(Y)
            
    def get_featureLabel_batch_generator(self, feature_space, subset, size):
        # Flatten the features for NN
        feature_space = np.reshape(feature_space, (feature_space.shape[0], feature_space.shape[1]*feature_space.shape[2], feature_space.shape[3]))
        
        self._label_path = self.base_dir + subset + "/labels/all_images/"
        labels = sorted(os.listdir(self._label_path))
        
        #assert feature_space.shape[0] == len(labels)
        print(feature_space.shape, len(labels))
        
        zipped = itertools.cycle(zip(feature_space, labels))
        while True:
            X = []
            Y = []
            for _ in range(self.batch_size):
                feat, label = next(zipped)
                X.append(feat)
                Y.append(self.getLabel(self._label_path+label, size))
            X = np.reshape(np.array(X), (np.array(X).shape[0]*np.array(X).shape[1], np.array(X).shape[2]))
            Y = np.reshape(np.array(Y), (np.array(Y).shape[0]*np.array(Y).shape[1], np.array(Y).shape[2]))

            yield X, Y
            
def save_in_hdf5_file(hdf5_path, data, data_shape):
    import tables
    img_dtype = tables.UInt8Atom()      # dtype in which the images will be saved
    data_shape = (0, data_shape[0], data_shape[1], 64)
    hdf5_file = tables.open_file(hdf5_path+"inference_output.hdf5", mode='w')       # open a hdf5 file and create earrays
    image_storage = hdf5_file.create_earray(hdf5_file.root, img_dtype, shape=data_shape)
    
    for i, datum in enumerate(data):
        hdf5_file[i] = datum

def main(args):
    # Create Data Injector 
    print("Building the data generators...")

    data = MyEnsembleGenerator(base_dir = args.data_dir, batch_size = args.batch_size)
    train_image_generator = data.get_image_batch_generator(subset = "training", size = (args.input_shape[0], args.input_shape[1]))
    val_image_generator = data.get_image_batch_generator(subset = "validation", size = (args.input_shape[0], args.input_shape[1]))
    
    print("Done.")
    
    # Create SegNet 
    model = SegNet()
    segnet = model.CreateSegNet(input_shape = args.input_shape, n_labels = args.n_labels)
        
    # Load weights
    segnet.load_weights('/share/ece592f17/RA/codebase/weigths-improvement-05-0.97_512_BS2.hdf5')
    
    segnet.compile(loss = losses.categorical_crossentropy,
                    optimizer = optimizers.Adam(),
                    metrics = ['accuracy'])
                    
    layer_name = 'conv25_activation'
    
    # Inference
    intermediate_layer_model = Model(inputs=segnet.input, outputs=segnet.get_layer(layer_name).output)
    intermediate_layer_train_model_output = intermediate_layer_model.predict_generator(generator = train_image_generator, steps = int(500/args.batch_size), workers = 0, use_multiprocessing = False, verbose = 1)
    intermediate_layer_val_model_output = intermediate_layer_model.predict_generator(generator = val_image_generator, steps = int(500/args.batch_size), workers = 0, use_multiprocessing = False, verbose = 1)
    
    # Explicit garbage collection :/
    del train_image_generator
    del val_image_generator
    del segnet
    
    train_featureLabel_batcherator = data.get_featureLabel_batch_generator(feature_space = intermediate_layer_train_model_output, subset = "training", size = (args.input_shape[0], args.input_shape[1]))
    
    val_featureLabel_batcherator = data.get_featureLabel_batch_generator(feature_space = intermediate_layer_val_model_output, subset = "validation", size = (args.input_shape[0], args.input_shape[1]))
    
    # Ensemble AddOns
    ensemble_addon = EnsembleModelAddOns().NNClassifier(input_shape = intermediate_layer_train_model_output.shape[-1])
        
    # Create Callbacks for Accuracy and Saving Checkpoints
    class AccuracyHistory(Callback):
        def on_train_begin(self, logs={}):
            self.acc = []

        def on_epoch_end(self, batch, logs={}):
            self.acc.append(logs.get('acc'))

    history = AccuracyHistory()
    filepath = "Ensemble_weigths-improvement-{epoch:02d}-{val_acc:.2f}_"+str(args.input_shape[0])+"_BS"+str(args.batch_size)+".hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    callbacks_list = [checkpoint, history]   
    
    ensemble_addon.compile(optimizer = optimizers.SGD(),
                            loss='categorical_crossentropy',
                            metrics = ['accuracy'])
                            
    
    ensemble_addon.fit_generator(generator = train_featureLabel_batcherator,
                        steps_per_epoch = int(500/args.batch_size),
                        epochs = args.n_epochs,
                        verbose = 1, callbacks = callbacks_list,
                        validation_data = val_featureLabel_batcherator,
                        validation_steps = int(500/args.batch_size))
        
        

if __name__ == "__main__":
    # command line argments
    parser = argparse.ArgumentParser(description="Segmentation on Mapillary dataset")
    '''
    parser.add_argument("--model",
            type=str,
            help="Model Architecture to be used",
            required = True)    
    '''
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
            default = "512 512 3",
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
