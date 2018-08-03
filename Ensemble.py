from Model import SegNet, EnsembleModelAddOns
import cv2
import numpy as np
#from Data import MyGenerator
import os
import argparse
import itertools
from keras import losses, optimizers, utils, models
from keras.models import Model

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

    def getLabel(self, path, size, grayscale = True):
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
            
            
def main(args):
    # Create Data Injector 
    print("Building the data generators...")

    data = MyEnsembleGenerator(base_dir = args.data_dir, batch_size = args.batch_size)
    train_image_generator = data.get_image_batch_generator(subset = "training", size = (args.input_shape[0], args.input_shape[1]))
#    val_image_generator = data.get_image_batch_generator(subset = "validation", size = (args.input_shape[0], args.input_shape[1]))
    
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
    
    intermediate_layer_model = Model(inputs=segnet.input, outputs=segnet.get_layer(layer_name).output)
    intermediate_layer_model_output = intermediate_layer_model.predict_generator(generator = train_image_generator, steps = int(18000/args.batch_size), verbose = 1)
    
    print(intermediate_layer_model_output.shape)
    
    # Ensemble AddOns
    ensemble_addon = EnsembleModelAddOns().MLPClassifier()
        
    def get_batch(array, batch_size):
        for i in range(array.shape[0] // batch_size):
            yield a[batch_size*i:batch_size*(i+1)]

    image_batcherator = get_batch(array = intermediate_layer_model_output, batch_size = args.batch_size)
    label_batcherator = data.get_label_batch_generator(subset = "training", size = (args.input_shape[0], args.input_shape[1]))
    
    zipped = zip(image_batcherator, label_batcherator)
    
    for image_batch, label_batch in next(zipped):
        print(image_batch.shape)
        print(label_batch.shape)
        
        ensemble_addon.partial_fit(image_batch, image_batch)
        

if __name__ == "__main__":
    # command line argments
    parser = argparse.ArgumentParser(description="SegNet Mapillary dataset")
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
