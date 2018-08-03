from keras import losses, optimizers, utils, models
from Model import SegNet
import cv2
import numpy as np
import argparse
import os
import itertools

os.environ["CUDA_VISIBLE_DEVICES"]="0"

class MyTestGenerator(object):
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
            
    def get_batch_generator(self, subset, size):
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
            
def main(args):
    data = MyTestGenerator(base_dir = args.data_dir, batch_size = args.batch_size)
    test_generator = data.get_batch_generator(subset = "validation", size = (args.input_shape[0], args.input_shape[1]))
    
    # Create SegNet 
    model = SegNet()
    #    input_shape = (args.batch_size, args.input_shape[0], args.input_shape[1], args.input_shape[2])
    segnet = model.CreateSegNet(input_shape = args.input_shape, n_labels = args.n_labels)

    # Load weights
    segnet.load_weights('/share/ece592f17/RA/codebase/weigths-improvement-01-0.97.hdf5')
#    segnet.load_weights('/share/ece592f17/RA/codebase/weigths-improvement-01-0.86.hdf5')
    segnet.compile(loss = losses.categorical_crossentropy,
                    optimizer = optimizers.Adam(),
                    metrics = ['accuracy'])
                    
    probabilities = segnet.predict_generator(generator = test_generator, steps = (2000/args.batch_size), verbose = 1)
    print(probabilities)
    print(np.unique(probabilities))
    print(probabilities.shape) # (5000, 262144, 2)
    probabilities = np.reshape(probabilities, (probabilities.shape[0], args.input_shape[0], args.input_shape[1], probabilities.shape[2])).astype(np.float32)
    print(probabilities.shape)
    probabilities = np.argmax(probabilities, axis = -1).astype(np.float32)
    print(probabilities.shape)
    for i, probability_map in enumerate(probabilities):
        _, img = cv2.threshold(probability_map.astype(np.float32), 0.8, 255, cv2.THRESH_BINARY)
        cv2.imwrite(args.data_dir+'output/Output_'+str(i)+'.png', img.astype('float32'))


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

    parser.add_argument("--n_labels",
            default=2,
            type=int,
            help="Number of Output Labels")
            
    parser.add_argument("--input_shape",
            type=int,
            nargs="*",
            help="Input Image Shape")
            
    args = parser.parse_args()

    main(args)
