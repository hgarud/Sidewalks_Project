import numpy as np
import cv2
import os
from keras.preprocessing.image import ImageDataGenerator
import itertools
#from import keras backend as K
#K.set_floatx('float16')

class MapillaryData(object):
    def __init__(self, base_dir, batch_size = 64):
        self.base_dir = base_dir
        self.batch_size = batch_size

    def get_batch(self, subset):
        _image_datagen = ImageDataGenerator(rescale=1)
        _label_datagen = ImageDataGenerator(rescale=1)
        self._image_path = self.base_dir + subset + "/images/"
        self._label_path = self.base_dir + subset + "/labels/"
        self._image_generator = _image_datagen.flow_from_directory(
                directory = self._image_path,
                target_size = (512, 512),
                color_mode = 'rgb',
                class_mode = None,
                batch_size = self.batch_size,
                seed = 1
                )
        self._label_generator = _label_datagen.flow_from_directory(
                directory = self._label_path,
                target_size = (512, 512),
                color_mode = 'grayscale',
                class_mode = None,
                batch_size = self.batch_size,
                seed = 1
                )

        self._generator = zip(self._image_generator, self._label_generator)
        return self._generator
        
class MyGenerator(object):
    def __init__(self, base_dir, n_labels, batch_size = 64):
        assert base_dir[-1] == '/'
        self.base_dir = base_dir
        self.batch_size = batch_size
        self.n_labels = n_labels
        
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
		    
    def getLabel(self, path, size, n_labels, grayscale = True):
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

#            print(img == 119.949)
#            if img == 119.949:
#                img = 255
#            else:
#                img = 0
#            print(np.unique(img))
#            unique_labels = np.unique(img)
#            print("*****************************************", len(unique_labels))
#            assert len(unique_labels) == n_labels
#            labels = np.zeros((size[0], size[1], len(unique_labels)))
    #            for i, unique_label in enumerate(unique_labels):
#                labels[:, :, i] = (img == unique_label).astype(int)

        except Exception as e:
            print (path, e)
            
        labels =  np.reshape(labels, (size[0]*size[1], n_labels))
        return labels
            
    def get_batch_generator(self, subset, size):
        self._image_path = self.base_dir + subset + "/images/all_images/"
        self._label_path = self.base_dir + subset + "/labels/all_images/"
        
        images = sorted(os.listdir(self._image_path))
        labels = sorted(os.listdir(self._label_path))

        assert len(images) == len(labels)

        zipped = itertools.cycle(zip(images,labels))
        while True:
            X = []
            Y = []
            for _ in range(self.batch_size):
                im , seg = next(zipped)
                X.append(self.getImage(self._image_path+im, size))
                Y.append(self.getLabel(path = self._label_path+seg, size = size, n_labels = self.n_labels))

            yield np.array(X) , np.array(Y)
        
if __name__ == '__main__':

    data = MyGenerator(base_dir = '/share/ece592f17/RA/Data/', n_labels = 2, batch_size = 10)
    train_generator = data.get_batch_generator("validation", size = (1944, 2592, 3))
    print("Generator generated :p")
    for x, y in train_generator:
        print(x.shape,y.shape)
    



