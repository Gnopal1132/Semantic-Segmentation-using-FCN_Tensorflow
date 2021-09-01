import tensorflow as tf
import numpy as np
from tensorflow.keras.applications.vgg16 import preprocess_input
from PreProcessing.preprocess_functions import read_image, read_mask, augment_me, one_hot_encoder


class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, config, dataset, shuffle=True, use_aug=False, is_train=True):
        self.config = config
        self.shuffle = shuffle
        self.use_aug = use_aug
        self.dataset = dataset
        self.dataset_len = len(dataset)
        self.indices = np.arange(self.dataset_len)
        self.num_classes = self.config["Network"]["num_classes"]
        self.batchsize = self.config["train"]["batch_size"]
        self.is_train = is_train
        self.on_epoch_end()  # Triggered once at very beginning
    
    def on_epoch_end(self):  # After Every Epoch
        # Updates index after each epoch 
        if self.shuffle:
            return np.random.shuffle(self.indices)
        
    def __len__(self):
        # Denotes number of Batches per epoch
        return int(np.floor(self.dataset_len/self.batchsize))
    
    # When batch corresponding to a given index is called,the generator executes the __getitem__ method
    def __getitem__(self, index):
        # Generates one batch of data
        
        # Getting the indices for datapoints
        indices = self.indices[index*self.batchsize:(index+1)*self.batchsize]
        
        # Getting List of Id's
        dataset_temp = [self.dataset[k] for k in indices]

        if self.is_train:
            # Generate the Data
            x, y = self.__data_generation(dataset_temp)
            return x, y
        else:
            return self.__data_generation(dataset_temp)
    
    def __data_generation(self, id):

        if self.is_train:
            x_batch = []
            y_batch = []
            for instance in id:
                size_x = self.config["Image"]["Size_x"]
                size_y = self.config["Image"]["Size_y"]

                image = read_image(instance, size_x, size_y)
                mask = read_mask(instance, size_x, size_y)

                if self.use_aug:
                    image, mask = augment_me(image, mask)

                one_hot = one_hot_encoder(mask, size_x, size_y, num_classes=self.num_classes)
                image = preprocess_input(image)
                x_batch.append(image)
                y_batch.append(one_hot)

            x_batch = np.asarray(x_batch, dtype=np.float32)
            y_batch = np.asarray(y_batch, dtype=np.float32)

            return x_batch, y_batch
        else:
            x_batch = []
            for instance in id:
                size_x = self.config["Image"]["Size_x"]
                size_y = self.config["Image"]["Size_y"]

                image = read_image(instance, size_x, size_y)
                image = preprocess_input(image)
                x_batch.append(image)
            x_batch = np.asarray(x_batch, dtype=np.float32)
            return x_batch
