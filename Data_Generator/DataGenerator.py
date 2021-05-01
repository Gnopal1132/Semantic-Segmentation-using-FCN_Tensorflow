#!/usr/bin/env python
# coding: utf-8

# In[4]:


import sys,os
sys.path.append(os.path.abspath(r"E:\Study Material\Python_Machine_AI\Deep Learning_Lessons\Praktisch\Tensorflow\Projects\Semantic Segmentation FCN"))


# In[5]:


import tensorflow as tf
import numpy as np
from tensorflow.keras.applications.vgg16 import preprocess_input
from PreProcessing.preprocess_functions import Read_Image,Read_mask,Augment_me,One_Hot_Encoder


# In[22]:


class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self,config,dataset,shuffle = True,use_aug = False):
        self.config = config
        self.shuffle = shuffle
        self.use_aug = use_aug
        self.dataset = dataset
        self.dataset_len = len(dataset)
        self.indices = np.arange(self.dataset_len)
        self.num_classes = self.config["Network"]["num_classes"]
        self.batchsize = self.config["train"]["batch_size"]
        self.on_epoch_end()  # Triggered once at very beginning
    
    def on_epoch_end(self):  # After Every Epoch
        # Updates index after each epoch 
        if self.shuffle:
            return np.random.shuffle(self.indices)
        
    def __len__(self):
        # Denotes number of Batches per epoch
        return int(np.floor(self.dataset_len/self.batchsize))
    
    # When batch corresponding to a given index is called,the generator executes the __getitem__ method
    def __getitem__(self,index):
        # Generates one batch of data
        
        # Getting the indices for datapoints
        indices = self.indices[index*self.batchsize : (index+1)*self.batchsize]
        
        # Getting List of Id's
        dataset_temp = [self.dataset[k] for k in indices]
        
        # Generate the Data
        X,y = self.__data_generation(dataset_temp)
        return X,y
    
    #This data generation method is most crucial one it produces the batch of data takes as argument list of IDs of Target Batch
    def __data_generation(self,ID):
        
        X_Batch = []
        Y_Batch = []
        for instance in ID:
            Size_x = self.config["Image"]["Size_x"]
            Size_y = self.config["Image"]["Size_y"]
            channels = self.config["Image"]["Size_channel"]
            
            image = Read_Image(instance,Size_x,Size_y)
            mask = Read_mask(instance,Size_x,Size_y)
        
            if self.use_aug:
                image,mask = Augment_me(image,mask)
            
            One_Hot = One_Hot_Encoder(mask,Size_x,Size_y,num_classes = self.num_classes)
            #image = Normalization_Scaling(image)
            
            image = preprocess_input(image)
            X_Batch.append(image)
            Y_Batch.append(One_Hot)
            
        X_Batch = np.asarray(X_Batch,dtype=np.float32)
        Y_Batch = np.asarray(Y_Batch,dtype=np.float32)
        
        return X_Batch,Y_Batch


# In[ ]:




