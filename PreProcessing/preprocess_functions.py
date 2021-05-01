#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from PIL import Image,ImageStat
import os
from albumentations import ChannelShuffle,Blur,RGBShift,RandomBrightness


# In[6]:


def Read_Image(Instance,size_x,size_y):
    image = Image.open(Instance[0])
    image = image.resize((size_x,size_y),Image.ANTIALIAS)
    image = image.convert('RGB')
    return np.array(image)


# In[3]:


def Read_mask(Instance,size_x,size_y):
    mask = Image.open(Instance[1])
    mask = mask.resize((size_x,size_y),Image.ANTIALIAS)
    mask = np.array(mask,dtype = np.float32)
    mask[mask == 255.] = 0.
    return mask


# In[4]:


def One_Hot_Encoder(mask,Size_x,Size_y,num_classes):
    One_hot_mask = np.zeros((Size_x,Size_y,num_classes))
    for i in range(Size_x):
        for j in range(Size_y):
            pix_val = mask[i,j]
            One_hot_mask[i,j,int(pix_val)] = 1
    return One_hot_mask


# In[3]:


def Augment_me(image,mask):
    X = image
    Y = mask
    
    # Executing with Default Parameters
    Aug = ChannelShuffle(p = 0.5)
    Augmented = Aug(image = X,mask = Y)
    X = Augmented["image"]
    Y = Augmented["mask"]
    
    Aug = Blur(p = 0.5)
    Augmented = Aug(image = X,mask = Y)
    X = Augmented["image"]
    Y = Augmented["mask"]
    
    Aug = RGBShift(p = 0.5)
    Augmented = Aug(image = X,mask = Y)
    X = Augmented["image"]
    Y = Augmented["mask"]
    
    Aug = RandomBrightness(p = 0.5)
    Augmented = Aug(image = X,mask = Y)
    X = Augmented["image"]
    Y = Augmented["mask"]
    
    return X,Y
    


# In[4]:


def Normalize_Scaling(img):
    img = Image.fromarray(img)
    stat = ImageStat.Stat(img)
    img = np.array(img)
    
    img = (img - stat.mean)/stat.stddev
    return img


# In[5]:


def Normalize_Sum_to_1(img):
    img = img.astype("float")
    img /= 255.
    return img


# In[6]:


def Zooming_Image(img,X_Shift,Y_shift):    
    img = img[X_Shift:-X_Shift, Y_shift:-Y_shift, :]
    return img


# In[ ]:




