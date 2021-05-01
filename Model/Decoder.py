#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf


# In[2]:


def Decoder_8x(Encoder_out,pool3,pool4,num_class):
    # Lets Deconvolutionalize at 16x
    score = tf.keras.layers.Conv2DTranspose(num_class,4,strides = 2,padding = "valid")(Encoder_out)
    pool4_out = tf.keras.layers.Conv2D(num_class,1,padding = "valid",use_bias = True)(pool4)
    pool4_resize = tf.keras.layers.Cropping2D(cropping = 5)(pool4_out)
    score_16x = tf.keras.layers.Add()([score,pool4_resize])
    
    # Deconvolutionalize at 8x
    score = tf.keras.layers.Conv2DTranspose(num_class,4,strides = 2,padding = "valid")(score_16x)
    pool3_out = tf.keras.layers.Conv2D(num_class,1,padding = "valid",use_bias = True)(pool3)
    score_pad = tf.keras.layers.ZeroPadding2D(padding = ((1,0),(1,0)))(score)
    pool3_resize = tf.keras.layers.Cropping2D(cropping = 9)(pool3_out)
    score_8x = tf.keras.layers.Add()([score_pad,pool3_resize])
    
    # Resize to image Shape
    Upsample = tf.keras.layers.Conv2DTranspose(num_class,16,strides = 8,padding = "same")(score_8x)
    Upsample = tf.keras.layers.Cropping2D(cropping = 28)(Upsample)
    
    Out = tf.keras.layers.Activation("softmax")(Upsample)
    
    return Out


# In[3]:


def Decoder_16x(Encoder_out,pool4,num_class):
    #Lets Deconvolutionalize at 16x
    score = tf.keras.layers.Conv2DTranspose(num_class,4,strides = 2,padding = "valid")(Encoder_out)
    pool4_out = tf.keras.layers.Conv2D(num_class,1,padding = "valid",use_bias = True)(pool4)
    pool4_resize = tf.keras.layers.Cropping2D(cropping = 6)(pool4_out)
    score_16x = tf.keras.layers.Add()([score,pool4_resize])
    
    # Resize to image Shape
    Upsample = tf.keras.layers.Conv2DTranspose(num_class,32,strides = 16,padding = "same")(score_16x)
    
    Out = tf.keras.layers.Activation("softmax")(Upsample)
    
    return Out


# In[4]:


def Decoder_32x(Encoder_out,num_class):
    # Resize to image Shape
    Upsample = tf.keras.layers.Conv2DTranspose(num_class,64,strides = 32,padding = "same")(Encoder_out)
    
    Out = tf.keras.layers.Activation("softmax")(Upsample)
    
    return Out


# In[ ]:




