#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf


# In[3]:


K = tf.keras.backend

def Categorical_Cross_Entropy():
    def Loss_function(Y_True,Y_pred):
        loss = K.categorical_crossentropy(Y_True,Y_pred,from_logits=False,axis = 3)
        return tf.math.reduce_mean(loss)
    return Loss_function


# In[ ]:




