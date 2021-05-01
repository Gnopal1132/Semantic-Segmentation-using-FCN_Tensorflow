#!/usr/bin/env python
# coding: utf-8

# In[3]:


import sys,os
sys.path.append(os.path.abspath(r"E:\Study Material\Python_Machine_AI\Deep Learning_Lessons\Praktisch\Tensorflow\Projects\Semantic Segmentation FCN"))
# In[4]:


import tensorflow as tf
from Model import Encoder,Decoder


# In[5]:


def FCN_Model(config):
    X_size = config["Image"]["Size_x"]
    Y_size = config["Image"]["Size_y"]
    Num_channels = config["Image"]["Size_channel"]
    Num_classes = config["Network"]["num_classes"]
    Use_Pretrained_Weights = config["train"]["weight_initialization"]["use_pretrained"]
    Train_Scratch = config["Network"]["train_from_scratch"]
    Graph_path = config["Network"]["graph_path"]
    Decode = config["Network"]["Decoder"]
    
    if Use_Pretrained_Weights:
        json_file = open(Graph_path,"r")
        model_json = json_file.read()
        json_file.close()
        model = tf.keras.models.model_from_json(model_json)
    else:
        if Train_Scratch:
            input_,pool_3,pool_4,Encoder_Out = Encoder.Random_Initialized_Net(X_size,Y_size,Num_channels=Num_channels,Num_classes=Num_classes)
        else:
            input_,pool_3,pool_4,Encoder_Out = Encoder.Model_With_VGG_Weights(X_size,Y_size,Num_channels=Num_channels,Num_classes=Num_classes)
    
    if Decode == "8X":
        decoder_out = Decoder.Decoder_8x(Encoder_out=Encoder_Out,pool3=pool_3,pool4=pool_4,num_class=Num_classes)
    elif Decode == "16X":
        decoder_out = Decoder.Decoder_16x(Encoder_out=Encoder_Out,pool3=pool_3,pool4=pool_4,num_class=Num_classes)
    elif Decode == "32X":
        decoder_out = Decoder.Decoder_32x(Encoder_out=Encoder_Out,pool3=pool_3,pool4=pool_4,num_class=Num_classes)
        
    else:
        raise Exception("Unknown Decoder")
        
    model = tf.keras.Model(inputs = input_,outputs = decoder_out)
    
    print(model.summary())
    return model
        


# In[ ]:




