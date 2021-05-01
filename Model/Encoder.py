#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf


# In[16]:


def Model_With_VGG_Weights(X_size,Y_size,Num_channels,Num_classes):
    
    # Lets retrieve the VGG Graph and modify the graph and reuse the weights
    vgg16 = tf.keras.applications.vgg16.VGG16(weights = "imagenet")
    
    # Getting the Blocks of VGG
    block1_conv1 = vgg16.get_layer("block1_conv1")
    block1_conv2 = vgg16.get_layer("block1_conv2")
    block1_pool = vgg16.get_layer("block1_pool")
    
    block2_conv1 = vgg16.get_layer("block2_conv1")
    block2_conv2 = vgg16.get_layer("block2_conv2")
    block2_pool = vgg16.get_layer("block2_pool")
    
    block3_conv1 = vgg16.get_layer('block3_conv1')
    block3_conv2 = vgg16.get_layer('block3_conv2')
    block3_conv3 = vgg16.get_layer('block3_conv3')
    block3_pool = vgg16.get_layer('block3_pool')
    
    block4_conv1 = vgg16.get_layer('block4_conv1')
    block4_conv2 = vgg16.get_layer('block4_conv2')
    block4_conv3 = vgg16.get_layer('block4_conv3')
    block4_pool = vgg16.get_layer('block4_pool')

    block5_conv1 = vgg16.get_layer('block5_conv1')
    block5_conv2 = vgg16.get_layer('block5_conv2')
    block5_conv3 = vgg16.get_layer('block5_conv3')
    block5_pool = vgg16.get_layer('block5_pool')
    
    FC1 = vgg16.get_layer('fc1')
    FC2 = vgg16.get_layer('fc2')
    
    # Converting the FC1 and FC2 to Convolutional Layer
    FC_1 = Convolutionalize_me(FC1,(7, 7, 512, 4096))  # Filter = 7x7, #filters = 512, Output Filter = 4096
    FC_2 = Convolutionalize_me(FC2,(1, 1, 4096, 4096))
    
    # Using these above weights to Recreate the Graph
    input_ = tf.keras.layers.Input(shape=(X_size,Y_size,Num_channels),name = "Input_Layer")
    # Adding some zero Padding symmetrically
    img = tf.keras.layers.ZeroPadding2D(100)(input_)
    
    # Block 1
    img = block1_conv1(img)
    img = block1_conv2(img)
    img = block1_pool(img)
    #Block 2
    img = block2_conv1(img)
    img = block2_conv2(img)
    img = block2_pool(img)
    #Block 3
    img = block3_conv1(img)
    img = block3_conv2(img)
    img = block3_conv3(img)
    img = block3_pool(img)
    pool3 = img
    #Block 4
    img = block4_conv1(img)
    img = block4_conv2(img)
    img = block4_conv3(img)
    img = block4_pool(img)
    pool4 = img
    #Block 5
    img = block5_conv1(img)
    img = block5_conv2(img)
    img = block5_conv3(img)
    img = block5_pool(img)
    
    # Fully Connected Layers
    img = FC_1(img)
    img = tf.keras.layers.Dropout(rate = 0.5)(img)
    
    img = FC_2(img)
    img = tf.keras.layers.Dropout(rate = 0.5)(img)
    
    Out = tf.keras.layers.Conv2D(Num_classes,1,padding = "valid",activation = "relu",use_bias = True,name = "Encoder_Output")(img)
    
    return input_,pool3,pool4,Out
    


# In[17]:


def Convolutionalize_me(fc,Output_dim):
    W,B = fc.get_weights()
    W_Reshaped = W.reshape(Output_dim)
    
    conv_layer = tf.keras.layers.Conv2D(Output_dim[-1],(Output_dim[0],Output_dim[1]),padding = "valid",activation = "relu",weights = [W_Reshaped,B])
    return conv_layer


# In[7]:


def Random_Initialized_Net(X_size,Y_size,Num_channels,Num_classes):
    input_ = tf.keras.layers.Input(shape=(X_size, Y_size, Num_channels), name='input_image')

    img = tf.keras.layers.ZeroPadding2D(padding=100)(input_)

 
    img = tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu', use_bias=True, name='block1_conv1')(img)
    img = tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu', use_bias=True, name='block1_conv2')(img)
    img = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), padding='valid', name='block1_pool')(img)
    

    img = tf.keras.layers.Conv2D(128, 3, padding='same', activation='relu', use_bias=True, name='block2_conv1')(img)
    img = tf.keras.layers.Conv2D(128, 3, padding='same', activation='relu', use_bias=True, name='block2_conv2')(img)
    img = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), padding='valid', name='block2_pool')(img)


    img = tf.keras.layers.Conv2D(256, 3, padding='same', activation='relu', use_bias=True, name='block3_conv1')(img)
    img = tf.keras.layers.Conv2D(256, 3, padding='same', activation='relu', use_bias=True, name='block3_conv2')(img)
    img = tf.keras.layers.Conv2D(256, 3, padding='same', activation='relu', use_bias=True, name='block3_conv3')(img)
    img = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), padding='valid', name='block3_pool')(img)
    pool_3 = img


    img = tf.keras.layers.Conv2D(512, 3, padding='same', activation='relu', use_bias=True, name='block4_conv1')(img)
    img = tf.keras.layers.Conv2D(512, 3, padding='same', activation='relu', use_bias=True, name='block4_conv2')(img)
    img = tf.keras.layers.Conv2D(512, 3, padding='same', activation='relu', use_bias=True, name='block4_conv3')(img)
    img = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), padding='valid', name='block4_pool')(img)   
    pool_4 = img



    img = tf.keras.layers.Conv2D(512, 3, padding='same', activation='relu', use_bias=True, name='block5_conv1')(img)
    img = tf.keras.layers.Conv2D(512, 3, padding='same', activation='relu', use_bias=True, name='block5_conv2')(img)
    img = tf.keras.layers.Conv2D(512, 3, padding='same', activation='relu', use_bias=True, name='block5_conv3')(img)
    img = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), padding='valid', name='block5_pool')(img)
    

    img = tf.keras.layers.Conv2D(4096, 7, padding='valid', activation='relu', use_bias=True, name='fc_1')(img)
    img = tf.keras.layers.Dropout(0.5)(img)
    

    img = tf.keras.layers.Conv2D(4096, 1, padding='valid', activation='relu', use_bias=True, name='fc_2')(img)
    img = tf.keras.layers.Dropout(0.5)(img)

    Out_ = tf.keras.layers.Conv2D(Num_classes, 1, padding='valid', activation='relu', use_bias=True, name='encoder_graph')(img)
    

    
    return input_, pool_3, pool_4, Out_


# In[ ]:




