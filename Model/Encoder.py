import tensorflow as tf
from functools import partial


def model_with_vgg_weights(x_size, y_size, num_channels, num_classes):
    
    # Lets retrieve the VGG Graph and modify the graph and reuse the weights
    vgg16 = tf.keras.applications.vgg16.VGG16(weights="imagenet")
    
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
    
    fc1 = vgg16.get_layer('fc1')
    fc2 = vgg16.get_layer('fc2')
    
    # Converting the FC1 and FC2 to Convolutional Layer
    fc_1 = convolutionalize_me(fc1, (7, 7, 512, 4096))  # Filter = 7x7, #filters = 512, Output Filter = 4096
    fc_2 = convolutionalize_me(fc2, (1, 1, 4096, 4096))
    
    # Using these above weights to Recreate the Graph
    input_ = tf.keras.layers.Input(shape=(x_size, y_size, num_channels), name="Input_Layer")
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
    img = fc_1(img)
    img = tf.keras.layers.Dropout(rate=0.5)(img)
    
    img = fc_2(img)
    img = tf.keras.layers.Dropout(rate=0.5)(img)
    
    out = tf.keras.layers.Conv2D(num_classes, 1, padding="valid", activation="relu", use_bias=True,
                                 name="Encoder_Output")(img)
    
    return input_, pool3, pool4, out


def convolutionalize_me(fc, output_dim):
    w, b = fc.get_weights()
    w_reshaped = w.reshape(output_dim)
    
    conv_layer = tf.keras.layers.Conv2D(output_dim[-1], (output_dim[0], output_dim[1]),
                                        padding="valid", activation="relu", weights=[w_reshaped, b])
    return conv_layer


def random_initialized_net(x_size, y_size, num_channels, num_classes):
    input_ = tf.keras.layers.Input(shape=(x_size, y_size, num_channels), name='input_image')
    default_conv = partial(tf.keras.layers.Conv2D, filters=64, kernel_size=3,
                           padding='same', activation='relu', use_bias=True, name='conv')
    img = tf.keras.layers.ZeroPadding2D(padding=100)(input_)

    img = default_conv(64, name='block1_conv1')(img)
    img = default_conv(64, name='block1_conv2')(img)
    img = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), padding='valid', name='block1_pool')(img)

    img = default_conv(128, name='block2_conv1')(img)
    img = default_conv(128, name='block2_conv2')(img)
    img = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), padding='valid', name='block2_pool')(img)

    img = default_conv(256, name='block3_conv1')(img)
    img = default_conv(256, name='block3_conv2')(img)
    img = default_conv(256, name='block3_conv3')(img)
    img = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), padding='valid', name='block3_pool')(img)
    pool_3 = img

    img = default_conv(512, name='block4_conv1')(img)
    img = default_conv(512, name='block4_conv2')(img)
    img = default_conv(512, name='block4_conv3')(img)
    img = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), padding='valid', name='block4_pool')(img)
    pool_4 = img

    img = default_conv(512, name='block5_conv1')(img)
    img = default_conv(512, name='block5_conv2')(img)
    img = default_conv(512, name='block5_conv3')(img)
    img = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), padding='valid', name='block5_pool')(img)

    img = tf.keras.layers.Conv2D(4096, 7, padding='valid', activation='relu', use_bias=True, name='fc_1')(img)
    img = tf.keras.layers.Dropout(0.5)(img)

    img = tf.keras.layers.Conv2D(4096, 1, padding='valid', activation='relu', use_bias=True, name='fc_2')(img)
    img = tf.keras.layers.Dropout(0.5)(img)

    out_ = tf.keras.layers.Conv2D(num_classes, 1, padding='valid', activation='relu', use_bias=True, name='encoder_graph')(img)

    return input_, pool_3, pool_4, out_





