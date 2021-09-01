import tensorflow as tf


def decoder_8x(encoder_out, pool3, pool4, num_class):

    # Lets Deconvolutionalize at 16x
    score = tf.keras.layers.Conv2DTranspose(num_class, 4, strides=2, padding="valid")(encoder_out)
    pool4_out = tf.keras.layers.Conv2D(num_class, 1, padding="valid", use_bias=True)(pool4)
    pool4_resize = tf.keras.layers.Cropping2D(cropping=5)(pool4_out)
    score_16x = tf.keras.layers.Add()([score, pool4_resize])
    
    # Deconvolutionalize at 8x
    score = tf.keras.layers.Conv2DTranspose(num_class, 4, strides=2, padding="valid")(score_16x)
    pool3_out = tf.keras.layers.Conv2D(num_class, 1, padding="valid", use_bias=True)(pool3)
    score_pad = tf.keras.layers.ZeroPadding2D(padding=((1, 0), (1, 0)))(score)
    pool3_resize = tf.keras.layers.Cropping2D(cropping=9)(pool3_out)
    score_8x = tf.keras.layers.Add()([score_pad, pool3_resize])
    
    # Resize to image Shape
    upsample = tf.keras.layers.Conv2DTranspose(num_class, 16, strides=8, padding="same")(score_8x)
    upsample = tf.keras.layers.Cropping2D(cropping=28)(upsample)
    
    out = tf.keras.layers.Activation("softmax")(upsample)
    
    return out


def decoder_16x(encoder_out, pool4, num_class):

    #Lets Deconvolutionalize at 16x
    score = tf.keras.layers.Conv2DTranspose(num_class, 4, strides=2, padding="valid")(encoder_out)
    pool4_out = tf.keras.layers.Conv2D(num_class, 1, padding="valid", use_bias=True)(pool4)
    pool4_resize = tf.keras.layers.Cropping2D(cropping=6)(pool4_out)
    score_16x = tf.keras.layers.Add()([score, pool4_resize])
    
    # Resize to image Shape
    upsample = tf.keras.layers.Conv2DTranspose(num_class, 32, strides=16, padding="same")(score_16x)
    
    out = tf.keras.layers.Activation("softmax")(upsample)
    
    return out


def decoder_32x(encoder_out, num_class):
    # Resize to image Shape
    upsample = tf.keras.layers.Conv2DTranspose(num_class, 64, strides=32, padding="same")(encoder_out)
    out = tf.keras.layers.Activation("softmax")(upsample)
    return out


