
import tensorflow as tf

K = tf.keras.backend

def categorical_cross_entropy():
    def Loss_function(Y_True,Y_pred):
        loss = K.categorical_crossentropy(Y_True, Y_pred, from_logits=False, axis=3)
        return tf.math.reduce_mean(loss)
    return Loss_function



