import tensorflow as tf
from models.residual_block import residual_block

def model(z_scale, z_offset, output_scale, output_offset):
    '''
    input is normalized intensity and z-coordinate in meters
    output is uz, log10(rho), log10(T) along z
    all inputs and outputs are in physical coordinates

    '''
    inputLayer = tf.keras.layers.Input(shape=(1,))
    inputScaling = tf.keras.layers.Rescaling(
        scale  = z_scale,
        offset = z_offset
    )(inputLayer)

    dnn1 = residual_block(inputScaling, 8)
    dnn2 = residual_block(dnn1, 16)
    dnn3 = tf.keras.layers.Dense(2, activation='linear')(dnn2)
    outputLayer = tf.keras.layers.Rescaling(
        scale  = output_scale,
        offset = output_offset
    )(dnn3)
    return tf.keras.Model(inputs=[inputLayer], outputs=[outputLayer])