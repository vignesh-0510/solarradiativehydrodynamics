from scales import scales
import tensorflow as tf
from models.residual_block import residual_block

def model(ndim, npar):
    '''
    input is normalized intensity and z-coordinate in meters
    output is uz, log10(rho), log10(T) along z
    all inputs and outputs are in physical coordinates

    '''
    input_scale, input_offset, output_scale, output_offset = scales()
    inputLayer = tf.keras.layers.Input(shape=(ndim,))
    inputScaling = tf.keras.layers.Rescaling(
        scale  = [1.0, input_scale[3]],
        offset = [0.0, input_offset[3]]
    )(inputLayer)

    dnn1 = residual_block(inputScaling, 8)
    dnn2 = residual_block(dnn1, 16)
    dnn3 = residual_block(dnn2, 32)
    dnn4 = tf.keras.layers.Dense(npar, activation='linear')(dnn3)
    outputLayer = tf.keras.layers.Rescaling(
        # [2:5] is scaling for [uz, log10(rho), log10(T)]
        scale  = output_scale[2:5],
        offset = output_offset[2:5]
    )(dnn4)
    return tf.keras.Model(inputs=[inputLayer], outputs=[outputLayer])