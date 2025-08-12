from scales import scales
import tensorflow as tf
from models.residual_block import residual_block
from models.densenet_block import densenet_block

def model():
    # neural network: input is (y,x,t,z), output is ux, uy, uz, log10(rho), log10(T)
    # all inputs and outputs are in physical units

    input_scale, input_offset, output_scale, output_offset = scales()
    ndim = len(input_scale)
    npar = len(output_scale)

    inputLayer = tf.keras.layers.Input(shape=(ndim,))
    inputScaling = tf.keras.layers.Rescaling(
            scale  = input_scale,
            offset = input_offset,
        )(inputLayer)
    dnn1 = residual_block(inputScaling, 32)
    dnn2 = residual_block(dnn1, 64)
    # dnn3 = residual_block(dnn2, 128)
    dnn4 = densenet_block(dnn2, 128)
    # dnn5 = residual_block(dnn3, 128)
    # dnn6 = residual_block(dnn4, 64)
    dnn7 = residual_block(dnn4, 32)
    dnn8 = tf.keras.layers.Dense(npar, activation='linear')(dnn7)

    outputScaling = tf.keras.layers.Rescaling(
            scale  = output_scale,
            offset = output_offset
        )(dnn8)
    outputLayer = tf.keras.layers.concatenate([outputScaling, inputLayer])

    return tf.keras.Model(inputs=[inputLayer], outputs=[outputLayer])
