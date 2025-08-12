import tensorflow as tf

def residual_block(x, output_dims, name=None):
    input_dim = int(x.shape[-1])

    identity = x
    if input_dim != output_dims:
        identity = tf.keras.layers.Dense(output_dims)(identity)

    out = tf.keras.layers.Dense(output_dims, kernel_initializer=tf.keras.initializers.GlorotNormal())(x)
    out = tf.keras.layers.Activation('tanh')(out)
    out = tf.keras.layers.LayerNormalization()(out)
    
    out = tf.keras.layers.Dense(output_dims, activation='linear', kernel_initializer=tf.keras.initializers.GlorotNormal())(out)
    out = tf.keras.layers.LayerNormalization()(out)

    out = tf.keras.layers.Add()([out, identity])
    out = tf.keras.layers.Activation('tanh')(out)
    return out