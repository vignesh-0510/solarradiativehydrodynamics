import tensorflow as tf

def dense_layer(x, output_dims, name=None):
    """A single dense layer followed by LayerNorm and tanh activation."""
    out = tf.keras.layers.Dense(output_dims, kernel_initializer=tf.keras.initializers.GlorotNormal())(x)
    out = tf.keras.layers.LayerNormalization()(out)
    out = tf.keras.layers.Activation('tanh')(out)
    return out

def densenet_block(x, output_dims, layers=4, name='densenet_block'):
    """DenseNet-style block with 4 dense layers and dense connectivity."""
    input_dim = int(x.shape[-1])

    if input_dim != output_dims:
        x = tf.keras.layers.Dense(output_dims)(x)
    
    inputs = [x]

    for i in range(layers):
        
        if len(inputs) > 1:
            x_concat = tf.keras.layers.Concatenate()(inputs)
        else:
            x_concat = inputs[0]
        out = dense_layer(x_concat, output_dims, name=f"{name}_layer{i}" if name else None)
        inputs.append(out)
    
    if len(inputs) > 1:
        return tf.keras.layers.Concatenate(name=f"{name}_output")(inputs)
    else:
        return inputs[0]