import tensorflow as tf

def drop_path(inputs, drop_prob,batch_size, is_training):

    if (not is_training) or (drop_prob == 0.):
        return inputs
    # Compute keep_prob
    keep_prob = 1.0 - drop_prob
    # Compute drop_connect tensor
    random_tensor = keep_prob
    shape = (batch_size,)+(1,)*(tf.shape(inputs).shape[0]-1)

    random_tensor += tf.random.uniform(shape, dtype=inputs.dtype)
    binary_tensor = tf.floor(random_tensor)
    output = tf.math.divide(inputs, keep_prob) * binary_tensor
    return output

 
class DropPath(tf.keras.layers.Layer):
    def __init__(self, batch_size, drop_prob=None):
        super().__init__()
        self.drop_prob = drop_prob
        self.batch_size = batch_size

    def call(self, x, training=None):
        output = drop_path(x, self.drop_prob, self.batch_size, training)
        return output
