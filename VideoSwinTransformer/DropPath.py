import tensorflow as tf

# def drop_path(inputs, drop_prob, is_training):
#     if (not is_training) or (drop_prob == 0.):
#         return inputs

#     # Compute keep_prob
#     keep_prob = 1.0 - drop_prob
#     # Compute drop_connect tensor
#     random_tensor = keep_prob
#     shape = (tf.shape(inputs)[0],) + (1,) * (tf.rank(inputs) - 1)
#     print(type(shape))
#     random_tensor += tf.random.uniform(shape, dtype=inputs.dtype)
#     binary_tensor = tf.floor(random_tensor)
#     print("--------drop_path",inputs.shape, shape, "random", random_tensor.shape)
#     # print("=====", binary_tensor.shape  )
#     output = tf.math.divide(inputs, keep_prob) * binary_tensor
#     # print("devided")
#     return output


# class DropPath(tf.keras.layers.Layer):
#     def __init__(self, drop_prob=None):
#         super().__init__()
#         self.drop_prob = drop_prob

#     def call(self, x, training=None):
#         # print("DropPath Class call")
#         output = drop_path(x, self.drop_prob, training)
#         # print(output.shape)
#         return output


# import tensorflow as tf
# from tensorflow.keras import layers


# # Referred from: github.com:rwightman/pytorch-image-models.
# class StochasticDepth(layers.Layer):
#     def __init__(self, drop_prop, **kwargs):
#         super().__init__(**kwargs)
#         self.drop_prob = float(drop_prop)

#     def call(self, x, training=False):
#         if training:
#             keep_prob = 1 - self.drop_prob
#             shape = (x.shape[0],)+(1,)*(tf.shape(x).shape[0]-1)
#             random_tensor = keep_prob + tf.random.uniform(shape, 0, 1)
#             random_tensor = tf.floor(random_tensor)
#             return (x / keep_prob) * random_tensor
#         return x

#     def get_config(self):
#         config = super().get_config()
#         config.update({"drop_prob": self.drop_prob})
#         return config

class DropPath(tf.keras.layers.Layer):
    def __init__(self, drop_prob=None):
        super().__init__()
        self.drop_prob = float(drop_prob)

    def call(self, x, training=None):
        if training:
            keep_prob = 1 - self.drop_prob
            shape = (tf.shape(x)[0],) + (1,) * (tf.rank(x) - 1)

            # shape = (x.shape[0],)+(1,)*(tf.shape(x).shape[0]-1)
            # print(" **** before random", keep_prob, x.shape, shape)
            random_tensor = keep_prob + tf.random.uniform(shape, 0, 1, dtype=x.dtype)
            # print("--------drop_path",x.shape, shape, "random", random_tensor.shape, self.drop_prob)
            random_tensor = tf.floor(random_tensor)

            output = (x / keep_prob) * random_tensor
        # print("droppath output:", output.shape)

            return output
        return x


