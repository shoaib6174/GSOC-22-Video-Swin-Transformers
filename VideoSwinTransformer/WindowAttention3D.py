import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout
import numpy as np

class WindowAttention3D(tf.keras.layers.Layer):
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5


        self.qkv = Dense(dim * 3, use_bias=qkv_bias)
        self.attn_drop = Dropout(attn_drop)
        self.proj = Dense(dim)
        self.proj_drop = Dropout(proj_drop)


    def build(self, shape_of_input):
        self.relative_position_bias_table = self.add_weight(shape=((2 * self.window_size[0] - 1) * (2 * self.window_size[1] - 1)* (2 * self.window_size[2] - 1), self.num_heads),
                                                            initializer=tf.initializers.Zeros(), trainable=True, name="relative_position_bias_table") # 2*Wd-1 * 2*Wh-1 * 2*Ww-1, nH

        coords_d = np.arange(self.window_size[0])
        coords_h = np.arange(self.window_size[1])
        coords_w = np.arange(self.window_size[2])
        coords = np.stack(np.meshgrid(coords_d, coords_h, coords_w, indexing='ij')) # 3, Wd, Wh, Ww
        coords_flatten = coords.reshape(3, -1)
        relative_coords = coords_flatten[:, :,
                                         None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.transpose([1, 2, 0])
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 2] += self.window_size[2] - 1

        relative_coords[:, :, 0] *= (2 * self.window_size[1] - 1) * (2 * self.window_size[2] - 1)
        relative_coords[:, :, 1] *= (2 * self.window_size[2] - 1)
        relative_position_index = relative_coords.sum(-1).astype(np.int64)
        

        self.relative_position_index = tf.Variable(initial_value=tf.convert_to_tensor(
            relative_position_index), trainable=False)
        self.built = True


    def call(self, x, mask=None):
        B_, N, C = tf.shape(x)[0], tf.shape(x)[1], tf.shape(x)[2]
        qkv = tf.transpose(tf.reshape(self.qkv(
            x), shape=[-1, N, 3, self.num_heads, C // self.num_heads]), perm=[2, 0, 3, 1, 4])
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = (q @ tf.transpose(k, perm=[0, 1, 3, 2]))
        index = tf.reshape(self.relative_position_index[:N, :N], shape=[-1])
        relative_position_bias = tf.gather(self.relative_position_bias_table, index)
        relative_position_bias = tf.reshape(relative_position_bias, shape=[N, N, -1])
        relative_position_bias = tf.transpose(relative_position_bias, perm=[2, 0, 1])
        attn = attn + tf.expand_dims(relative_position_bias, axis=0)

        if mask is not None:

            nW = tf.shape(mask)[0]  # tf.shape(mask)[0]
            attn = tf.reshape(attn, shape=[-1, nW, self.num_heads, N, N]) + tf.cast(
                tf.expand_dims(tf.expand_dims(mask, axis=1), axis=0), attn.dtype)

            attn = tf.reshape(attn, shape=[-1, self.num_heads, N, N])
            attn = tf.nn.softmax(attn, axis=-1)
        else:
            attn = tf.nn.softmax(attn, axis=-1)

        attn = self.attn_drop(attn)

        x = tf.transpose((attn @ v), perm=[0, 2, 1, 3])
        x = tf.reshape(x, shape=[-1, N, C])
        x = self.proj(x)
        x = self.proj_drop(x)
        return x