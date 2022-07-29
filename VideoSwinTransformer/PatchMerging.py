import tensorflow as tf
from keras.layers import Dense, LayerNormalization 

class PatchMerging(tf.keras.layers.Layer):
    def __init__(self, dim, norm_layer=LayerNormalization):
        super().__init__()
        self.dim = dim
        self.reduction = Dense(2 * dim, use_bias=False,  activation=None)
        self.norm = norm_layer(epsilon=1e-5)
        

    def call(self, x):
        B, D, H, W, C = tf.shape(x)[0], tf.shape(x)[1], tf.shape(x)[2] , tf.shape(x)[3] , tf.shape(x)[4] 

        #print(x, x.shape, tf.shape(x))

        # padding
        pad_input = (H % 2 == 1) or (W % 2 == 1)
        if pad_input:
            x = tf.pad(x, [[0,0], [0,0], [0,H%2], [0,W%2], [0,0]])

        x0 = x[:, :, 0::2, 0::2, :]  # B D H/2 W/2 C
        x1 = x[:, :, 1::2, 0::2, :]  # B D H/2 W/2 C
        x2 = x[:, :, 0::2, 1::2, :]  # B D H/2 W/2 C
        x3 = x[:, :, 1::2, 1::2, :]  # B D H/2 W/2 C

        x = tf.concat([x0, x1, x2, x3], axis=-1) # B D H/2 W/2 4*C

        x = self.norm(x)
        x = self.reduction(x)
        
        return x