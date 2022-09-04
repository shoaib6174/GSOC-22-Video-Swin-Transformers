import tensorflow as tf
from tensorflow.keras.layers import  Conv3D

class PatchEmbed3D(tf.keras.Model):
    def __init__(self, patch_size=(2, 4, 4), in_chans=3, embed_dim=96, norm_layer=None ):
        super().__init__(name='projection')
        
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = Conv3D(embed_dim, kernel_size=patch_size,
                           strides=patch_size, name='embed_proj')
        
        if norm_layer is not None:
            self.norm = norm_layer(epsilon=1e-5, name='embed_norm')
        else:
            self.norm = None

    def call(self, x):
        B, C, D, H, W = x.get_shape().as_list()
        x = tf.transpose(x, perm=[0, 2,3,4, 1 ])

        x = self.proj(x)
        x = tf.transpose(x, perm=[0, 4, 1, 2,3 ])

        
        if self.norm is not None:
          print("patch embed", x.shape)
          B, C, D, Wh, Ww = x.shape
          # if B == None:
          #   B = 1
          x  = tf.reshape(x, shape=[B, C, -1]) ### **** change
          x = tf.transpose(x, perm=[0 , 2, 1])   
          x = self.norm(x)
          x   = tf.transpose(x, perm=[0,2,1])

          x = tf.reshape(x, shape=[-1, self.embed_dim, D, Wh, Ww])

        return x