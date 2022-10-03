from tensorflow.keras.layers import Dropout , LayerNormalization
import tensorflow as tf
import numpy as np
from .DropPath import DropPath
from math import ceil
from .PatchMerging import  PatchMerging
from .BasicLayer import BasicLayer
from .PatchEmbed3D import PatchEmbed3D

from tensorflow.keras.layers import  Conv3D

class SwinTransformer3D(tf.keras.Model):
    def __init__(self, shape_of_input = (3 ,32, 224,224), pretrained=None,
                 pretrained2d=True,
                 patch_size=(4,4,4),
                 in_chans=3,
                 embed_dim=96,
                 depths=[2, 2, 6, 2],
                 num_heads=[3, 6, 12, 24],
                 window_size=(2,7,7),
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.2,
                 norm_layer= LayerNormalization,
                 patch_norm=False,
                 frozen_stages=-1,
                 use_checkpoint=False,
                 isTest =False):    ## ****** remove it later
      
        super().__init__()


        self.pretrained = pretrained
        self.pretrained2d = pretrained2d
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.patch_norm = patch_norm
        self.frozen_stages = frozen_stages
        self.window_size = window_size
        self.patch_size = patch_size
        self.mlp_ratio = mlp_ratio
        self.shape_of_input = list(shape_of_input)

        self.isTest = isTest ## remove later

        self.projection = PatchEmbed3D(patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
                            norm_layer= norm_layer if self.patch_norm else None) ## ***** make is patchembed and change the conver function
       

        

        self.pos_drop = Dropout(drop_rate)

         # stochastic depth
        dpr = [x for x in np.linspace(0., drop_path_rate, sum(depths))] # stochastic depth decay rule

        # build layers
        self.shape_of_input[1] = int(self.shape_of_input[1] / 2)
        self.layers3D = []

        for i_layer in range(self.num_layers):
            
            self.shape_of_input[0] = int(embed_dim * 2 ** i_layer)

            if i_layer == 0:
 
                self.shape_of_input[2] , self.shape_of_input[3] = int(shape_of_input[2] // 4) , int(shape_of_input[3] // 4)
                
            else:
                self.shape_of_input[2] = ceil(self.shape_of_input[2] / 2 )
                self.shape_of_input[3] = ceil(self.shape_of_input[3] / 2)

   
            self.layers3D.append(BasicLayer(dim= int(embed_dim * 2 ** i_layer),
                                                shape_of_input = tuple(self.shape_of_input),

                                                depth=depths[i_layer],
                                                num_heads=num_heads[i_layer],
                                                window_size=window_size,
                                                mlp_ratio=self.mlp_ratio,
                                                qkv_bias=qkv_bias, qk_scale=qk_scale,
                                                drop=drop_rate, attn_drop=attn_drop_rate,

                                                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                                                norm_layer=norm_layer,
                                                downsample=PatchMerging if (
                                                    i_layer < self.num_layers - 1) else None,
                                                use_checkpoint=use_checkpoint) 
                            )
        
        self.num_features = int(embed_dim * 2**(self.num_layers-1))

         # add a norm layer for each output
        self.norm = norm_layer(epsilon=1e-5)





    def call(self, x):
        x = self.projection(x)

        layer_out = {}
        
        x = self.pos_drop(x)

        for layer in self.layers3D:
            x = layer(x)

        x = tf.transpose(x, perm=[0, 2,3,4, 1 ])
        x = self.norm(x)
        x = tf.transpose(x, perm=[0, 4, 1, 2,3 ])
        

        if self.isTest:                 # remove later
            return layer_out, x
        else:
            return x

    def build_graph(self):
        x = tf.keras.Input(shape=(3,8,224,224))
        return tf.keras.Model(inputs=[x], outputs=self.call(x))

