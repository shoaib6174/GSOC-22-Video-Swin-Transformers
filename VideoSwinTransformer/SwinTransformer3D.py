from keras.layers import Dropout , LayerNormalization
import tensorflow as tf
import numpy as np
from einops import rearrange
from .DropPath import DropPath

from .PatchMerging import  PatchMerging
from .BasicLayer import BasicLayer
from .PatchEmbed3D import PatchEmbed3D

from keras.layers import  Conv3D

class SwinTransformer3D(tf.keras.Model):
    def __init__(self, pretrained=None,
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
                 use_checkpoint=False):
      
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


        # # split image into non-overlapping patches
        # self.patch_embed = PatchEmbed3D(
        #     patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
        #     norm_layer=LayerNormalization)
        
        self.projection = tf.keras.Sequential(
            [
                Conv3D(
                    embed_dim, kernel_size = patch_size , strides= patch_size , name= "conv_projection"
                )
            ],
            name = "projection"
        )

        if self.patch_norm:
            self.projection.add(norm_layer(epsilon=1e-5))
        

        self.pos_drop = Dropout(drop_rate)

         # stochastic depth
        dpr = [x for x in np.linspace(0., drop_path_rate, sum(depths))] # stochastic depth decay rule

        # build layers
        self.layers3D = [BasicLayer(dim=int(embed_dim * 2 ** i_layer),
                                               
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
                            for i_layer in range(self.num_layers)]
        
        self.num_features = int(embed_dim * 2**(self.num_layers-1))

         # add a norm layer for each output
        self.norm = norm_layer(epsilon=1e-5)

        self._freeze_stages()

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            self.patch_embed.eval()
            self.patch_embed.trainable = False

        if self.frozen_stages >= 1:
            self.pos_drop.eval()
            for i in range(0, self.frozen_stages):
                m = self.layers3D[i]
                m.eval()
                m.trainable = False



    def call(self, x):
        #print("/*/*/*/*/*/*/*/*/*/*/*/*/*/*/start", x.shape)
        # x = self.patch_embed(x)

        x = self.projection(x)
        x = rearrange(x, 'b d h w c -> b c d h w')

        
        # #print("embed", x.shape)

        x = self.pos_drop(x)

        for layer in self.layers3D:
            x = layer(x)
            
        x = rearrange(x, 'n c d h w -> n d h w c') 
        x = self.norm(x)
        x = rearrange(x, 'n d h w c -> n c d h w')

        return x

    def train(self, mode=True):
        """Convert the model into training mode while keep layers freezed."""
        super(SwinTransformer3D, self).train(mode)
        self._freeze_stages()


  ### todo: inflate weight, init weight