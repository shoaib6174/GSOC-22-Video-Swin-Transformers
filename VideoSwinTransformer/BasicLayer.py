from traceback import print_tb
from scipy.fftpack import shift
import tensorflow as tf
import numpy as np

from tensorflow.keras.layers import LayerNormalization
from tensorflow.python.ops.gen_math_ops import imag

from .SwinTransformerBlock3D import SwinTransformerBlock3D
from .get_window_size import get_window_size
from .window_partition import window_partition




class BasicLayer(tf.keras.Model):
    """ A basic Swin Transformer layer for one stage.
    Args:
        dim (int): Number of feature channels
        depth (int): Depths of this stage.
        num_heads (int): Number of attention head.
        window_size (tuple[int]): Local window size. Default: (1,7,7).
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
    """

    def __init__(self,
                 dim,
                 shape_of_input,
                 depth,
                 num_heads,
                 window_size=(1,7,7),
                 mlp_ratio=4.,
                 qkv_bias=False,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 norm_layer=LayerNormalization,
                 downsample=None,
                 use_checkpoint=False):
        super().__init__()
        self.window_size = window_size
        self.shift_size = tuple(i // 2 for i in window_size)
        self.depth = depth
        self.use_checkpoint = use_checkpoint
        self.shape_of_input = shape_of_input

        self.compute_mask_info = {
            "shape_of_input": self.shape_of_input,
            "window_size": self.window_size, 
            "shift_size": self.shift_size
        }

        # build 
        self.blocks = [
            SwinTransformerBlock3D(
                dim=dim,
                compute_mask_info = self.compute_mask_info,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=(0,0,0) if (i % 2 == 0) else self.shift_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop,
                attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer,
                use_checkpoint=use_checkpoint,
            )
            for i in range(depth)]
        
        self.downsample = downsample
        if self.downsample is not None:
            self.downsample = downsample(dim=dim, norm_layer=norm_layer, shape_of_input = self.compute_mask_info['shape_of_input'])

    def call(self, x):
        """ Forward function.
        Args:
            x: Input feature, tensor size (B, C, D, H, W).
        """


        # calculate attention mask for SW-MSA
        B, C, D, H, W = tf.shape(x)[0], tf.shape(x)[1], tf.shape(x)[2] , tf.shape(x)[3] , tf.shape(x)[4] 

        x = tf.transpose(x, perm=[0, 2,3,4, 1 ])




        for blk in self.blocks:
            x = blk(x)

        x = tf.reshape(x, [B, D, H, W, -1])


        if self.downsample is not None:
            x = self.downsample(x)
        x = tf.transpose(x, perm=[0, 4,1,2,3  ])

        return x
