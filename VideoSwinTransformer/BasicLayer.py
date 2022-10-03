from traceback import print_tb
from scipy.fftpack import shift
import tensorflow as tf
import numpy as np

from tensorflow.keras.layers import LayerNormalization
from tensorflow.python.ops.gen_math_ops import imag

from .SwinTransformerBlock3D import SwinTransformerBlock3D
from .get_window_size import get_window_size
from .window_partition import window_partition

from functools import  lru_cache

def compute_mask(D, H, W, window_size, shift_size):
    img_mask = np.zeros((1, D, H, W, 1)) 

    cnt = 0

    for d in slice(-window_size[0]), slice(-window_size[0], -shift_size[0]), slice(-shift_size[0],None):
        for h in slice(-window_size[1]), slice(-window_size[1], -shift_size[1]), slice(-shift_size[1],None):
            for w in slice(-window_size[2]), slice(-window_size[2], -shift_size[2]), slice(-shift_size[2],None):
                img_mask[:, d, h, w, :] = cnt
                cnt = cnt + 1
    img_mask = tf.convert_to_tensor(img_mask, dtype="float32")

    mask_windows = window_partition(img_mask, window_size)  # nW, ws[0]*ws[1]*ws[2], 1

    mask_windows = tf.squeeze(mask_windows, axis = -1)  # nW, ws[0]*ws[1]*ws[2] ??
    attn_mask = tf.expand_dims(mask_windows, axis=1) - tf.expand_dims(mask_windows, axis=2)
    

    attn_mask = tf.cast(attn_mask, dtype="float64")

    attn_mask = tf.where(attn_mask != 0, -100.0, attn_mask)
    attn_mask = tf.where(attn_mask == 0, 0.0 , attn_mask)
    return attn_mask



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


        C, D, H, W = self.compute_mask_info["shape_of_input"]
        
        mask_window_size, mask_shift_size = get_window_size((D,H,W), self.compute_mask_info["window_size"], self.compute_mask_info["shift_size"])  #### change
         
        Dp = int(tf.math.ceil(D/ mask_window_size[0])) * mask_window_size[0]
        Hp = int(tf.math.ceil(H / mask_window_size[1])) * mask_window_size[1]
        Wp = int(tf.math.ceil(W / mask_window_size[2])) * mask_window_size[2]
        self.attn_mask = compute_mask(Dp, Hp, Wp, mask_window_size, mask_shift_size)

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
            x = blk(x, self.attn_mask)

        if self.downsample is not None:
            x = self.downsample(x)
        x = tf.transpose(x, perm=[0, 4,1,2,3  ])

        return x
