
import tensorflow as tf
from tensorflow.keras.layers import LayerNormalization
import numpy as np

from .WindowAttention3D import WindowAttention3D
from .DropPath import DropPath
# from .Mlp import Mlp
from .mlp2 import mlp_block

from .window_partition import window_partition
from .window_reverse import window_reverse
from .get_window_size import get_window_size


from functools import  lru_cache
@lru_cache()
def compute_mask(D, H, W, window_size, shift_size):

    img_mask = np.zeros((1, D, H, W, 1)) 

 
    cnt = 0

    for d in slice(-window_size[0]), slice(-window_size[0], -shift_size[0]), slice(-shift_size[0],None):
        for h in slice(-window_size[1]), slice(-window_size[1], -shift_size[1]), slice(-shift_size[1],None):
            for w in slice(-window_size[2]), slice(-window_size[2], -shift_size[2]), slice(-shift_size[2],None):
                img_mask[:, d, h, w, :] = cnt
                cnt = cnt + 1
    img_mask = tf.convert_to_tensor(img_mask, dtype="float32")

    # print("basic compute", img_mask.shape, window_size, shift_size, D, H, W)

    mask_windows = window_partition(img_mask, window_size)  # nW, ws[0]*ws[1]*ws[2], 1

    mask_windows = tf.squeeze(mask_windows, axis = -1)  # nW, ws[0]*ws[1]*ws[2] ??
    attn_mask = tf.expand_dims(mask_windows, axis=1) - tf.expand_dims(mask_windows, axis=2)
    

    attn_mask = tf.cast(attn_mask, dtype="float64")

    attn_mask = tf.where(attn_mask != 0, -100.0, attn_mask)
    attn_mask = tf.where(attn_mask == 0, 0.0 , attn_mask)

    return attn_mask


class SwinTransformerBlock3D(tf.keras.Model):
    """ Swin Transformer Block.
    Args:
        dim (int): Number of input channels.
        num_heads (int): Number of attention heads.
        window_size (tuple[int]): Window size.
        shift_size (tuple[int]): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, compute_mask_info, num_heads, window_size=(2,7,7), shift_size=(0,0,0),
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=tf.keras.activations.gelu, norm_layer=LayerNormalization, use_checkpoint=False):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        self.use_checkpoint=use_checkpoint
        self.compute_mask_info = compute_mask_info
        

        # delete this
        self.drop_path_val = drop_path

        assert 0 <= self.shift_size[0] < self.window_size[0], "shift_size must in 0-window_size"
        assert 0 <= self.shift_size[1] < self.window_size[1], "shift_size must in 0-window_size"
        assert 0 <= self.shift_size[2] < self.window_size[2], "shift_size must in 0-window_size"

        self.norm1 = norm_layer(axis=-1, epsilon=1e-5)
        self.attn = WindowAttention3D(
            dim, window_size=self.window_size, num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        # compute mask
        B, C, D, H, W = self.compute_mask_info["shape_of_input"]
        D = 4
        mask_window_size, mask_shift_size = get_window_size((D,H,W), self.compute_mask_info["window_size"], self.compute_mask_info["shift_size"])  #### change
         
        Dp = int(tf.math.ceil(D/ mask_window_size[0])) * mask_window_size[0]
        Hp = int(tf.math.ceil(H / mask_window_size[1])) * mask_window_size[1]
        Wp = int(tf.math.ceil(W / mask_window_size[2])) * mask_window_size[2]

        self.attn_mask = compute_mask(Dp, Hp, Wp, mask_window_size, mask_shift_size)
        print("compute mask parameters", (Dp, Hp, Wp, mask_window_size, mask_shift_size))
        print("attn_mask", self.attn_mask.shape)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else tf.identity
        self.norm2 = norm_layer(epsilon=1e-5)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = mlp_block(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward_part1(self, x):
        
        B, D, H, W, C = tf.shape(x)[0], tf.shape(x)[1], tf.shape(x)[2] , tf.shape(x)[3] , tf.shape(x)[4] 
        
        b, c, d, h ,w = self.compute_mask_info['shape_of_input']
        d = 4       #### change
        

        window_size, shift_size = get_window_size((d, h, w), self.window_size, self.shift_size)
        x = self.norm1(x)

        pad_l = pad_t = pad_d0 = 0
 
        pad_d1 = (window_size[0] - d % window_size[0]) % window_size[0]
        pad_b = (window_size[1] - h % window_size[1]) % window_size[1]
        pad_r = (window_size[2] - w % window_size[2]) % window_size[2]

  

        paddings = [[0,0] , [pad_d0, pad_d1] , [pad_t, pad_b] , [pad_l, pad_r], [0, 0] ]


        x = tf.pad(x, paddings)
        
        
        _, Dp, Hp, Wp, _ = tf.shape(x)[0], tf.shape(x)[1], tf.shape(x)[2] , tf.shape(x)[3] , tf.shape(x)[4] 



        # cyclic shift
        # print("cyclic shift", self.shift_size)
        if any(i > 0 for i in self.shift_size):
            shifted_x = tf.roll(x, shift=[-self.shift_size[0], -self.shift_size[1], -self.shift_size[2]], axis=[1, 2, 3]) #?
            attn_mask = self.attn_mask
        else:
            shifted_x = x
            attn_mask = None
        # print("after cyclic sifted_x", shifted_x.shape)

        # partition windows
        x_windows = window_partition(shifted_x, window_size)  # B*nW, Wd*Wh*Ww, C
        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, mask=attn_mask)  # B*nW, Wd*Wh*Ww, C
        # merge windows
        attn_windows = tf.reshape( attn_windows ,  [-1, *(window_size+(C,))] )
        # print( "attn_windows",attn_windows.shape)
        shifted_x = window_reverse(attn_windows, window_size, B, Dp, Hp, Wp)  # B D' H' W' C
        # reverse cyclic shift
        # print("sifted_x", shifted_x.shape)
        if any(i > 0 for i in self.shift_size):
            x = tf.roll(shifted_x, shift=[self.shift_size[0], self.shift_size[1], self.shift_size[2]], axis=[1, 2, 3]) #?
        else:
            x = shifted_x

        if pad_d1 >0 or pad_r > 0 or pad_b > 0:
            x = x[:, :D, :H, :W, :]
        return x

    def forward_part2(self, x):
        # print("drop path part 2", x.shape)
        # print("forward_part2")
        return self.drop_path(self.mlp(self.norm2(x)))

    def call(self, x):
        """ Forward function.
        Args:
            x: Input feature, tensor size (B, D, H, W, C).
            mask_matrix: Attention mask for cyclic shift.
        """

        # print(self.name, self.drop_path_val, x.shape)


        shortcut = x
        x = self.forward_part1(x)

        # print("after 1", x.shape)
        # print("call")

        # print(self.drop_path, self.drop_path(x))
        x = shortcut + self.drop_path(x)



        x = x + self.forward_part2(x)
        

        return x