
import tensorflow as tf
from keras.layers import LayerNormalization


from .WindowAttention3D import WindowAttention3D
from .DropPath import DropPath
from .Mlp import Mlp
from .mlp2 import mlp_block

from .window_partition import window_partition
from .window_reverse import window_reverse
from .get_window_size import get_window_size


class SwinTransformerBlock3D(tf.keras.layers.Layer):
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

    def __init__(self, dim, num_heads, window_size=(2,7,7), shift_size=(0,0,0),
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=tf.keras.activations.gelu, norm_layer=LayerNormalization, use_checkpoint=False):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        self.use_checkpoint=use_checkpoint

        assert 0 <= self.shift_size[0] < self.window_size[0], "shift_size must in 0-window_size"
        assert 0 <= self.shift_size[1] < self.window_size[1], "shift_size must in 0-window_size"
        assert 0 <= self.shift_size[2] < self.window_size[2], "shift_size must in 0-window_size"
        self.norm1 = norm_layer(axis=-1, epsilon=1e-5)
        self.attn = WindowAttention3D(
            dim, window_size=self.window_size, num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        
        # print("drop_path", drop_path)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else tf.identity
        self.norm2 = norm_layer(epsilon=1e-5)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = mlp_block(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward_part1(self, x, mask_matrix):

        # print('forward1')
        B, D, H, W, C = x.shape
        window_size, shift_size = get_window_size((D, H, W), self.window_size, self.shift_size)
        # print(x.shape, self.dim)
        x = self.norm1(x)
        # pad feature maps to multiples of window size
        pad_l = pad_t = pad_d0 = 0
        pad_d1 = (window_size[0] - D % window_size[0]) % window_size[0]
        pad_b = (window_size[1] - H % window_size[1]) % window_size[1]
        pad_r = (window_size[2] - W % window_size[2]) % window_size[2]
        
        paddings = tf.constant([[0,0] , [pad_d0, pad_d1] , [pad_t, pad_b] , [pad_l, pad_r], [0, 0] ])
        x = tf.pad(x, paddings)


        _, Dp, Hp, Wp, _ = x.shape
        # cyclic shift
        if any(i > 0 for i in shift_size):
            shifted_x = tf.roll(x, shift=[-shift_size[0], -shift_size[1], -shift_size[2]], axis=[1, 2, 3]) #?
            attn_mask = mask_matrix
        else:
            shifted_x = x
            attn_mask = None
        # partition windows
        x_windows = window_partition(shifted_x, window_size)  # B*nW, Wd*Wh*Ww, C
        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, mask=attn_mask)  # B*nW, Wd*Wh*Ww, C
        # merge windows
        attn_windows = tf.reshape( attn_windows ,  [-1, *(window_size+(C,))] )
        shifted_x = window_reverse(attn_windows, window_size, B, Dp, Hp, Wp)  # B D' H' W' C
        # reverse cyclic shift
        if any(i > 0 for i in shift_size):
            x = tf.roll(shifted_x, shift=[shift_size[0], shift_size[1], shift_size[2]], axis=[1, 2, 3]) #?
        else:
            x = shifted_x

        if pad_d1 >0 or pad_r > 0 or pad_b > 0:
            x = x[:, :D, :H, :W, :]
        return x

    def forward_part2(self, x):
        # print("forward-2")
        return self.drop_path(self.mlp(self.norm2(x)))

    def call(self, x, mask_matrix):
        """ Forward function.
        Args:
            x: Input feature, tensor size (B, D, H, W, C).
            mask_matrix: Attention mask for cyclic shift.
        """
        # print(x.shape, "swinBlock")
        shortcut = x
        # if self.use_checkpoint:
        #     #x = checkpoint.checkpoint(self.forward_part1, x, mask_matrix)
        #     x = self.forward_part1(x, mask_matrix)
            
        # else:
        x = self.forward_part1(x, mask_matrix)
        x = shortcut + self.drop_path(x)


        # if self.use_checkpoint:
        #     # x = x + checkpoint.checkpoint(self.forward_part2, x)
        #     x = x + self.forward_part2(x)

        # else:
        x = x + self.forward_part2(x)
        
        # print(x.shape, "swin-out")
        return x