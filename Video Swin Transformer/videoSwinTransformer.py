from keras.layers import Dense, Dropout, Conv3D, LayerNormalization
import tensorflow as tf
import numpy as np

from einops import rearrange
from functools import reduce, lru_cache


def get_window_size(x_size, window_size, shift_size=None):
    use_window_size = list(window_size)
    if shift_size is not None:
        use_shift_size = list(shift_size)
    for i in range(len(x_size)):
        if x_size[i] <= window_size[i]:
            use_window_size[i] = x_size[i]
            if shift_size is not None:
                use_shift_size[i] = 0

    if shift_size is None:
        return tuple(use_window_size)
    else:
        return tuple(use_window_size), tuple(use_shift_size)

def window_partition_tf(x, window_size):
    """
    Args:
        x: (B, D, H, W, C)
        window_size (tuple[int]): window size
    Returns:
        windows: (B*num_windows, window_size*window_size, C)
    """
    B, D, H, W, C = x.shape
    print(x.shape, "partition")
    x = tf.reshape(x, [B, D // window_size[0], window_size[0], H // window_size[1], window_size[1], W // window_size[2], window_size[2], C])
    windows = tf.reshape(tf.transpose(x, perm=[0, 1, 3, 5, 2, 4, 6, 7]), [-1, reduce((lambda x, y: x * y), window_size), C])                                    
                                               
    return windows

def window_reverse_tf(windows, window_size, B, D , H, W):
    x = tf.reshape(windows, shape=[ B, D // window_size[0], H // window_size[1], W // window_size[2], window_size[0], window_size[1], window_size[2], -1])
    x = tf.transpose(x, perm=[0, 1, 4, 2, 5, 3, 6, 7])
    x = tf.reshape(x, shape=[B, D, H, W, -1])
    return x

class PatchEmbed3D_tf(tf.keras.layers.Layer):
    def __init__(self, patch_size=(2, 4, 4), in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__(name='patch_embed')

        
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.embed_dim = embed_dim

        
        self.proj = Conv3D(embed_dim, kernel_size=patch_size,
                           strides=patch_size, name='proj', data_format = "channels_first")
        
        if norm_layer is not None:
            self.norm = norm_layer(epsilon=1e-5, name='norm')
        else:
            self.norm = None

    def call(self, x):
        B, C, D, H, W = x.get_shape().as_list()
        ## padding
        print("embed in " , x.shape)
        x = self.proj(x)
        
        if self.norm is not None:
          B, C, D, Wh, Ww = x.get_shape().as_list()
          x = tf.reshape(x, shape=[B, -1, C])
          x = self.norm(x)
          x = tf.reshape(x, shape=[B, C, -1])
          x = tf.reshape(x, shape=[-1, self.embed_dim, D, Wh, Ww])
        print("embed_out",x.get_shape())
        return x

class PatchMerging_tf(tf.keras.layers.Layer):
    def __init__(self, dim, norm_layer=LayerNormalization):
        super().__init__()
        
        self.dim = dim
        self.reduction = Dense(2 * dim, use_bias=False,  activation=None)
        self.norm = norm_layer(epsilon=1e-5)
        

    def call(self, x):
        B, D, H, W, C = x.shape

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



class Mlp_tf(tf.keras.layers.Layer):
    def __init__(self, in_features, hidden_features=None, act_layer= tf.keras.activations.gelu,  out_features=None, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = Dense(hidden_features)
        self.act = act_layer
        self.fc2 = Dense(out_features)
        self.drop = Dropout(drop)

    def call(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

def drop_path(inputs, drop_prob, is_training):
    if (not is_training) or (drop_prob == 0.):
        return inputs

    # Compute keep_prob
    keep_prob = 1.0 - drop_prob

    # Compute drop_connect tensor
    random_tensor = keep_prob
    shape = (tf.shape(inputs)[0],) + (1,) * \
        (len(tf.shape(inputs)) - 1)
    random_tensor += tf.random.uniform(shape, dtype=inputs.dtype)
    binary_tensor = tf.floor(random_tensor)
    output = tf.math.divide(inputs, keep_prob) * binary_tensor
    return output


class DropPath(tf.keras.layers.Layer):
    def __init__(self, drop_prob=None):
        super().__init__()
        self.drop_prob = drop_prob

    def call(self, x, training=None):
        return drop_path(x, self.drop_prob, training)




class WindowAttention3D_tf(tf.keras.layers.Layer):
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


    def build(self, input_shape):
        self.relative_position_bias_table = self.add_weight(shape=((2 * self.window_size[0] - 1) * (2 * self.window_size[1] - 1)* (2 * self.window_size[2] - 1), self.num_heads),
                                                            initializer=tf.initializers.Zeros(), trainable=True) # 2*Wd-1 * 2*Wh-1 * 2*Ww-1, nH

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
        print(x.shape, "attention")
        B_, N, C = x.get_shape().as_list()
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
            nW = mask.get_shape()[0]  # tf.shape(mask)[0]
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



@lru_cache()
def compute_mask(D, H, W, window_size, shift_size, device):
    print(D,H, W, window_size, shift_size)
    img_mask = np.zeros((1, D, H, W, 1))  # 1 Dp Hp Wp 1.  # ? device
    print("compute mask")
    cnt = 0
    for d in slice(-window_size[0]), slice(-window_size[0], -shift_size[0]), slice(-shift_size[0],None):
        for h in slice(-window_size[1]), slice(-window_size[1], -shift_size[1]), slice(-shift_size[1],None):
            for w in slice(-window_size[2]), slice(-window_size[2], -shift_size[2]), slice(-shift_size[2],None):
                img_mask[:, d, h, w, :] = cnt
                cnt += 1
    mask_windows = window_partition_tf(img_mask, window_size)  # nW, ws[0]*ws[1]*ws[2], 1
    print(mask_windows.shape)

    mask_windows = tf.squeeze(mask_windows, axis = -1)  # nW, ws[0]*ws[1]*ws[2] ??
    print(mask_windows.shape)
    attn_mask = tf.expand_dims(mask_windows, axis=1) - tf.expand_dims(mask_windows, axis=2)
    attn_mask = tf.where(attn_mask != 0, -100.0, attn_mask)
    attn_mask = tf.where(attn_mask == 0, 0.0, attn_mask)

    return attn_mask


class BasicLayer_tf(tf.keras.layers.Layer):
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

        # build 
        self.block = []
        self.blocks = [
            SwinTransformerBlock3D_tf(
                dim=dim,
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
            self.downsample = downsample(dim=dim, norm_layer=norm_layer)

    def call(self, x):
        """ Forward function.
        Args:
            x: Input feature, tensor size (B, C, D, H, W).
        """
        # calculate attention mask for SW-MSA
        B, C, D, H, W = x.shape
        window_size, shift_size = get_window_size((D,H,W), self.window_size, self.shift_size)
        x = rearrange(x, 'b c d h w -> b d h w c')
        Dp = int(np.ceil(D / window_size[0])) * window_size[0]
        Hp = int(np.ceil(H / window_size[1])) * window_size[1]
        Wp = int(np.ceil(W / window_size[2])) * window_size[2]
        attn_mask = compute_mask(Dp, Hp, Wp, window_size, shift_size, x.device) #??
        print(x.shape, attn_mask.shape, "befor blc in basic")
        for blk in self.blocks:
            x = blk(x, attn_mask)
        x = tf.reshape(x, [B, D, H, W, -1])

        if self.downsample is not None:
            x = self.downsample(x)
        x = rearrange(x, 'b d h w c -> b c d h w')
        print(x.shape, "basic-out")
        return x



class SwinTransformerBlock3D_tf(tf.keras.layers.Layer):
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
        self.attn = WindowAttention3D_tf(
            dim, window_size=self.window_size, num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        
        
        self.drop_path = DropPath(drop_path) if drop_path > 0. else tf.identity
        self.norm2 = norm_layer(epsilon=1e-5)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp_tf(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward_part1(self, x, mask_matrix):

        print('forward1')
        B, D, H, W, C = x.shape
        window_size, shift_size = get_window_size((D, H, W), self.window_size, self.shift_size)
        print(x.shape, self.dim)
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
        x_windows = window_partition_tf(shifted_x, window_size)  # B*nW, Wd*Wh*Ww, C
        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, mask=attn_mask)  # B*nW, Wd*Wh*Ww, C
        # merge windows
        attn_windows = tf.reshape( attn_windows ,  [-1, *(window_size+(C,))] )
        shifted_x = window_reverse_tf(attn_windows, window_size, B, Dp, Hp, Wp)  # B D' H' W' C
        # reverse cyclic shift
        if any(i > 0 for i in shift_size):
            x = tf.roll(shifted_x, shift=[shift_size[0], shift_size[1], shift_size[2]], axis=[1, 2, 3]) #?
        else:
            x = shifted_x

        if pad_d1 >0 or pad_r > 0 or pad_b > 0:
            x = x[:, :D, :H, :W, :]
        return x

    def forward_part2(self, x):
        print("forward-2")
        return self.drop_path(self.mlp(self.norm2(x)))

    def call(self, x, mask_matrix):
        """ Forward function.
        Args:
            x: Input feature, tensor size (B, D, H, W, C).
            mask_matrix: Attention mask for cyclic shift.
        """
        print(x.shape, "swinBlock")
        shortcut = x
        if self.use_checkpoint:
            #x = checkpoint.checkpoint(self.forward_part1, x, mask_matrix)
            x = self.forward_part1(x, mask_matrix)
            
        else:
            x = self.forward_part1(x, mask_matrix)
        x = shortcut + self.drop_path(x)


        if self.use_checkpoint:
            # x = x + checkpoint.checkpoint(self.forward_part2, x)
            x = x + self.forward_part2(x)

        else:
            x = x + self.forward_part2(x)
        print(x.shape, "swin-out")
        return x



class SwinTransformer3D_tf(tf.keras.Model):
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


        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed3D_tf(
            patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)
        
        
    

        self.pos_drop = Dropout(drop_rate)

         # stochastic depth
        dpr = [x for x in np.linspace(0., drop_path_rate, sum(depths))] # stochastic depth decay rule

        # build layers
        self.layers3D = [BasicLayer_tf(dim=int(embed_dim * 2 ** i_layer),
                                               
                                                depth=depths[i_layer],
                                                num_heads=num_heads[i_layer],
                                                window_size=window_size,
                                                mlp_ratio=self.mlp_ratio,
                                                qkv_bias=qkv_bias, qk_scale=qk_scale,
                                                drop=drop_rate, attn_drop=attn_drop_rate,

                                                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                                                norm_layer=norm_layer,
                                                downsample=PatchMerging_tf if (
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
        x = self.patch_embed(x)

        x = self.pos_drop(x)

        for layer in self.layers3D:

            x = layer(x)
        x = rearrange(x, 'n c d h w -> n d h w c') 
        x = self.norm(x)
        x = rearrange(x, 'n d h w c -> n c d h w')

        return x

    def train(self, mode=True):
        """Convert the model into training mode while keep layers freezed."""
        super(SwinTransformer3D_tf, self).train(mode)
        self._freeze_stages()


  ### todo: inflate weight, init weight