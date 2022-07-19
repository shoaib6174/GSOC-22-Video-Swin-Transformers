import tensorflow as tf
from functools import reduce


def window_partition(x, window_size):
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