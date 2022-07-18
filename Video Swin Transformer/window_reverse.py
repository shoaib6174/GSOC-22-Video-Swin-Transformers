import tensorflow as tf

def window_reverse_tf(windows, window_size, B, D , H, W):
    x = tf.reshape(windows, shape=[ B, D // window_size[0], H // window_size[1], W // window_size[2], window_size[0], window_size[1], window_size[2], -1])
    x = tf.transpose(x, perm=[0, 1, 4, 2, 5, 3, 6, 7])
    x = tf.reshape(x, shape=[B, D, H, W, -1])
    return x