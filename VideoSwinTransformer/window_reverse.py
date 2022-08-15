import tensorflow as tf

def window_reverse(windows, window_size, B, D , H, W):
    shape=[ B, D // window_size[0], H // window_size[1], W // window_size[2], window_size[0], window_size[1], window_size[2], windows.shape[-1]]
    # print("reverse input", windows.shape, window_size)
    # print(B, D,H,W)
    # print("\n", shape)
    x = tf.reshape(windows,shape )
    # print("1 reverse", x.shape )
    
    x = tf.transpose(x, perm=[0, 1, 4, 2, 5, 3, 6, 7])
    # print("2 reverse", x.shape )

    x = tf.reshape(x, shape=[B, D, H, W, windows.shape[-1]])
    # print("3 reverse", x.shape )
    return x