from re import S
import tensorflow as tf

from VideoSwinTransformer import SwinTransformer3D


swin = SwinTransformer3D()


if __name__ == "__main__":
    x = tf.random.uniform([10,  8, 32,32, 3])
    x  = swin(x)
    print(x.shape)