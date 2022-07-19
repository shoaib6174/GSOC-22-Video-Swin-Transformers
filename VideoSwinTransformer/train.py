from re import S
import tensorflow as tf

from SwinTransformer3D import SwinTransformer3D


swin = SwinTransformer3D()


if __name__ == "__main__":
    x = tf.random.uniform([10,  8, 32,32, 3])
    swin(x)