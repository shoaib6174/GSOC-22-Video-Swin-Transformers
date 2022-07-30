from re import S
import tensorflow as tf
import numpy
from VideoSwinTransformer import SwinTransformer3D

from VideoSwinTransformer import *

# x = tf.random.normal([1000, 8, 32,32, 3])
# y = tf.random.uniform(shape=[1000], minval=0, maxval=5, dtype='int64')
# model = tf.keras.Sequential()
# model.add(SwinTransformer3D())
# model.add(tf.keras.layers.Flatten())

# model.add(tf.keras.layers.Dense(5))

# model.compile(optimizer='sgd', loss='mse')
# # This builds the model for the first time:


if __name__ == "__main__":
   # model.fit(x, y, batch_size=10, epochs=2)
   

   #print(numpy.version)

   x = tf.random.normal([2, 8, 32,32, 3],   dtype="float32")
   # x = tf.keras.Input((8,32,32,3))
   swin = SwinTransformer3D()
   output = swin(x)



   swin.save("model")
