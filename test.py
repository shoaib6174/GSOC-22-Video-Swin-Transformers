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
   input_shape = (2,3, 32,224, 224)
   swin = SwinTransformer3D(shape_of_input = input_shape)
   x = tf.random.normal(input_shape,   dtype="float32")
   # x  = tf.keras.layers.Input(tensor=x)
   x  = tf.keras.layers.Input((3,32,224,224))


   output = swin(x, training= False)
   # print("output shape: ",output.shape)
   # print(swin.get_layer("basic_layer").layers)
   
   print("saving model \n \n","=============================================")


   # swin.save("weights/model")
