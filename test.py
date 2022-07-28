from re import S
import tensorflow as tf

from VideoSwinTransformer import SwinTransformer3D



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

   x = tf.random.normal([2, 8, 32,32, 3])
   swin = SwinTransformer3D()
   output = swin(x)
   print(output.shape)

   print(swin.projection.layers)
   print(swin.layers[-1])
   # print("basic",swin.layers3D[0].blocks.layers[0].layers)
   for i in range(len(swin.weights)):
  
      print(swin.weights[i].name)
