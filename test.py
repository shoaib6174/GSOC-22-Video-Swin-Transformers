from re import S
import tensorflow as tf

from VideoSwinTransformer import SwinTransformer3D



x = tf.random.normal([1000, 8, 32,32, 3])
y = tf.random.uniform(shape=[1000], minval=0, maxval=5, dtype='int64')
model = tf.keras.Sequential()
model.add(SwinTransformer3D())
model.add(tf.keras.layers.Flatten())

model.add(tf.keras.layers.Dense(5))

model.compile(optimizer='sgd', loss='mse')
# This builds the model for the first time:


if __name__ == "__main__":
   model.fit(x, y, batch_size=10, epochs=2)