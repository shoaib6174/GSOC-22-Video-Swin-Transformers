from tensorflow.keras.layers import Dense, Dropout
import tensorflow as tf

class Mlp(tf.keras.Model):
    def __init__(self, in_features, hidden_features=None, act_layer= tf.keras.activations.gelu,  out_features=None, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = Dense(hidden_features)
        self.act = act_layer
        print(out_features, "Dense")
        self.fc2 = Dense(out_features)
        self.drop = Dropout(drop)

    def call(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        print("x", x.shape)
        x = self.fc2(x)
        x = self.drop(x)
        return x