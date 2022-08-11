import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.nn import gelu


def mlp_block(in_features, hidden_features=None, act_layer= gelu,  out_features=None, drop=0., name = "mlp"):
    """FFN for a Transformer block."""
    out_features = out_features or in_features
    hidden_features = hidden_features or in_features
    ffn = keras.Sequential(name=name)
    ffn.add( layers.Dense(hidden_features, activation=act_layer))
    ffn.add(layers.Dropout(drop))
    ffn.add( layers.Dense(out_features))
    ffn.add(layers.Dropout(drop))


    return ffn
        
