import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import Dense, Layer, Conv2D, MaxPool2D, Flatten, Reshape, Rescaling, Concatenate


def create_network(state_dim, action_dim, feature_size=256):
    inputs = keras.Input(shape=state_dim)
    x = Rescaling(scale=1.0 / 255.0, offset=0)(inputs)
    x = Conv2D(filters=32, kernel_size=8, strides=4, padding="valid", activation=tf.nn.relu)(x)
    x = Conv2D(filters=64, kernel_size=4, strides=2, padding="valid", activation=tf.nn.relu)(x)
    x = Conv2D(filters=64, kernel_size=3, strides=1, padding="valid", activation=tf.nn.relu)(x)
    x = Flatten()(x)
    x = Dense(feature_size)(x)
    model = keras.Model(inputs=inputs, outputs=x)
    return model, state_dim, action_dim
