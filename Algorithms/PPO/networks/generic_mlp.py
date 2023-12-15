import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import Dense


def create_network(state_dim, action_dim):
    inputs = keras.Input(shape=state_dim)
    x = Dense(256, activation=tf.nn.relu)(inputs)
    x = Dense(256, activation=tf.nn.relu)(x)
    x = Dense(256, activation=tf.nn.relu)(x)
    model = keras.Model(inputs=inputs, outputs=x)
    return model, state_dim, action_dim
