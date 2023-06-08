import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import Dense, LSTM


def create_policy_network(state_dim, action_dim):
    inputs = keras.Input(shape=(None,) + state_dim)
    x = LSTM(256, return_sequences=True)(inputs)
    x = LSTM(256)(x)
    x = Dense(256, activation=tf.nn.relu)(x)
    mu = Dense(action_dim, activation=None)(x)
    sigma = Dense(action_dim, activation=tf.nn.softplus)(x)
    model = keras.Model(inputs=inputs, outputs=(mu, sigma))
    return model


def create_value_network(state_dim):
    inputs = keras.Input(shape=(None,) + state_dim)
    x = LSTM(256, return_sequences=True)(inputs)
    x = LSTM(256)(x)
    x = Dense(256, activation=tf.nn.relu)(x)
    out = Dense(1, activation=None)(x)
    model = keras.Model(inputs=inputs, outputs=out)
    return model
