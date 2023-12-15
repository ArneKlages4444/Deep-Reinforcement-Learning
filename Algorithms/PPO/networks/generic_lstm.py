import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import Dense, LSTM


def create_network(state_dim, action_dim):
    inputs = keras.Input(shape=state_dim)
    x = LSTM(256, return_sequences=True)(inputs)
    x = LSTM(256)(x)
    x = Dense(256, activation=tf.nn.relu)(x)
    model = keras.Model(inputs=inputs, outputs=x)
    return model, state_dim, action_dim
