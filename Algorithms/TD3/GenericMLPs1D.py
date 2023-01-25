import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense, Concatenate


def create_policy_network(learning_rate, state_dim, action_dim):
    inputs = keras.Input(shape=state_dim)
    x = Dense(256, activation=tf.nn.relu)(inputs)
    x = Dense(256, activation=tf.nn.relu)(x)
    x = Dense(256, activation=tf.nn.relu)(x)
    out = Dense(action_dim, activation=tf.nn.tanh)(x)
    model = keras.Model(inputs=inputs, outputs=out)
    model.compile(optimizer=Adam(learning_rate=learning_rate))
    return model


def create_q_network(learning_rate, state_dim, action_dim):
    inputs_s = keras.Input(shape=state_dim)
    inputs_a = keras.Input(shape=action_dim)
    x = Concatenate()([inputs_s, inputs_a])
    x = Dense(256, activation=tf.nn.relu)(x)
    x = Dense(256, activation=tf.nn.relu)(x)
    x = Dense(256, activation=tf.nn.relu)(x)
    out = Dense(1, activation=None)(x)
    model = keras.Model(inputs=(inputs_s, inputs_a), outputs=out)
    model.compile(optimizer=Adam(learning_rate=learning_rate))
    return model
