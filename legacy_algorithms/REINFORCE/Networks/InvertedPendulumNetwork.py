import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense


def create_policy_network(learning_rate, state_dim=4, action_dim=1):
    inputs = keras.Input(shape=state_dim)
    x = Dense(256, activation=tf.nn.relu)(inputs)
    x = Dense(256, activation=tf.nn.relu)(x)
    mu = Dense(action_dim, activation=None)(x)  # TODO: better activation sigmoid?
    # TODO: better activation, sigmoid or tf.clip_by_value(sigma, 1e-6, 1)?
    sigma = Dense(action_dim, activation=tf.nn.softplus)(x)
    model = keras.Model(inputs=inputs, outputs=(mu, sigma))
    model.compile(optimizer=Adam(learning_rate=learning_rate))
    return model
