import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import Dense

from gaussian_actor_critic_policy import GaussianActorCriticPolicy, log_sigma_processing


class MlpGaussianActorCriticPolicy(GaussianActorCriticPolicy):
    def __init__(self, state_dim, action_dim, sigma_processing=log_sigma_processing, shared_networks=True):
        super().__init__(state_dim, action_dim, sigma_processing)
        self.shared_networks = shared_networks

    def create_actor_critic_network(self):
        if self.shared_networks:
            return self._creat_network()
        else:
            return self._creat_network_separate()

    def _creat_network(self):
        inputs = keras.Input(shape=self.state_dim)
        x = Dense(256, activation=tf.nn.relu)(inputs)
        x = Dense(256, activation=tf.nn.relu)(x)
        x = Dense(256, activation=tf.nn.relu)(x)
        value = Dense(1, activation=None)(x)
        mu = Dense(self.action_dim, activation=None)(x)
        sigma = Dense(self.action_dim, activation=None)(x)
        model = keras.Model(inputs=inputs, outputs=(mu, sigma, value))
        return model

    def _creat_network_separate(self):
        inputs = keras.Input(shape=self.state_dim)

        x = Dense(256, activation=tf.nn.relu)(inputs)
        x = Dense(256, activation=tf.nn.relu)(x)
        x = Dense(256, activation=tf.nn.relu)(x)
        mu = Dense(self.action_dim, activation=None)(x)
        sigma = Dense(self.action_dim, activation=tf.nn.softplus)(x)

        y = Dense(256, activation=tf.nn.relu)(inputs)
        y = Dense(256, activation=tf.nn.relu)(y)
        y = Dense(256, activation=tf.nn.relu)(y)
        value = Dense(1, activation=None)(y)

        model = keras.Model(inputs=inputs, outputs=(mu, sigma, value))
        return model
