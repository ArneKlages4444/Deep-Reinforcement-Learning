import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import Dense, Layer

import tensorflow as tf
from tensorflow import math as tfm
from tensorflow_probability import distributions as tfd


class ActorCriticPolicy:
    def __init__(self, state_dim, action_dim):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.network = self.create_actor_critic_network()

    # return combined_actor_critic_model
    def create_actor_critic_network(self):
        pass

    # return prob_current_policy, entropy, value
    def __call__(self, s, a):
        pass

    # return value
    def get_value(self, state):
        pass

    # return actions_prime, log_probs, value
    def act_stochastic(self, state):
        pass

    # return actions
    def act_deterministic(self, state):
        pass


def log_sigma_processing(log_sigma):
    return tfm.exp(log_sigma)


def clip_sigma_processing(sigma):
    return tf.clip_by_value(sigma, 0, tf.float32.max)


def softplus_sigma_processing(sigma):
    return tfm.softplus(sigma)


class GaussianActorCriticPolicy(ActorCriticPolicy):
    def __init__(self, state_dim, action_dim, sigma_processing=log_sigma_processing):
        super().__init__(state_dim, action_dim)
        self.sigma_processing = sigma_processing
        self.network = self.create_actor_critic_network()

    def create_actor_critic_network(self):
        raise NotImplementedError()

    def __call__(self, s, a):
        mu, sigma, value = self.network(s)
        sigma = self.sigma_processing(sigma)
        distribution = tfd.Normal(mu, sigma)
        prob_current_policy = self._log_probs_from_distribution(distribution, a)
        entropy = distribution.entropy()
        return prob_current_policy, entropy, value

    def get_value(self, state):
        _, _, value = self.network(state)
        return value

    def _log_probs_from_distribution(self, distribution, actions):
        log_probs = distribution.log_prob(actions)
        return tfm.reduce_sum(log_probs, axis=-1, keepdims=True)

    def act_stochastic(self, state):
        mu, sigma, value = self.network(state)
        sigma = self.sigma_processing(sigma)
        distribution = tfd.Normal(mu, sigma)
        actions_prime = distribution.sample()
        log_probs = self._log_probs_from_distribution(distribution, actions_prime)
        return actions_prime, log_probs, value

    def act_deterministic(self, state):
        mu, _, _ = self.network(state)
        return mu


class SigmaLayer(Layer):

    # initializer= 'zeros' or 'ones' or 'uniform'
    def __init__(self, dim, initializer='zeros', **kwargs):
        self.dim = dim
        self.initializer_sigma = initializer
        super(SigmaLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.sigma = self.add_weight(name='sigma',
                                     shape=(self.dim,),
                                     initializer=self.initializer_sigma,
                                     trainable=True)
        super(SigmaLayer, self).build(input_shape)

    def call(self, x=None):
        return self.sigma

    def compute_output_shape(self, input_shape):
        return (self.dim,)


class MlpGaussianActorCriticPolicy(GaussianActorCriticPolicy):
    def __init__(self, state_dim, action_dim, sigma_processing=log_sigma_processing, shared_networks=False):
        self.shared_networks = shared_networks
        super().__init__(state_dim, action_dim, sigma_processing)

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
        sigma = Dense(self.action_dim, kernel_initializer="zeros")(x)
        model = keras.Model(inputs=inputs, outputs=(mu, sigma, value))
        return model

    def _creat_network_separate(self):
        inputs = keras.Input(shape=self.state_dim)

        x = Dense(256, activation=tf.nn.relu)(inputs)
        x = Dense(256, activation=tf.nn.relu)(x)
        x = Dense(256, activation=tf.nn.relu)(x)
        mu = Dense(self.action_dim, activation=None)(x)
        sigma = Dense(self.action_dim, kernel_initializer="zeros")(x)

        y = Dense(256, activation=tf.nn.relu)(inputs)
        y = Dense(256, activation=tf.nn.relu)(y)
        y = Dense(256, activation=tf.nn.relu)(y)
        value = Dense(1, activation=None)(y)

        model = keras.Model(inputs=inputs, outputs=(mu, sigma, value))
        return model
