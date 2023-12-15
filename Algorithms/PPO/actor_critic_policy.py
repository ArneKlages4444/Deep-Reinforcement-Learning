import tensorflow as tf
from tensorflow import math as tfm
from tensorflow_probability import distributions as tfd
import tensorflow.keras as keras
from tensorflow.keras.layers import Dense


class ActorCriticPolicy:
    def __init__(self, network_generator, separate_actor_critic=False):
        self.network = self._creat_network(
            network_generator) if separate_actor_critic else self._creat_network_separate(network_generator)

    def _creat_network(self, network_generator):
        net, state_dim, action_dim = network_generator()
        inputs = keras.Input(shape=state_dim)
        x = net(inputs)
        value = Dense(1, activation=None)(x)
        mu = Dense(action_dim, activation=None)(x)
        sigma = Dense(action_dim, activation=tf.nn.softplus)(x)
        model = keras.Model(inputs=inputs, outputs=(mu, sigma, value))
        return model

    def _creat_network_separate(self, network_generator):
        net_actor, state_dim, action_dim = network_generator()
        net_critic, _, _ = network_generator()
        inputs = keras.Input(shape=state_dim)
        x = net_actor(inputs)
        y = net_critic(inputs)
        value = Dense(1, activation=None)(y)
        mu = Dense(action_dim, activation=None)(x)
        sigma = Dense(action_dim, activation=tf.nn.softplus)(x)
        model = keras.Model(inputs=inputs, outputs=(mu, sigma, value))
        return model

    def __call__(self, s, a):
        mu, sigma, value = self.network(s)
        distribution = tfd.Normal(mu, sigma)
        prob_current_policy = self.log_probs_from_distribution(distribution, a)
        entropy = distribution.entropy()
        return prob_current_policy, entropy, value

    def get_value(self, state):
        _, _, value = self.network(state)
        return value

    def log_probs_from_distribution(self, distribution, actions):
        log_probs = distribution.log_prob(actions)
        return tfm.reduce_sum(log_probs, axis=-1, keepdims=True)

    def act_stochastic(self, state):
        mu, sigma, value = self.network(state)
        distribution = tfd.Normal(mu, sigma)
        actions_prime = distribution.sample()
        log_probs = self.log_probs_from_distribution(distribution, actions_prime)
        return actions_prime, log_probs, value

    def act_deterministic(self, state):
        mu, _, _ = self.network(state)
        return mu
