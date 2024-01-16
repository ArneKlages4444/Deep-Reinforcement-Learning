import tensorflow as tf
from tensorflow import math as tfm
from tensorflow_probability import distributions as tfd

from actor_critic_policy import ActorCriticPolicy


def log_sigma_processing(log_sigma):
    return tfm.exp(log_sigma)


def clip_sigma_processing(sigma):
    return tf.clip_by_value(sigma, 0 + 1e-8, tf.float32.max)


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
