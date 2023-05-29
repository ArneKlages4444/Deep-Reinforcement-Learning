import tensorflow as tf
from tensorflow import math as tfm
from tensorflow_probability import distributions as tfd


class Policy:

    def __init__(self, policy_network):
        self._policy_network = policy_network

    def distribution_from_policy(self, state):
        mu, sigma = self._policy_network(state)
        return tfd.Normal(mu, sigma)

    def sample_actions_from_policy(self, state):
        distribution = self.distribution_from_policy(state)
        actions = distribution.sample()
        log_probs = self.log_probs_from_distribution(distribution, actions)
        return actions, log_probs

    def log_probs_from_distribution(self, distribution, actions):
        log_probs = distribution.log_prob(actions)
        return tfm.reduce_sum(log_probs, axis=-1, keepdims=True)

    def act_stochastic(self, state, environment):
        actions_prime, log_probs = self.sample_actions_from_policy(tf.convert_to_tensor([state], dtype=tf.float32))
        observation_prime, reward, terminated, truncated, _ = environment.step(actions_prime[0])
        return actions_prime, observation_prime, reward, terminated or truncated, log_probs
