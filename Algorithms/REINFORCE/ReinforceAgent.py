from EpisodeBuffer import EpisodeBuffer
import tensorflow as tf
from tensorflow import math as tfm
from tensorflow_probability import distributions as tfd
import numpy as np


class Agent:

    def __init__(self, environment, policy_network_generator, learning_rate=0.0003, gamma=0.99):
        self._environment = environment
        self._gamma = gamma
        self._policy_network = policy_network_generator(learning_rate)

    def learn(self, episode):  # TODO: check dimensions
        with tf.GradientTape() as tape:
            loss = 0
            for state, action, _, g, i in episode:
                prob_of_action = self.log_probs_of_action_in_state_form_policy(state, action)
                loss += -(tfm.pow(self._gamma, i) * g * prob_of_action)  # TODO: tfm.pow(self.gamma, i) necessary?
        gradients = tape.gradient(loss, self._policy_network.trainable_variables)
        self._policy_network.optimizer.apply_gradients(zip(gradients, self._policy_network.trainable_variables))

    def distribution_of_policy_in_state(self, state):
        mu, sigma = self._policy_network(state)
        # TODO: MultivariateNormalDiag(loc=mus, scale_diag=sigmas) better?
        distribution = tfd.Normal(mu, sigma)
        return distribution

    def log_probs_of_action_in_state_form_policy(self, state, action):
        distribution = self.distribution_of_policy_in_state(state)
        log_probs = distribution.log_prob(action)
        return log_probs

    def sample_actions_form_policy(self, state):
        distribution = self.distribution_of_policy_in_state(state)
        actions = distribution.sample()
        return actions

    def act_deterministic(self, state):
        actions_prime, _ = self._policy_network(tf.convert_to_tensor([state], dtype=tf.float32))
        return self._act(actions_prime)

    def act_stochastic(self, state):
        actions_prime = self.sample_actions_form_policy(tf.convert_to_tensor([state], dtype=tf.float32))
        return self._act(actions_prime)

    def _act(self, actions):
        observation_prime, reward, terminated, truncated, _ = self._environment.step(actions[0])
        return actions, observation_prime, reward, terminated or truncated

    def sample_to_episode_buffer(self):
        buffer = EpisodeBuffer(self._gamma)
        observation, _ = self._environment.reset()
        done = 0
        ret = 0
        while not done:
            action, observation_prime, reward, done = self.act_stochastic(observation)
            ret += reward
            buffer.add(observation, action, reward)
            observation = observation_prime
        return buffer, ret

    def train(self, epochs):
        print("start training!")
        rets = []
        for e in range(epochs):
            buffer, ret = self.sample_to_episode_buffer()
            rets.append(ret)
            print("epoch:", e, "return:", ret, "avg return:", np.average(rets[-50:]))
            episode = buffer.get_as_data_set()
            self.learn(episode)
        print("training finished!")
