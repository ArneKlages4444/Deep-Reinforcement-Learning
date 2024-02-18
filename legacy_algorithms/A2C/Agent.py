from EpisodeBuffer import EpisodeBuffer
import tensorflow as tf
from tensorflow import math as tfm
from tensorflow_probability import distributions as tfd
import numpy as np


class Agent:

    def __init__(self, environment, actor_network_generator, critic_network_generator,
                 gae_lambda=0.95, learning_rate=0.0003, gamma=0.99):
        self._environment = environment
        self._gae_lambda = gae_lambda
        self._gamma = gamma
        self._learning_rate = learning_rate
        self._mse = tf.keras.losses.MeanSquaredError()
        self._policy_network = actor_network_generator(learning_rate)
        self._value_network = critic_network_generator(learning_rate)

    # generalized advantage estimate
    def estimate_advantage(self, rewards, values, dones):  # TODO: rework
        advantage = np.zeros(len(rewards), dtype=np.float32)
        for t in range(len(rewards) - 1):
            discount = 1
            a_t = 0
            for k in range(t, len(rewards) - 1):
                a_t += discount * (rewards[k] + self._gamma * values[k + 1] * (1 - dones[k]) - values[k])
                discount *= self._gamma * self._gae_lambda
            advantage[t] = a_t
        return advantage

    def calc_rewards_to_go(self, rewards):
        g = np.zeros_like(rewards, dtype=np.float32)  # TODO: better implementation for discounting (cumsum, tf.scan)
        for t in range(len(rewards)):
            g_sum = 0
            gamma_t = 1
            for k in range(t, len(rewards)):
                g_sum += rewards[k] * gamma_t
                gamma_t *= self._gamma
            g[t] = g_sum
        return g

    @tf.function
    def distribution_form_policy(self, state):
        mu, sigma = self._policy_network(state)
        return tfd.Normal(mu, sigma)

    @tf.function
    def sample_actions_form_policy(self, state):
        distribution = self.distribution_form_policy(state)
        actions = distribution.sample()
        return actions

    @tf.function
    def log_probs_form_policy(self, state, actions):
        distribution = self.distribution_form_policy(state)
        log_probs = distribution.log_prob(actions)
        log_probs = tfm.reduce_sum(log_probs, axis=-1, keepdims=True)
        return log_probs

    def act_deterministic(self, state):
        actions_prime, _ = self._actor(tf.convert_to_tensor([state], dtype=tf.float32))
        return self._act(actions_prime)

    def act_stochastic(self, state):
        actions_prime = self.sample_actions_form_policy(tf.convert_to_tensor([state], dtype=tf.float32))
        return self._act(actions_prime)

    def _act(self, actions):
        observation_prime, reward, done, what, _ = self._environment.step(actions[0])
        return actions, observation_prime, reward, done

    # @tf.function
    def learn(self, episode, episode_size):
        self.train_step_actor(episode)
        self.train_step_critic(episode, episode_size)

    # @tf.function
    def train_step_actor(self, episode):
        with tf.GradientTape() as tape:
            loss = 0
            for s, a, _, _, adv in episode:
                prob_of_a = self.log_probs_form_policy(s, a)
                loss -= prob_of_a * adv
        gradients = tape.gradient(loss, self._policy_network.trainable_variables)
        self._policy_network.optimizer.apply_gradients(zip(gradients, self._policy_network.trainable_variables))

    # @tf.function
    def train_step_critic(self, episode, episode_size):
        with tf.GradientTape() as tape:
            loss = 0
            for s, _, _, r_sum, _ in episode:
                prev_v = self._value_network(s)
                loss += self._mse(r_sum, prev_v)
            loss = loss / episode_size
        gradients = tape.gradient(loss, self._value_network.trainable_variables)
        self._value_network.optimizer.apply_gradients(zip(gradients, self._value_network.trainable_variables))

    def sample_to_episode_buffer(self):
        buffer = EpisodeBuffer(self.estimate_advantage, self.calc_rewards_to_go)
        s, _ = self._environment.reset()
        d = 0
        ret = 0
        while not d:
            a, s_p, r, d = self.act_stochastic(s)
            ret += r
            v = self._value_network(tf.convert_to_tensor([s], dtype=tf.float32))
            buffer.add(s, a, r, v, d)
            s = s_p
        return buffer, ret

    def train(self, epochs):
        print("start training!")
        rets = []
        for e in range(epochs):
            print("epoch:", e)
            buffer, ret = self.sample_to_episode_buffer()
            rets.append(ret)
            print("return of episode:", ret, "avg 100:", np.average(rets[-100:]))
            episode = buffer.get_as_data_set()
            self.learn(episode, buffer.size())
        print("training finished!")
