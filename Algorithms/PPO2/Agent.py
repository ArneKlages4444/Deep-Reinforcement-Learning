from EpisodeBuffer import EpisodeBuffer
import tensorflow as tf
from tensorflow import math as tfm
from tensorflow_probability import distributions as tfd
import numpy as np


class Agent:

    def __init__(self, environment, actor_network_generator, critic_network_generator, updates_per_episode=80,
                 epsilon=0.2, gae_lambda=0.95, learning_rate=0.0003, gamma=0.99, alpha=0.2, kld_threshold=0.05):
        self._updates_per_episode = updates_per_episode
        self._environment = environment
        self._epsilon = epsilon
        self._gae_lambda = gae_lambda
        self._gamma = gamma
        self._alpha = alpha
        self._learning_rate = learning_rate
        self._mse = tf.keras.losses.MeanSquaredError()
        self._policy_network = actor_network_generator(learning_rate)
        self._value_network = critic_network_generator(learning_rate)
        self._kld_threshold = kld_threshold

    # generalized advantage estimate
    def estimate_advantage(self, rewards, values, dones):  # TODO: rework
        advantage = np.zeros_like(rewards, dtype=np.float32)
        for t in range(len(rewards) - 1):
            discount = 1
            a_t = 0
            for k in range(t, len(rewards) - 1):
                a_t += discount * (rewards[k] + self._gamma * values[k + 1] * (1 - dones[k]) - values[k])
                discount *= self._gamma * self._gae_lambda
            advantage[t] = a_t
        return advantage

    def calc_rewards_to_go(self, rewards, values, advantages):
        g = np.zeros_like(rewards, dtype=np.float32)  # TODO: better implementation for discounting (cumsum, tf.scan)
        for t in range(len(rewards)):
            g_sum = 0
            gamma_t = 1
            for k in range(t, len(rewards)):
                g_sum += rewards[k] * gamma_t
                gamma_t *= self._gamma
            g[t] = g_sum
        return g

    def calc_rewards_to_go2(self, rewards, values, advantages):
        return advantages + np.asarray(values)

    @tf.function
    def distribution_form_policy(self, state):
        mu, sigma = self._policy_network(state)
        return tfd.Normal(mu, sigma)

    @tf.function
    def sample_actions_form_policy(self, state):
        distribution = self.distribution_form_policy(state)
        actions = distribution.sample()
        log_probs = self.log_probs_form_distribution(distribution, actions)
        return actions, log_probs

    @tf.function
    def log_probs_form_policy(self, state, actions):
        distribution = self.distribution_form_policy(state)
        return self.log_probs_form_distribution(distribution, actions), distribution.entropy()

    def log_probs_form_distribution(self, distribution, actions):
        log_probs = distribution.log_prob(actions)
        return tfm.reduce_sum(log_probs, axis=-1, keepdims=True)

    def act_deterministic(self, state):
        actions_prime, _ = self._actor(tf.convert_to_tensor([state], dtype=tf.float32))
        return self._act(actions_prime)

    def act_stochastic(self, state):
        actions_prime, log_probs = self.sample_actions_form_policy(tf.convert_to_tensor([state], dtype=tf.float32))
        return self._act(actions_prime) + (log_probs,)

    def _act(self, actions):
        observation_prime, reward, terminated, truncated, _ = self._environment.step(actions[0])
        return actions, observation_prime, reward, terminated or truncated

    def learn(self, episode):
        s, a, _, r_sum, adv, porb_old_policy = episode
        for _ in range(self._updates_per_episode):
            if self.train_step_actor(s, a, adv, porb_old_policy):
                break
        for _ in range(self._updates_per_episode):
            self.train_step_critic(s, r_sum)

    @tf.function
    def train_step_actor(self, s, a, adv, porb_old_policy):
        early_stoppling = False
        with tf.GradientTape() as tape:
            porb_current_policy, entropy = self.log_probs_form_policy(s, a)
            kld = tf.math.reduce_mean(porb_current_policy - porb_old_policy)  # aproximated Kullback Leibler Divergence
            if tfm.abs(kld) > self._kld_threshold:  # early stoppling if KLD is too high
                early_stoppling = True
            else:
                # prob of current policy / prob of old policy (log probs: p/p2 = log(p)-log(p2)
                p = tf.math.exp(porb_current_policy - porb_old_policy)  # exp() to un do log(p)
                clipped_p = tf.clip_by_value(p, 1 - self._epsilon, 1 + self._epsilon)
                policy_loss = -tfm.reduce_mean(tfm.minimum(p * adv, clipped_p * adv))
                # entropy_loss = -tfm.reduce_mean(-porb_current_policy)  # approximate entropy
                entropy_loss = -tfm.reduce_mean(entropy)
                loss = policy_loss + self._alpha * entropy_loss

                gradients = tape.gradient(loss, self._policy_network.trainable_variables)
                self._policy_network.optimizer.apply_gradients(zip(gradients, self._policy_network.trainable_variables))
        return early_stoppling

    @tf.function
    def train_step_critic(self, s, r_sum):
        with tf.GradientTape() as tape:
            prev_v = self._value_network(s)
            loss = self._mse(r_sum, prev_v)
        gradients = tape.gradient(loss, self._value_network.trainable_variables)
        self._value_network.optimizer.apply_gradients(zip(gradients, self._value_network.trainable_variables))

    def sample_to_episode_buffer(self, max_steps_per_episode):
        buffer = EpisodeBuffer(self.estimate_advantage, self.calc_rewards_to_go)
        s, _ = self._environment.reset()
        d = 0
        ret = 0
        i = 0
        while not d and (max_steps_per_episode is None or i < max_steps_per_episode):
            a, s_p, r, d, p = self.act_stochastic(s)
            ret += r
            v = self._value_network(tf.convert_to_tensor([s], dtype=tf.float32))
            buffer.add(s, tf.squeeze(a, 1), [r], tf.squeeze(v, 1), tf.squeeze(p, 1), d)
            s = s_p
            i += 1
        return buffer, ret

    def train(self, epochs, max_steps_per_episode=None):
        print("start training!")
        rets = []
        for e in range(epochs):
            buffer, ret = self.sample_to_episode_buffer(max_steps_per_episode)
            rets.append(ret)
            print("epoch:", e, "return of episode:", ret, "avg 100:", np.average(rets[-100:]))
            episode = buffer.get_episode()
            self.learn(episode)
        print("training finished!")
