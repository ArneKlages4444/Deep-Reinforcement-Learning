from ReplayBuffer import ReplayBuffer
from Policy import Policy
from RollOutWorker import RollOutWorker
import tensorflow as tf
from tensorflow import math as tfm
from tensorflow.keras.optimizers import Adam
import numpy as np
from collections.abc import Iterable


class Agent:

    def __init__(self, environments, actor_network_generator, critic_network_generator, updates_per_episode=80,
                 epsilon=0.2, gae_lambda=0.95, learning_rate=0.0003, gamma=0.99, alpha=0.2, kld_threshold=0.05,
                 normalize_adv=False):
        self._updates_per_epoch = updates_per_episode
        self._epsilon = epsilon
        self._gae_lambda = gae_lambda
        self._gamma = gamma
        self._alpha = alpha
        self._learning_rate = learning_rate
        self._mse = tf.keras.losses.MeanSquaredError()
        self._policy_network = actor_network_generator()
        self._value_network = critic_network_generator()
        self._optimizer_policy = Adam(learning_rate=learning_rate)  # TODO: one optimizer for both?
        self._optimizer_value = Adam(learning_rate=learning_rate)
        self._kld_threshold = kld_threshold
        self._normalize_adv = normalize_adv
        self._policy = Policy(self._policy_network)
        # option for multiple workers for future parallelization
        if not isinstance(environments, Iterable):
            environments = [environments]
        self._workers = [RollOutWorker(self._policy, self._value_network, env, self._gamma, self._gae_lambda)
                         for env in environments]

    @tf.function
    def learn(self, data_set):
        kld, actor_loss, critic_loss = 0.0, 0.0, 0.0
        kld_next, actor_loss_next = 0.0, 0.0
        i = 0.0
        for s, a, _, r_sum, adv, prob_old_policy in data_set:
            early_stopping, kld_next, actor_loss_next = self.train_step_actor(s, a, adv, prob_old_policy)
            if early_stopping:
                break
            kld += kld_next
            actor_loss += actor_loss_next
            critic_loss += self.train_step_critic(s, r_sum)
            i += 1
        return kld / i, actor_loss / i, critic_loss / i, i, kld_next

    # Alternative that does not terminate the training of the value network if KLD is too high
    @tf.function
    def learn2(self, data_set):
        kld, actor_loss, critic_loss = 0.0, 0.0, 0.0
        kld_next, actor_loss_next = 0.0, 0.0
        i = 0.0
        j = 0.0
        for s, a, _, r_sum, adv, prob_old_policy in data_set:
            early_stopping, kld_next, actor_loss_next = self.train_step_actor(s, a, adv, prob_old_policy)
            if early_stopping:
                break
            kld += kld_next
            actor_loss += actor_loss_next
            i += 1
        for s, a, _, r_sum, adv, prob_old_policy in data_set:
            critic_loss += self.train_step_critic(s, r_sum)
            j += 1
        return kld / i, actor_loss / i, critic_loss / j, i, kld_next

    @tf.function
    def train_step_actor(self, s, a, adv, prob_old_policy):
        early_stopping = False
        loss = 0.0
        if self._normalize_adv:
            adv = (adv - tfm.reduce_mean(adv)) / (tfm.reduce_std(adv) + 1e-8)
        with tf.GradientTape() as tape:
            distribution = self._policy.distribution_from_policy(s)
            prob_current_policy = self._policy.log_probs_from_distribution(distribution, a)
            log_ratio = prob_current_policy - prob_old_policy
            kld = tf.math.reduce_mean((tf.math.exp(log_ratio) - 1) - log_ratio)
            if kld > self._kld_threshold:  # early stoppling if KLD is too high
                early_stopping = True
            else:
                # prob of current policy / prob of old policy (log probs: p/p2 = log(p)-log(p2)
                p = tf.math.exp(prob_current_policy - prob_old_policy)  # exp() to un do log(p)
                clipped_p = tf.clip_by_value(p, 1 - self._epsilon, 1 + self._epsilon)
                policy_loss = -tfm.reduce_mean(tfm.minimum(p * adv, clipped_p * adv))
                # entropy_loss = -tfm.reduce_mean(-prob_current_policy)  # approximate entropy
                entropy_loss = -tfm.reduce_mean(distribution.entropy())
                loss = policy_loss + self._alpha * entropy_loss

                gradients = tape.gradient(loss, self._policy_network.trainable_variables)
                self._optimizer_policy.apply_gradients(zip(gradients, self._policy_network.trainable_variables))
        return early_stopping, kld, loss

    @tf.function
    def train_step_critic(self, s, r_sum):
        with tf.GradientTape() as tape:
            prev_v = self._value_network(s)
            loss = self._mse(r_sum, prev_v)
        gradients = tape.gradient(loss, self._value_network.trainable_variables)
        self._optimizer_value.apply_gradients(zip(gradients, self._value_network.trainable_variables))
        return loss

    def train(self, epochs, batch_size=64, sub_epochs=4, steps_per_trajectory=1024):
        print("start training!")
        rets = []
        replay_buffer = ReplayBuffer(batch_size)
        for e in range(epochs):
            trajectories = [worker.sample_trajectories(steps_per_trajectory) for worker in
                            self._workers]
            ac_ret = 0.0
            ac_dones = 0
            for episodes, ret, dones in trajectories:
                replay_buffer.add_episodes(episodes)
                ac_ret += ret
                ac_dones += dones
            ac_ret = ac_ret / len(self._workers)
            rets.append(ac_ret)
            print("epoch:", e, "return of episode:", ac_ret, "avg 10:", np.average(rets[-10:]), "dones:", ac_dones)
            kld, actor_loss, critic_loss, i, last_kld = self.learn(replay_buffer.get_as_dataset_repeated(sub_epochs))
            print(
                f"kld: {kld}, actor_loss: {actor_loss}, critic_loss: {critic_loss}, updates: {i}, last_kld: {last_kld}")
            replay_buffer.clear()
        print("training finished!")
