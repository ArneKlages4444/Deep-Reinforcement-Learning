from ReplayBuffer import ReplayBuffer
from Policy import Policy
from RollOutWorker import RollOutWorker
import tensorflow as tf
from tensorflow import math as tfm
from tensorflow.keras.optimizers import Adam
from collections.abc import Iterable
import datetime


class Agent:

    def __init__(self, environments, actor_network_generator, critic_network_generator,
                 epsilon=0.2, gae_lambda=0.95, learning_rate=0.0003, gamma=0.99, alpha=0.2, kld_threshold=0.05,
                 window_size=None, normalize_adv=False, logs=False):
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
        self._logs = logs
        # option for multiple workers for future parallelization
        if not isinstance(environments, Iterable):
            environments = [environments]
        self._workers = [
            RollOutWorker(self._policy, self._value_network, env, self._gamma, self._gae_lambda, window_size)
            for env in environments]
        # Monitoring
        self._episode_return = tf.keras.metrics.Mean('episode_return', dtype=tf.float32)
        self._policy_loss = tf.keras.metrics.Mean('policy_loss', dtype=tf.float32)
        self._value_loss = tf.keras.metrics.Mean('value_loss', dtype=tf.float32)
        self._entropy_loss = tf.keras.metrics.Mean('entropy_loss', dtype=tf.float32)
        self._approx_kld = tf.keras.metrics.Mean('approx_kld', dtype=tf.float32)
        self._last_kld = tf.keras.metrics.Mean('last_kld', dtype=tf.float32)
        self._network_updates = tf.keras.metrics.Sum('network_updates', dtype=tf.int32)
        if logs:
            self.summary_writer = tf.summary.create_file_writer(
                f'logs/PPO_{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}')

    def reset_metrics(self):
        self._episode_return.reset_states()
        self._policy_loss.reset_states()
        self._value_loss.reset_states()
        self._entropy_loss.reset_states()
        self._approx_kld.reset_states()
        self._last_kld.reset_states()
        self._network_updates.reset_states()

    def save_metrics(self, epoch):
        with self.summary_writer.as_default():
            tf.summary.scalar('episode_return', self._episode_return.result(), step=epoch)
            tf.summary.scalar('policy_loss', self._policy_loss.result(), step=epoch)
            tf.summary.scalar('value_loss', self._value_loss.result(), step=epoch)
            tf.summary.scalar('entropy_loss', self._entropy_loss.result(), step=epoch)
            tf.summary.scalar('approx_kld', self._approx_kld.result(), step=epoch)
            tf.summary.scalar('last_kld', self._last_kld.result(), step=epoch)
            tf.summary.scalar('network_updates', self._network_updates.result(), step=epoch)

    @tf.function
    def learn(self, data_set):
        kld = 0.0
        for s, a, ret, adv, prob_old_policy in data_set:
            early_stopping, kld = self.train_step_actor(s, a, adv, prob_old_policy)
            if early_stopping:
                break
            self.train_step_critic(s, ret)
            self._approx_kld(kld)
            self._network_updates(1)
        self._last_kld(kld)

    @tf.function
    def train_step_actor(self, s, a, adv, prob_old_policy):
        early_stopping = False
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
                self._policy_loss(loss)
                self._entropy_loss(entropy_loss)
        self._approx_kld(kld)
        return early_stopping, kld

    @tf.function
    def train_step_critic(self, s, ret):
        with tf.GradientTape() as tape:
            prev_v = self._value_network(s)
            loss = self._mse(ret, prev_v)
        gradients = tape.gradient(loss, self._value_network.trainable_variables)
        self._optimizer_value.apply_gradients(zip(gradients, self._value_network.trainable_variables))
        self._value_loss(loss)

    def train(self, epochs, batch_size=64, sub_epochs=4, steps_per_trajectory=1024):
        print("start training!")
        replay_buffer = ReplayBuffer(batch_size)
        for e in range(epochs):
            trajectories = [worker.sample_trajectories(steps_per_trajectory) for worker in
                            self._workers]
            for episodes, ret, dones in trajectories:
                replay_buffer.add_episodes(episodes)
                if dones > 0:
                    self._episode_return(ret)
            print(f"epoch: {e} return of episode: {self._episode_return.result()}")
            self.learn(replay_buffer.get_as_dataset_repeated(sub_epochs))
            if self._logs:
                self.save_metrics(e)
            else:
                print(
                    f"actor_loss: {self._policy_loss.result()}, critic_loss: {self._value_loss.result()}, "
                    f"updates: {self._network_updates.result()}, kld: {self._approx_kld.result()}, "
                    f"last_kld: {self._last_kld.result()}")
            replay_buffer.clear()
            self.reset_metrics()
        print("training finished!")
