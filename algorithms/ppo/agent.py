from typing import Union

import tensorflow as tf
from tensorflow import math as tfm
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import LearningRateSchedule
from gymnasium.vector.vector_env import VectorEnv
import time

from policies import ActorCriticPolicy
from roll_out_worker import RollOutWorker


class Agent:
    def __init__(
            self,
            env: VectorEnv,
            policy: ActorCriticPolicy,
            epsilon=0.2,
            gae_lambda=0.95,
            learning_rate: Union[float, LearningRateSchedule] = 0.0003,
            gamma=0.99,
            alpha=0.2,
            kld_threshold=0.05,
            normalize_adv=False,
            value_loss_coefficient=0.5,
            value_clip_range=None,
            global_clipnorm=0.5,
            log_dir=None,
            verbose=False,
            batch_size=256,
            data_set_repeats=4,
            steps_per_epoch=2048
    ):
        """
        Proximal Policy Optimization algorithm (PPO)

        Paper: https://arxiv.org/abs/1707.06347
        Pseudocode: https://spinningup.openai.com/en/latest/algorithms/ppo.html
        Improvements: https://ppo-details.cleanrl.dev//2021/11/05/ppo-implementation-details/

        :param env: Environment to learn from (conforming to te VectorEnv interface)
        :param policy: Policy model to be use (determining if continuous or discrete action spaces are used
            and the neural network models)
        :param epsilon: Policy clipping parameter
        :param gae_lambda: Bias vs variance trade-off factor for Generalized Advantage Estimator
        :param learning_rate: Learning rate for the Adam optimizer
            (instead of a fixed learning rate, a learning rate schedule can be used)
        :param gamma: Discount factor
        :param alpha: Entropy coefficient
        :param kld_threshold: Maximum Kullback-Leibler divergence between updates for early stopping
        :param normalize_adv: Controls if advantages are normalized
        :param value_loss_coefficient: Value function coefficient
        :param value_clip_range: Clipping parameter for the value function
        :param global_clipnorm: global clipnorm parameter of the Adam optimizer
        :param log_dir: Log folder for the tensorboard logs
        :param verbose: Print progress to the console
        :param batch_size: Minibatch size
        :param data_set_repeats: Number of repeats of the sampled memory
        :param steps_per_epoch: Number of time steps sampled per episode/epoch (steps_per_epoch = num_envs * steps)
        """
        self._epsilon = epsilon
        self._gae_lambda = gae_lambda
        self._gamma = gamma
        self._alpha = alpha
        self._learning_rate = learning_rate
        self._mse = tf.keras.losses.MeanSquaredError()
        self._optimizer = Adam(learning_rate=learning_rate, global_clipnorm=global_clipnorm)
        self._kld_threshold = kld_threshold
        self._normalize_adv = normalize_adv
        self._value_loss_coefficient = value_loss_coefficient
        self._value_clip_range = value_clip_range
        self._policy = policy
        self._log_dir = log_dir
        self._verbose = verbose
        self._num_envs = env.num_envs
        self._batch_size = batch_size
        self._data_set_repeats = data_set_repeats
        self._steps_per_trajectory = steps_per_epoch // self._num_envs
        self._shuffle_buffer_size = 1024
        self._roll_out_worker = RollOutWorker(
            self._policy,
            env,
            self._gamma,
            self._gae_lambda,
            self._num_envs,
        )
        # Monitoring
        self._episode_return = tf.keras.metrics.Mean("episode_return", dtype=tf.float32)
        self._policy_loss = tf.keras.metrics.Mean("policy_loss", dtype=tf.float32)
        self._value_loss = tf.keras.metrics.Mean("value_loss", dtype=tf.float32)
        self._entropy_loss = tf.keras.metrics.Mean("entropy_loss", dtype=tf.float32)
        self._approx_kld = tf.keras.metrics.Mean("approx_kld", dtype=tf.float32)
        self._last_kld = tf.keras.metrics.Mean("last_kld", dtype=tf.float32)
        self._network_updates = tf.keras.metrics.Sum("network_updates", dtype=tf.int32)
        self._finished_episodes = tf.keras.metrics.Sum("finished_episodes", dtype=tf.int32)
        self._time_steps = tf.keras.metrics.Sum("time_steps", dtype=tf.int32)
        if log_dir is not None:
            self.summary_writer = tf.summary.create_file_writer(log_dir)

    def reset_metrics(self):
        self._episode_return.reset_states()
        self._policy_loss.reset_states()
        self._value_loss.reset_states()
        self._entropy_loss.reset_states()
        self._approx_kld.reset_states()
        self._last_kld.reset_states()
        self._network_updates.reset_states()

    def save_metrics(self, epoch, dones):
        with self.summary_writer.as_default():
            episode_return = self._episode_return.result()
            tf.summary.scalar("policy_loss", self._policy_loss.result(), step=epoch)
            tf.summary.scalar("value_loss", self._value_loss.result(), step=epoch)
            tf.summary.scalar("entropy_loss", self._entropy_loss.result(), step=epoch)
            tf.summary.scalar("approx_kld", self._approx_kld.result(), step=epoch)
            tf.summary.scalar("last_kld", self._last_kld.result(), step=epoch)
            tf.summary.scalar("network_updates", self._network_updates.result(), step=epoch)
            tf.summary.scalar("finished_episodes", self._finished_episodes.result(), step=epoch)
            time_steps = self._time_steps.result()
            tf.summary.scalar("time_steps", time_steps, step=epoch)
            time_elapsed = time.time() - self.__start_time
            tf.summary.scalar("time_elapsed", time_elapsed, step=epoch)

            time_steps = tf.cast(time_steps, tf.int64)
            tf.summary.scalar("time_elapsed_after_steps", time_elapsed, step=time_steps)
            if dones > 0:
                tf.summary.scalar("episode_return", episode_return, step=epoch)
                tf.summary.scalar("rollout/ep_rew_mean", episode_return, step=time_steps)

    @tf.function
    def learn(self, data_set):
        kld = 0.
        for s, a, ret, adv, prob_old_policy, old_value in data_set:
            if self._normalize_adv:
                adv = (adv - tfm.reduce_mean(adv)) / (tfm.reduce_std(adv) + 1e-8)
            with tf.GradientTape() as tape:
                prob_current_policy, entropy, predicted_value = self._policy(s, a)
                actor_loss = self.actor_loss(adv, prob_old_policy, prob_current_policy, entropy)

                log_ratio = prob_current_policy - prob_old_policy
                kld = tf.math.reduce_mean((tf.math.exp(log_ratio) - 1) - log_ratio)  # approximate kld
                self._approx_kld(kld)
                if not self._kld_threshold or kld < self._kld_threshold:
                    if self._value_clip_range is not None:
                        predicted_value = old_value + tf.clip_by_value(predicted_value - old_value,
                                                                       -self._value_clip_range,
                                                                       self._value_clip_range)
                    value_loss = self._value_loss_coefficient * self._mse(ret, predicted_value)
                    combined_loss = actor_loss + value_loss

                    gradients = tape.gradient(combined_loss, self._policy.parameters())
                    self._optimizer.apply_gradients(zip(gradients, self._policy.parameters()))

                    self._value_loss(value_loss)
                    self._approx_kld(kld)
                    self._network_updates(1)
                else:
                    break
        self._last_kld(kld)

    @tf.function
    def actor_loss(self, adv, prob_old_policy, prob_current_policy, entropy):
        # prob of current policy / prob of old policy (log probs: p/p2 = log(p)-log(p2)
        p = tf.math.exp(prob_current_policy - prob_old_policy)  # exp() to un do log(p)
        clipped_p = tf.clip_by_value(p, 1 - self._epsilon, 1 + self._epsilon)
        policy_loss = -tfm.reduce_mean(tfm.minimum(p * adv, clipped_p * adv))
        # entropy_loss = -tfm.reduce_mean(-prob_current_policy)  # approximate entropy
        entropy_loss = -tfm.reduce_mean(entropy)
        loss = policy_loss + self._alpha * entropy_loss

        self._policy_loss(loss)
        self._entropy_loss(entropy_loss)
        return loss

    def train(self, epochs):
        """
        Train function to start the PPO training
        :param epochs: number of epochs to train (epochs = total_number_of_time_steps / steps_per_epoch)
        """
        print("start training!")
        self.__start_time = time.time()
        for e in range(epochs):
            episodes_data_ds, ret, dones = self._roll_out_worker.sample_trajectories(self._steps_per_trajectory)
            episodes_data_ds = episodes_data_ds.shuffle(
                self._shuffle_buffer_size).batch(
                self._batch_size).repeat(
                self._data_set_repeats)
            if dones > 0:
                self._finished_episodes(dones)
                self._episode_return(ret)
            print(f"epoch: {e} return of episode: {self._episode_return.result()}, done episodes: {dones}")
            self.learn(episodes_data_ds)
            self._time_steps(self._steps_per_trajectory * self._num_envs)
            if self._log_dir is not None:
                self.save_metrics(e, dones)
            if self._verbose:
                print(
                    f"actor_loss: {self._policy_loss.result()}, critic_loss: {self._value_loss.result()}, "
                    f"updates: {self._network_updates.result()}, kld: {self._approx_kld.result()}, "
                    f"last_kld: {self._last_kld.result()}"
                )
            self.reset_metrics()
        print("training finished!")
