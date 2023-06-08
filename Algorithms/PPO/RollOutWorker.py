import tensorflow as tf
import numpy as np
from gymnasium.wrappers import FrameStack


class RollOutWorker:

    def __init__(self, policy, value_network, environment, gamma, gae_lambda, window_size):
        self._policy = policy
        self._value_network = value_network
        self._environment = environment if window_size is None else FrameStack(environment, window_size)
        self._gamma = gamma
        self._gae_lambda = gae_lambda
        self._s = []
        self._a = []
        self._r = []
        self._v = []
        self._p = []
        self._d = []

        self.__s, _ = self._environment.reset()
        self.__d = 0
        self.__d_p = 0.0
        self.__s_p = None
        self.__ret = 0

    def add(self, s, a, r, v, p, d):
        self._s.append(tf.convert_to_tensor(s, dtype=tf.float32))
        self._a.append(tf.convert_to_tensor(a, dtype=tf.float32))
        self._r.append(tf.convert_to_tensor(r, dtype=tf.float32))
        self._v.append(tf.convert_to_tensor(v, dtype=tf.float32))
        self._p.append(tf.convert_to_tensor(p, dtype=tf.float32))
        self._d.append(tf.convert_to_tensor(d, dtype=tf.float32))

    def clear(self):
        self._s.clear()
        self._a.clear()
        self._r.clear()
        self._v.clear()
        self._p.clear()
        self._d.clear()

    # generalized advantage estimate (taken from https://ppo-details.cleanrl.dev//2021/11/05/ppo-implementation-details/)
    def estimate_advantage(self, rewards, values, dones, next_done, next_value):  # TODO: rework
        adv = np.zeros_like(rewards)
        last_gae_lamda = 0
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_non_terminal = 1.0 - next_done
                next_values = next_value
            else:
                next_non_terminal = 1.0 - dones[t + 1]
                next_values = values[t + 1]
            delta = rewards[t] + self._gamma * next_values * next_non_terminal - values[t]
            adv[t] = last_gae_lamda = delta + self._gamma * self._gae_lambda * next_non_terminal * last_gae_lamda
        return adv

    # TODO: Make Vectorized!!!
    def sample_trajectories(self, steps_per_trajectory):
        ack_ret = 0
        x = 0
        for _ in range(steps_per_trajectory):
            a, self.__s_p, r, self.__d_p, p = self._policy.act_stochastic(self.__s, self._environment)
            self.__d_p = float(self.__d_p)
            self.__ret += r
            v = self._value_network(tf.convert_to_tensor([self.__s], dtype=tf.float32))
            self.add(self.__s, tf.squeeze(a, 1), [r], tf.squeeze(v, 1), tf.squeeze(p, 1), self.__d)
            if self.__d_p:
                x += 1
                ack_ret += self.__ret
                self.__ret = 0
                self.__s, _ = self._environment.reset()
            self.__s = self.__s_p
            self.__d = self.__d_p
        v_p = self._value_network(tf.convert_to_tensor([self.__s_p], dtype=tf.float32))
        adv = self.estimate_advantage(self._r, self._v, self._d, next_done=self.__d_p, next_value=v_p)
        g = adv + np.asarray(self._v)  # TD(lambda)
        ds = tf.data.Dataset.from_tensor_slices((tf.convert_to_tensor(self._s), tf.convert_to_tensor(self._a),
                                                 tf.convert_to_tensor(g), tf.convert_to_tensor(adv),
                                                 tf.convert_to_tensor(self._p)))
        self.clear()
        return ds, self.__ret if x < 1 else ack_ret / x, x
