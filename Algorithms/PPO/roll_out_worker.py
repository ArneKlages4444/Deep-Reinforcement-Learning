import tensorflow as tf
import numpy as np


class RollOutWorker:
    def __init__(
            self,
            policy,
            environment,
            gamma,
            gae_lambda,
            num_envs,
    ):
        self._policy = policy
        self._environment = environment
        self._gamma = gamma
        self._gae_lambda = gae_lambda
        self._num_envs = num_envs
        self._s = []
        self._a = []
        self._r = []
        self._v = []
        self._p = []
        self._d = []
        self.__s, _ = self._environment.reset()
        self.__d = np.zeros(num_envs)
        self.__d_p = np.zeros(num_envs)
        self.__s_p = None
        self.__ret = np.zeros(num_envs)

    def add(self, s, a, r, v, p, d):
        self._s.append(s)
        self._a.append(a)
        self._r.append(r)
        self._v.append(v)
        self._p.append(p)
        self._d.append(d)

    def clear(self):
        self._s.clear()
        self._a.clear()
        self._r.clear()
        self._v.clear()
        self._p.clear()
        self._d.clear()

    def results_to_tensors(self):
        s = tf.convert_to_tensor(self._s, dtype=tf.float32)
        a = tf.convert_to_tensor(self._a, dtype=tf.float32)
        r = tf.convert_to_tensor(self._r, dtype=tf.float32)
        v = tf.convert_to_tensor(self._v, dtype=tf.float32)
        p = tf.convert_to_tensor(self._p, dtype=tf.float32)
        d = tf.convert_to_tensor(self._d, dtype=tf.float32)
        return s, a, r, v, p, d

    # generalized advantage estimate
    def estimate_advantage(self, rewards, values, dones, last_done, last_value):
        next_non_terminal = 1.0 - last_done
        adv = tf.zeros(self._num_envs, dtype=tf.float32)
        _, _, advantages = tf.scan(
            self.estimate_advantage_aux, (rewards, values, dones),
            initializer=(last_value, next_non_terminal, adv),
            reverse=True
        )
        return advantages

    def estimate_advantage_aux(self, rets, inputs):
        rewards, values, dones = inputs
        next_values, next_non_terminal, advantage = rets
        delta = (rewards + self._gamma * next_values * next_non_terminal - values)
        advantage = delta + self._gamma * self._gae_lambda * next_non_terminal * advantage
        next_non_terminal = 1.0 - dones
        next_values = values
        return next_values, next_non_terminal, advantage

    def sample_trajectories(self, steps_per_trajectory):
        ack_ret = 0.0
        x = 0.0
        for _ in range(steps_per_trajectory):
            a, self.__s_p, r, self.__d_p, p, v = self._policy.act_stochastic_in_env(self.__s, self._environment)
            self.__d_p = self.__d_p.astype(float)
            self.__ret += r
            v = tf.squeeze(v, -1)
            self.add(self.__s, a, r, v, p, self.__d)
            if np.any(self.__d_p):
                x += np.sum(self.__d_p)
                ack_ret += np.sum(np.multiply(self.__ret, self.__d_p))
                self.__ret = np.multiply(self.__ret, np.logical_not(self.__d_p))
            self.__s = self.__s_p
            self.__d = self.__d_p
        v_p = tf.squeeze(
            self._policy.get_value(tf.convert_to_tensor(self.__s_p, dtype=tf.float32)),
            -1
        )
        states, actions, rewards, values, log_probs, dones = self.results_to_tensors()
        adv = self.estimate_advantage(
            rewards, values, dones,
            last_done=tf.convert_to_tensor(self.__d_p, dtype=tf.float32),
            last_value=v_p
        )
        g = adv + values  # TD(lambda)
        returns = tf.expand_dims(g, -1)
        advantages = tf.expand_dims(adv, -1)
        ds = tf.data.Dataset.from_tensor_slices((states, actions, returns, advantages, log_probs)).unbatch()
        self.clear()
        return ds, self.__ret if x < 1 else ack_ret / x, x
