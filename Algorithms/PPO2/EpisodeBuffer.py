import tensorflow as tf
import numpy as np


class EpisodeBuffer:

    def __init__(self, gamma, gae_lambda):
        self._gamma = gamma
        self._gae_lambda = gae_lambda
        self._s = []
        self._a = []
        self._r = []
        self._v = []
        self._p = []
        self._d = []

    def add(self, s, a, r, v, p, d):
        self._s.append(tf.convert_to_tensor(s, dtype=tf.float32))
        self._a.append(tf.convert_to_tensor(a, dtype=tf.float32))
        self._r.append(tf.convert_to_tensor(r, dtype=tf.float32))
        self._v.append(tf.convert_to_tensor(v, dtype=tf.float32))
        self._p.append(tf.convert_to_tensor(p, dtype=tf.float32))
        self._d.append(tf.convert_to_tensor(d, dtype=tf.float32))

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

    def get_episode(self):
        adv = self.estimate_advantage(self._r, self._v, self._d)
        g = adv + np.asarray(self._v)
        return (tf.convert_to_tensor(self._s), tf.convert_to_tensor(self._a), tf.convert_to_tensor(g),
                tf.convert_to_tensor(adv), tf.convert_to_tensor(self._p))
