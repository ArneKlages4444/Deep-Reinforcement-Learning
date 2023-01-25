import tensorflow as tf
import numpy as np


class EpisodeBuffer:

    def __init__(self, advantage_estimator, calc_rewards_to_go, batch_size=256):
        self._advantage_estimator = advantage_estimator
        self._calc_rewards_to_go = calc_rewards_to_go
        self._batch_size = batch_size
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

    def get_as_data_set(self):
        adv = self._advantage_estimator(self._r, self._v, self._d)
        g = self._calc_rewards_to_go(self._r, self._v, adv)
        return tf.data.Dataset.from_tensor_slices((self._s, self._a, self._r, g, adv, self._p)).shuffle(
            np.minimum(512)).batch(self._batch_size)
