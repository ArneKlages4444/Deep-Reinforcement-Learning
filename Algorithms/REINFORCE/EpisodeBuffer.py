import tensorflow as tf
import numpy as np


class EpisodeBuffer:

    def __init__(self, gamma):
        self._gamma = gamma
        self._state_memory = []
        self._action_memory = []
        self.reward_memory = []

    def add(self, state, action, reward):
        self._state_memory.append(tf.convert_to_tensor(state, dtype=tf.float32))
        self._action_memory.append(tf.convert_to_tensor(action, dtype=tf.float32))
        self.reward_memory.append(tf.convert_to_tensor(reward, dtype=tf.float32))

    def get_as_data_set(self, batch_size=1):
        # TODO: better implementation for discounting (cumsum, tf.scan)
        g = np.zeros_like(self.reward_memory, dtype=np.float32)
        for t in range(len(self.reward_memory)):
            g_sum = 0
            gamma_t = 1
            for k in range(t, len(self.reward_memory)):
                g_sum += self.reward_memory[k] * gamma_t  # r_n[k].numpy()
                gamma_t *= self._gamma
            g[t] = g_sum
        # g = (g - np.mean(g)) / (np.std(g) + 1e-10) # normalize g?
        index = tf.range(len(self._state_memory), dtype=tf.float32)
        return tf.data.Dataset.from_tensor_slices(
            (self._state_memory, self._action_memory, self.reward_memory, g, index)).batch(batch_size)
