import tensorflow as tf
import numpy as np


class ExperienceReplayBuffer:
    def __init__(self, state_dims, action_dims, max_size=1000000, batch_size=256):
        self._max_size = max_size
        self._batch_size = batch_size
        self._size = 0
        self._current_position = 0
        self._state_memory = np.zeros((self._max_size, *state_dims))
        self._state_prime_memory = np.zeros((self._max_size, *state_dims))
        self._action_memory = np.zeros((self._max_size, action_dims))
        self._reward_memory = np.zeros((self._max_size, 1))
        self._done_memory = np.zeros((self._max_size, 1), dtype=np.bool)

    def size(self):
        return self._size

    def add_transition(self, state, action, reward, state_, done):
        self._state_memory[self._current_position] = state
        self._state_prime_memory[self._current_position] = state_
        self._action_memory[self._current_position] = action
        self._reward_memory[self._current_position] = reward
        self._done_memory[self._current_position] = done
        # self.un_norm_r[self.current_position] = r
        # self.r = (self.un_norm_r - np.mean(self.un_norm_r)) / (np.std(self.un_norm_r) + 1e-10)
        if self._size < self._max_size:
            self._size += 1
        self._current_position = (self._current_position + 1) % self._max_size

    def sample_batch(self):
        batch_indices = np.random.choice(self._size, self._batch_size, replace=False)
        states = tf.convert_to_tensor(self._state_memory[batch_indices], dtype=tf.float32)
        states_prime = tf.convert_to_tensor(self._state_prime_memory[batch_indices], dtype=tf.float32)
        actions = tf.convert_to_tensor(self._action_memory[batch_indices], dtype=tf.float32)
        rewards = tf.convert_to_tensor(self._reward_memory[batch_indices], dtype=tf.float32)
        dones = tf.convert_to_tensor(self._done_memory[batch_indices], dtype=tf.float32)
        return states, actions, rewards, states_prime, dones
