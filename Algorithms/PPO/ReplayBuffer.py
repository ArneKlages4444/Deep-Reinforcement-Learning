import tensorflow as tf


class ReplayBuffer:

    def __init__(self, batch_size, shuffle_buffer_size=1024):
        self._episodes = []
        self._batch_size = batch_size
        self._shuffle_buffer_size = shuffle_buffer_size

    def add_episodes(self, episodes):
        self._episodes.append(episodes)

    def get_as_dataset_of_length(self, dataset_size=10000):
        choice_dataset = tf.data.Dataset.from_tensor_slices(
            tf.squeeze(tf.random.categorical([[float(x) for x in range(len(self._episodes))]], dataset_size)))
        episodes = [ds.shuffle(self._shuffle_buffer_size).repeat() for ds in self._episodes]
        return tf.data.Dataset.choose_from_datasets(episodes, choice_dataset).batch(self._batch_size)

    def get_as_dataset(self):
        ds = self._episodes[0]
        for e in self._episodes[1:len(self._episodes)]:
            ds = ds.concatenate(e)
        return ds.shuffle(self._shuffle_buffer_size).batch(self._batch_size)

    def get_as_dataset_repeated(self, count=4):
        return self.get_as_dataset().repeat(count)

    def clear(self):
        self._episodes.clear()
