import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import Layer, Dense, LSTM, Conv2D, MaxPool2D, Flatten, Reshape, Rescaling, Concatenate
from tensorflow import math as tfm
from tensorflow_probability import distributions as tfd


class ActorCriticPolicy:
    def __init__(self, state_dim, action_dim):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.network = self.create_actor_critic_network()

    # return combined_actor_critic_model
    def create_actor_critic_network(self):
        pass

    # return prob_current_policy, entropy, value
    def __call__(self, s, a):
        pass

    # return value
    def get_value(self, state):
        pass

    # return actions_prime, log_probs, value
    def act_stochastic(self, state):
        pass

    # return actions
    def act_deterministic(self, state):
        pass


def log_sigma_processing(log_sigma):
    return tfm.exp(log_sigma)


def clip_sigma_processing(sigma):
    return tf.clip_by_value(sigma, 0, tf.float32.max)


def softplus_sigma_processing(sigma):
    return tfm.softplus(sigma)


class GaussianActorCriticPolicy(ActorCriticPolicy):
    def __init__(self, state_dim, action_dim, sigma_processing=log_sigma_processing):
        super().__init__(state_dim, action_dim)
        self.sigma_processing = sigma_processing
        self.network = self.create_actor_critic_network()

    def create_actor_critic_network(self):
        raise NotImplementedError()

    def __call__(self, s, a):
        mu, sigma, value = self.network(s)
        sigma = self.sigma_processing(sigma)
        distribution = tfd.Normal(mu, sigma)
        prob_current_policy = self._log_probs_from_distribution(distribution, a)
        entropy = distribution.entropy()
        return prob_current_policy, entropy, value

    def get_value(self, state):
        _, _, value = self.network(state)
        return value

    def _log_probs_from_distribution(self, distribution, actions):
        log_probs = distribution.log_prob(actions)
        return tfm.reduce_sum(log_probs, axis=-1, keepdims=True)

    def act_stochastic(self, state):
        mu, sigma, value = self.network(state)
        sigma = self.sigma_processing(sigma)
        distribution = tfd.Normal(mu, sigma)
        actions_prime = distribution.sample()
        log_probs = self._log_probs_from_distribution(distribution, actions_prime)
        return actions_prime, log_probs, value

    def act_deterministic(self, state):
        mu, _, _ = self.network(state)
        return mu


class MlpGaussianActorCriticPolicy(GaussianActorCriticPolicy):
    def __init__(self, state_dim, action_dim, sigma_processing=log_sigma_processing, shared_networks=False):
        self.shared_networks = shared_networks
        super().__init__(state_dim, action_dim, sigma_processing)

    def create_actor_critic_network(self):
        if self.shared_networks:
            return self._creat_network()
        else:
            return self._creat_network_separate()

    def _creat_network(self):
        inputs = keras.Input(shape=self.state_dim)
        x = Dense(256, activation=tf.nn.relu)(inputs)
        x = Dense(256, activation=tf.nn.relu)(x)
        x = Dense(256, activation=tf.nn.relu)(x)
        value = Dense(1, activation=None)(x)
        mu = Dense(self.action_dim, activation=None)(x)
        sigma = Dense(self.action_dim, kernel_initializer="zeros")(x)
        model = keras.Model(inputs=inputs, outputs=(mu, sigma, value))
        return model

    def _creat_network_separate(self):
        inputs = keras.Input(shape=self.state_dim)

        x = Dense(256, activation=tf.nn.relu)(inputs)
        x = Dense(256, activation=tf.nn.relu)(x)
        x = Dense(256, activation=tf.nn.relu)(x)
        mu = Dense(self.action_dim, activation=None)(x)
        sigma = Dense(self.action_dim, kernel_initializer="zeros")(x)

        y = Dense(256, activation=tf.nn.relu)(inputs)
        y = Dense(256, activation=tf.nn.relu)(y)
        y = Dense(256, activation=tf.nn.relu)(y)
        value = Dense(1, activation=None)(y)

        model = keras.Model(inputs=inputs, outputs=(mu, sigma, value))
        return model


class CnnGaussianActorCriticPolicy(GaussianActorCriticPolicy):
    def __init__(self, state_dim, action_dim, sigma_processing=log_sigma_processing):
        super().__init__(state_dim, action_dim, sigma_processing)

    def create_actor_critic_network(self):
        inputs = keras.Input(shape=self.state_dim)
        x = Rescaling(scale=1.0 / 255.0, offset=0)(inputs)
        x = Conv2D(filters=32, kernel_size=8, strides=4, padding="valid", activation=tf.nn.relu)(x)
        x = Conv2D(filters=64, kernel_size=4, strides=2, padding="valid", activation=tf.nn.relu)(x)
        x = Conv2D(filters=64, kernel_size=3, strides=1, padding="valid", activation=tf.nn.relu)(x)
        x = Flatten()(x)
        x = Dense(256)(x)
        value = Dense(1, activation=None)(x)
        mu = Dense(self.action_dim, activation=None)(x)
        sigma = Dense(self.action_dim, kernel_initializer="zeros")(x)
        model = keras.Model(inputs=inputs, outputs=(mu, sigma, value))
        return model


class LstmGaussianActorCriticPolicy(GaussianActorCriticPolicy):
    def __init__(self, state_dim, action_dim, sigma_processing=log_sigma_processing, shared_networks=False):
        self.shared_networks = shared_networks
        super().__init__(state_dim, action_dim, sigma_processing)

    def create_actor_critic_network(self):
        if self.shared_networks:
            return self._creat_network()
        else:
            return self._creat_network_separate()

    def _creat_network(self):
        inputs = keras.Input(shape=self.state_dim)
        x = LSTM(256, return_sequences=True)(inputs)
        x = LSTM(256)(x)
        x = Dense(256, activation=tf.nn.relu)(x)
        value = Dense(1, activation=None)(x)
        mu = Dense(self.action_dim, activation=None)(x)
        sigma = Dense(self.action_dim, kernel_initializer="zeros")(x)
        model = keras.Model(inputs=inputs, outputs=(mu, sigma, value))
        return model

    def _creat_network_separate(self):
        inputs = keras.Input(shape=self.state_dim)

        x = LSTM(256, return_sequences=True)(inputs)
        x = LSTM(256)(x)
        x = Dense(256, activation=tf.nn.relu)(x)
        mu = Dense(self.action_dim, activation=None)(x)
        sigma = Dense(self.action_dim, kernel_initializer="zeros")(x)

        y = LSTM(256, return_sequences=True)(inputs)
        y = LSTM(256)(y)
        y = Dense(256, activation=tf.nn.relu)(y)
        value = Dense(1, activation=None)(y)

        model = keras.Model(inputs=inputs, outputs=(mu, sigma, value))
        return model
