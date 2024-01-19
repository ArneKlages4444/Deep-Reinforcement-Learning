import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import Layer, Dense, LSTM, Conv2D, MaxPool2D, Flatten, Reshape, Rescaling, Concatenate
from tensorflow import math as tfm
from tensorflow_probability import distributions as tfd
import numpy as np


class ActorCriticPolicy:
    def __init__(self, state_dim):
        self._state_dim = state_dim
        self._network = self.create_actor_critic_network()

    def parameters(self):
        return self._network.trainable_variables

    # return combined_actor_critic_model
    def create_actor_critic_network(self):
        return None

    # return prob_current_policy, entropy, value
    def __call__(self, s, a):
        raise NotImplementedError()

    # return value
    def get_value(self, state):
        raise NotImplementedError()

    # return actions_prime, log_probs, value
    def act_stochastic(self, state):
        raise NotImplementedError()

    # return actions
    def act_deterministic(self, state):
        raise NotImplementedError()

    def act_stochastic_in_env(self, state, environment):
        raise NotImplementedError()


def log_sigma_processing(log_sigma):
    return tfm.exp(log_sigma)


def clip_sigma_processing(sigma):
    return tf.clip_by_value(sigma, 0, tf.float32.max)


def softplus_sigma_processing(sigma):
    return tfm.softplus(sigma)


def no_action_handling(actions, min_action, max_action):
    return actions


def clip_action_handling(actions, min_action, max_action):
    return tf.clip_by_value(actions, min_action, max_action)


class GaussianActorCriticPolicy(ActorCriticPolicy):
    def __init__(self, state_dim, action_dim, action_space,
                 sigma_processing=log_sigma_processing,
                 action_handling=clip_action_handling):
        self._action_dim = action_dim
        super().__init__(state_dim)
        self._min_action = tf.constant(action_space.low, dtype=tf.float32)
        self._max_action = tf.constant(action_space.high, dtype=tf.float32)
        self._sigma_processing = sigma_processing
        self._action_handling = action_handling

    def create_actor_critic_network(self):
        raise NotImplementedError()

    def __call__(self, s, a):
        mu, sigma, value = self._network(s)
        sigma = self._sigma_processing(sigma)
        distribution = tfd.Normal(mu, sigma)
        prob_current_policy = self._log_probs_from_distribution(distribution, a)
        entropy = distribution.entropy()
        return prob_current_policy, entropy, value

    def get_value(self, state):
        _, _, value = self._network(state)
        return value

    def _log_probs_from_distribution(self, distribution, actions):
        log_probs = distribution.log_prob(actions)
        return tfm.reduce_sum(log_probs, axis=-1, keepdims=True)

    def act_stochastic(self, state):
        mu, sigma, value = self._network(state)
        sigma = self._sigma_processing(sigma)
        distribution = tfd.Normal(mu, sigma)
        actions_prime = distribution.sample()
        log_probs = self._log_probs_from_distribution(distribution, actions_prime)
        actions_prime = self._action_handling(actions_prime, self._min_action, self._max_action)
        return actions_prime, log_probs, value

    def act_deterministic(self, state):
        mu, _, _ = self._network(state)
        actions = self._action_handling(mu, self._min_action, self._max_action)
        return actions

    def act_stochastic_in_env(self, state, environment):
        actions_prime, log_probs, value = self.act_stochastic(state)
        observation_prime, reward, terminated, truncated, _ = environment.step(actions_prime.numpy())
        return actions_prime, observation_prime, reward, np.logical_or(terminated, truncated), log_probs, value


class MlpGaussianActorCriticPolicy(GaussianActorCriticPolicy):
    def __init__(self, state_dim, action_dim, action_space,
                 sigma_processing=log_sigma_processing,
                 action_handling=no_action_handling,
                 shared_networks=False):
        self.shared_networks = shared_networks
        super().__init__(state_dim, action_dim, action_space, sigma_processing, action_handling)

    def create_actor_critic_network(self):
        if self.shared_networks:
            return self._creat_network()
        else:
            return self._creat_network_separate()

    def _creat_network(self):
        inputs = keras.Input(shape=self._state_dim)
        x = Dense(256, activation=tf.nn.relu)(inputs)
        x = Dense(256, activation=tf.nn.relu)(x)
        x = Dense(256, activation=tf.nn.relu)(x)
        value = Dense(1, activation=None)(x)
        mu = Dense(self._action_dim, activation=None)(x)
        sigma = Dense(self._action_dim, kernel_initializer="zeros")(x)
        model = keras.Model(inputs=inputs, outputs=(mu, sigma, value))
        return model

    def _creat_network_separate(self):
        inputs = keras.Input(shape=self._state_dim)

        x = Dense(256, activation=tf.nn.relu)(inputs)
        x = Dense(256, activation=tf.nn.relu)(x)
        x = Dense(256, activation=tf.nn.relu)(x)
        mu = Dense(self._action_dim, activation=None)(x)
        sigma = Dense(self._action_dim, kernel_initializer="zeros")(x)

        y = Dense(256, activation=tf.nn.relu)(inputs)
        y = Dense(256, activation=tf.nn.relu)(y)
        y = Dense(256, activation=tf.nn.relu)(y)
        value = Dense(1, activation=None)(y)

        model = keras.Model(inputs=inputs, outputs=(mu, sigma, value))
        return model


class CnnGaussianActorCriticPolicy(GaussianActorCriticPolicy):
    def __init__(self, state_dim, action_dim, action_space,
                 sigma_processing=log_sigma_processing,
                 action_handling=no_action_handling):
        super().__init__(state_dim, action_dim, action_space, sigma_processing, action_handling)

    def create_actor_critic_network(self):
        inputs = keras.Input(shape=self._state_dim)
        x = Rescaling(scale=1.0 / 255.0, offset=0)(inputs)
        x = Conv2D(filters=32, kernel_size=8, strides=4, padding="valid", activation=tf.nn.relu)(x)
        x = Conv2D(filters=64, kernel_size=4, strides=2, padding="valid", activation=tf.nn.relu)(x)
        x = Conv2D(filters=64, kernel_size=3, strides=1, padding="valid", activation=tf.nn.relu)(x)
        x = Flatten()(x)
        x = Dense(512)(x)

        y = Dense(256)(x)
        value = Dense(1, activation=None)(y)

        z = Dense(256)(x)
        mu = Dense(self._action_dim, activation=None)(z)
        sigma = Dense(self._action_dim, kernel_initializer="zeros")(z)

        model = keras.Model(inputs=inputs, outputs=(mu, sigma, value))
        return model


class LstmGaussianActorCriticPolicy(GaussianActorCriticPolicy):
    def __init__(self, state_dim, action_dim, action_space,
                 sigma_processing=log_sigma_processing,
                 action_handling=no_action_handling, shared_networks=False):
        self.shared_networks = shared_networks
        super().__init__(state_dim, action_dim, action_space, sigma_processing, action_handling)

    def create_actor_critic_network(self):
        if self.shared_networks:
            return self._creat_network()
        else:
            return self._creat_network_separate()

    def _creat_network(self):
        inputs = keras.Input(shape=self._state_dim)
        x = LSTM(256, return_sequences=True)(inputs)
        x = LSTM(256)(x)
        x = Dense(256, activation=tf.nn.relu)(x)
        value = Dense(1, activation=None)(x)
        mu = Dense(self._action_dim, activation=None)(x)
        sigma = Dense(self._action_dim, kernel_initializer="zeros")(x)
        model = keras.Model(inputs=inputs, outputs=(mu, sigma, value))
        return model

    def _creat_network_separate(self):
        inputs = keras.Input(shape=self._state_dim)

        x = LSTM(256, return_sequences=True)(inputs)
        x = LSTM(256)(x)
        x = Dense(256, activation=tf.nn.relu)(x)
        mu = Dense(self._action_dim, activation=None)(x)
        sigma = Dense(self._action_dim, kernel_initializer="zeros")(x)

        y = LSTM(256, return_sequences=True)(inputs)
        y = LSTM(256)(y)
        y = Dense(256, activation=tf.nn.relu)(y)
        value = Dense(1, activation=None)(y)

        model = keras.Model(inputs=inputs, outputs=(mu, sigma, value))
        return model


class DiscreteActorCriticPolicy(ActorCriticPolicy):
    def __init__(self, state_dim, n_actions):
        self._n_actions = n_actions
        super().__init__(state_dim)

    def create_actor_critic_network(self):
        raise NotImplementedError()

    def __call__(self, s, a):
        logits, value = self._network(s)
        distribution = tfd.Categorical(probs=logits)
        prob_current_policy = self._log_probs_from_distribution(distribution, tf.squeeze(a, -1))
        entropy = distribution.entropy()
        return prob_current_policy, entropy, value

    def get_value(self, state):
        _, value = self._network(state)
        return value

    def _log_probs_from_distribution(self, distribution, actions):
        log_probs = distribution.log_prob(actions)
        return tf.expand_dims(log_probs, -1)

    def act_stochastic(self, state):
        logits, value = self._network(state)
        distribution = tfd.Categorical(probs=logits)
        actions_prime = distribution.sample()
        log_probs = self._log_probs_from_distribution(distribution, actions_prime)
        actions_prime = tf.expand_dims(actions_prime, -1)
        return actions_prime, log_probs, value

    def act_deterministic(self, state):
        logits, _ = self._network(state)
        return tf.math.argmax(logits, axis=-1, output_type=tf.dtypes.int32)

    def act_stochastic_in_env(self, state, environment):
        actions_prime, log_probs, value = self.act_stochastic(state)
        observation_prime, reward, terminated, truncated, _ = environment.step(tf.squeeze(actions_prime, -1).numpy())
        return actions_prime, observation_prime, reward, np.logical_or(terminated, truncated), log_probs, value


class MlpDiscreteActorCriticPolicy(DiscreteActorCriticPolicy):
    def __init__(self, state_dim, n_actions, shared_networks=False):
        self._shared_networks = shared_networks
        super().__init__(state_dim, n_actions)

    def create_actor_critic_network(self):
        if self._shared_networks:
            return self._creat_network()
        else:
            return self._creat_network_separate()

    def _creat_network(self):
        inputs = keras.Input(shape=self._state_dim)
        x = Dense(256, activation=tf.nn.relu)(inputs)
        x = Dense(256, activation=tf.nn.relu)(x)
        x = Dense(256, activation=tf.nn.relu)(x)
        value = Dense(1, activation=None)(x)
        logits = Dense(self._n_actions, activation=None)(x)
        model = keras.Model(inputs=inputs, outputs=(logits, value))
        return model

    def _creat_network_separate(self):
        inputs = keras.Input(shape=self._state_dim)

        x = Dense(256, activation=tf.nn.relu)(inputs)
        x = Dense(256, activation=tf.nn.relu)(x)
        x = Dense(256, activation=tf.nn.relu)(x)
        logits = Dense(self._n_actions, activation='softmax')(x)

        y = Dense(256, activation=tf.nn.relu)(inputs)
        y = Dense(256, activation=tf.nn.relu)(y)
        y = Dense(256, activation=tf.nn.relu)(y)
        value = Dense(1, activation=None)(y)

        model = keras.Model(inputs=inputs, outputs=(logits, value))
        return model
