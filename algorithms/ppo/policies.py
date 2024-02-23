import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import Layer, Dense, LSTM, Conv2D, MaxPool2D, Flatten, Reshape, Rescaling, Concatenate
from tensorflow import math as tfm
from tensorflow_probability import distributions as tfd
from tensorflow.keras.initializers import Orthogonal
from tensorflow.keras.initializers import Zeros
import numpy as np


class ActorCriticPolicy:
    def __init__(self, state_dim):
        """
        ActorCriticPolicy base class
        :param state_dim: Dimensions of the state space
        """
        self._state_dim = state_dim
        self._network = self.create_actor_critic_network()

    def parameters(self):
        return self._network.trainable_variables

    def create_actor_critic_network(self):
        """
        must override function to create the neural network models
        :returns combined_actor_critic_model
        """
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

    def process_actions_for_environment(self, actions):
        return actions

    def save(self, path):
        self._network.save_weights(path)

    def load(self, path_to_parameters):
        self._network.load_weights(path_to_parameters)


@tf.function
def log_sigma_processing(log_sigma):
    return tfm.exp(log_sigma)


@tf.function
def clip_sigma_processing(sigma):
    return tf.clip_by_value(sigma, 0, tf.float32.max)


@tf.function
def softplus_sigma_processing(sigma):
    return tfm.softplus(sigma)


@tf.function
def no_action_handling(actions, min_action, max_action):
    return actions


@tf.function
def clip_action_handling(actions, min_action, max_action):
    return tf.clip_by_value(actions, min_action, max_action)


class GaussianActorCriticPolicy(ActorCriticPolicy):
    def __init__(self, state_dim, action_dim, action_space,
                 sigma_processing=log_sigma_processing,
                 action_handling=clip_action_handling):
        """
        GaussianActorCriticPolicy: Base class for a continuous gaussian policy
        :param state_dim: Dimensions of the state space
        :param state_dim: Dimensions of the action space
        :param sigma_processing: determines the handling of the standard deviation (as default log std is used)
        :param action_handling: determines the handling of actions
            (as default actions that are out of the defined interval are clipped)
        """
        self._action_dim = action_dim
        super().__init__(state_dim)
        self._min_action = tf.constant(action_space.low, dtype=tf.float32)
        self._max_action = tf.constant(action_space.high, dtype=tf.float32)
        self._sigma_processing = sigma_processing
        self._action_handling = action_handling

    def create_actor_critic_network(self):
        raise NotImplementedError()

    @tf.function
    def __call__(self, s, a):
        mu, sigma, value = self._network(s)
        sigma = self._sigma_processing(sigma)
        distribution = tfd.Normal(mu, sigma)
        prob_current_policy = self._log_probs_from_distribution(distribution, a)
        entropy = distribution.entropy()
        return prob_current_policy, entropy, value

    @tf.function
    def get_value(self, state):
        _, _, value = self._network(state)
        return value

    @tf.function
    def _log_probs_from_distribution(self, distribution, actions):
        log_probs = distribution.log_prob(actions)
        return tfm.reduce_sum(log_probs, axis=-1, keepdims=True)

    @tf.function
    def act_stochastic(self, state):
        mu, sigma, value = self._network(state)
        sigma = self._sigma_processing(sigma)
        distribution = tfd.Normal(mu, sigma)
        actions_prime = distribution.sample()
        log_probs = self._log_probs_from_distribution(distribution, actions_prime)
        actions_prime = self._action_handling(actions_prime, self._min_action, self._max_action)
        return actions_prime, log_probs, value

    @tf.function
    def act_deterministic(self, state):
        mu, _, _ = self._network(state)
        actions = self._action_handling(mu, self._min_action, self._max_action)
        return actions

    def process_actions_for_environment(self, actions):
        return actions.numpy()


class DiscreteActorCriticPolicy(ActorCriticPolicy):
    def __init__(self, state_dim, n_actions):
        """
        GaussianActorCriticPolicy: Base class for a continuous gaussian policy
        :param state_dim: Dimensions of the state space
        :param n_actions: number of actions
        """
        self._n_actions = n_actions
        super().__init__(state_dim)

    def create_actor_critic_network(self):
        raise NotImplementedError()

    @tf.function
    def __call__(self, s, a):
        logits, value = self._network(s)
        distribution = tfd.Categorical(logits=logits)
        prob_current_policy = self._log_probs_from_distribution(distribution, tf.squeeze(a, -1))
        entropy = distribution.entropy()
        return prob_current_policy, entropy, value

    @tf.function
    def get_value(self, state):
        _, value = self._network(state)
        return value

    @tf.function
    def _log_probs_from_distribution(self, distribution, actions):
        log_probs = distribution.log_prob(actions)
        return tf.expand_dims(log_probs, -1)

    @tf.function
    def act_stochastic(self, state):
        logits, value = self._network(state)
        distribution = tfd.Categorical(logits=logits)
        actions_prime = distribution.sample()
        log_probs = self._log_probs_from_distribution(distribution, actions_prime)
        actions_prime = tf.expand_dims(actions_prime, -1)
        return actions_prime, log_probs, value

    @tf.function
    def act_deterministic(self, state):
        logits, _ = self._network(state)
        return tf.math.argmax(logits, axis=-1, output_type=tf.dtypes.int32)

    def process_actions_for_environment(self, actions):
        return tf.squeeze(actions, -1).numpy()


class MlpGaussianActorCriticPolicy(GaussianActorCriticPolicy):
    def __init__(self, state_dim, action_dim, action_space,
                 sigma_processing=log_sigma_processing,
                 action_handling=no_action_handling,
                 shared_networks=False):
        self.shared_networks = shared_networks
        super().__init__(state_dim, action_dim, action_space, sigma_processing, action_handling)

    def create_actor_critic_network(self):
        initializer = Orthogonal(np.sqrt(2), seed=1)

        inputs = keras.Input(shape=self._state_dim)

        x = Dense(256, activation=tf.nn.relu,
                  kernel_initializer=initializer,
                  bias_initializer=Zeros())(inputs)
        x = Dense(128, activation=tf.nn.relu,
                  kernel_initializer=initializer,
                  bias_initializer=Zeros())(x)
        x = Dense(64, activation=tf.nn.relu,
                  kernel_initializer=initializer,
                  bias_initializer=Zeros())(x)
        mu = Dense(self._action_dim, activation=None,
                   kernel_initializer=Orthogonal(0.01),
                   bias_initializer=Zeros())(x)
        sigma = Dense(self._action_dim,
                      kernel_initializer=Zeros(),
                      bias_initializer=Zeros())(x)

        y = Dense(256, activation=tf.nn.relu,
                  kernel_initializer=initializer,
                  bias_initializer=Zeros())(inputs)
        y = Dense(128, activation=tf.nn.relu,
                  kernel_initializer=initializer,
                  bias_initializer=Zeros())(y)
        y = Dense(64, activation=tf.nn.relu,
                  kernel_initializer=initializer,
                  bias_initializer=Zeros())(y)
        value = Dense(1, activation=None,
                      kernel_initializer=Orthogonal(1),
                      bias_initializer=Zeros())(y)

        model = keras.Model(inputs=inputs, outputs=(mu, sigma, value))
        return model


class MlpDiscreteActorCriticPolicy(DiscreteActorCriticPolicy):
    def __init__(self, state_dim, n_actions):
        super().__init__(state_dim, n_actions)

    def create_actor_critic_network(self):
        initializer = Orthogonal(np.sqrt(2), seed=1)

        inputs = keras.Input(shape=self._state_dim)

        x = Dense(256, activation=tf.nn.relu,
                  kernel_initializer=initializer,
                  bias_initializer=Zeros())(inputs)
        x = Dense(128, activation=tf.nn.relu,
                  kernel_initializer=initializer,
                  bias_initializer=Zeros())(x)
        x = Dense(64, activation=tf.nn.relu,
                  kernel_initializer=initializer,
                  bias_initializer=Zeros())(x)
        logits = Dense(self._n_actions, activation=None,
                       kernel_initializer=Orthogonal(0.01),
                       bias_initializer=Zeros())(x)

        y = Dense(256, activation=tf.nn.relu,
                  kernel_initializer=initializer,
                  bias_initializer=Zeros())(inputs)
        y = Dense(128, activation=tf.nn.relu,
                  kernel_initializer=initializer,
                  bias_initializer=Zeros())(y)
        y = Dense(64, activation=tf.nn.relu,
                  kernel_initializer=initializer,
                  bias_initializer=Zeros())(y)
        value = Dense(1, activation=None,
                      kernel_initializer=Orthogonal(1),
                      bias_initializer=Zeros())(y)

        model = keras.Model(inputs=inputs, outputs=(logits, value))
        return model


class MlpGaussianActorCriticPolicyIndependentSigma(GaussianActorCriticPolicy):

    def create_actor_critic_network(self):
        class MyMlpModel(keras.Model):

            def __init__(self, action_dim):
                super(MyMlpModel, self).__init__()

                initializer = Orthogonal(np.sqrt(2), seed=1)

                self.mu_0 = Dense(256, activation=tf.nn.relu,
                                  kernel_initializer=initializer,
                                  bias_initializer=Zeros())
                self.mu_1 = Dense(128, activation=tf.nn.relu,
                                  kernel_initializer=initializer,
                                  bias_initializer=Zeros())
                self.mu_2 = Dense(64, activation=tf.nn.relu,
                                  kernel_initializer=initializer,
                                  bias_initializer=Zeros())
                self.mu_out = Dense(action_dim, activation=None,
                                    kernel_initializer=Orthogonal(0.01),
                                    bias_initializer=Zeros())

                self.sigma = tf.Variable(initial_value=tf.zeros(action_dim), trainable=True)

                self.v_0 = Dense(256, activation=tf.nn.relu,
                                 kernel_initializer=initializer,
                                 bias_initializer=Zeros())
                self.v_1 = Dense(128, activation=tf.nn.relu,
                                 kernel_initializer=initializer,
                                 bias_initializer=Zeros())
                self.v_2 = Dense(64, activation=tf.nn.relu,
                                 kernel_initializer=initializer,
                                 bias_initializer=Zeros())
                self.value = Dense(1, activation=None,
                                   kernel_initializer=Orthogonal(1),
                                   bias_initializer=Zeros())

            @tf.function
            def call(self, inputs):
                x = self.mu_0(inputs)
                x = self.mu_1(x)
                x = self.mu_2(x)
                mu = self.mu_out(x)

                y = self.v_0(inputs)
                y = self.v_1(y)
                y = self.v_2(y)
                va = self.value(y)
                return mu, self.sigma, va

        return MyMlpModel(self._action_dim)


class CnnGaussianActorCriticPolicyIndependentSigma(GaussianActorCriticPolicy):
    def __init__(self, state_dim, action_dim, action_space,
                 sigma_processing=log_sigma_processing,
                 action_handling=no_action_handling):
        super().__init__(state_dim, action_dim, action_space, sigma_processing, action_handling)

    def create_actor_critic_network(self):
        class MyCnnModel(keras.Model):

            def __init__(self, action_dim):
                super(MyCnnModel, self).__init__()
                initializer = Orthogonal(np.sqrt(2), seed=1)
                # self.x_0 = Rescaling(scale=1.0 / 127.5, offset=-1)
                self.x_0 = Rescaling(scale=1.0 / 255.0)
                self.x_1 = Conv2D(filters=32, kernel_size=8, strides=4, padding="valid", activation=tf.nn.relu,
                                  kernel_initializer=initializer,
                                  bias_initializer=Zeros())
                self.x_2 = Conv2D(filters=64, kernel_size=4, strides=2, padding="valid", activation=tf.nn.relu,
                                  kernel_initializer=initializer,
                                  bias_initializer=Zeros())
                self.x_3 = Conv2D(filters=64, kernel_size=3, strides=1, padding="valid", activation=tf.nn.relu,
                                  kernel_initializer=initializer,
                                  bias_initializer=Zeros())
                self.x_4 = Flatten()
                self.x_5 = Dense(512, activation=tf.nn.relu,
                                 kernel_initializer=initializer,
                                 bias_initializer=Zeros())

                self.y = Dense(512, activation=tf.nn.relu,
                               kernel_initializer=initializer,
                               bias_initializer=Zeros())
                self.value = Dense(1, activation=None,
                                   kernel_initializer=Orthogonal(1),
                                   bias_initializer=Zeros())

                self.z = Dense(512, activation=tf.nn.relu,
                               kernel_initializer=initializer,
                               bias_initializer=Zeros())
                self.mu = Dense(action_dim, activation=None,
                                kernel_initializer=Orthogonal(0.01),
                                bias_initializer=Zeros())

                self.sigma = tf.Variable(initial_value=tf.zeros(action_dim), trainable=True)

            @tf.function
            def call(self, inputs):
                x = self.x_0(inputs)
                x = self.x_1(x)
                x = self.x_2(x)
                x = self.x_3(x)
                x = self.x_4(x)
                x = self.x_5(x)

                y = self.y(x)
                value = self.value(y)

                z = self.z(x)
                mu = self.mu(z)
                return mu, self.sigma, value

        return MyCnnModel(self._action_dim)


class LstmGaussianActorCriticPolicyIndependentSigma(GaussianActorCriticPolicy):
    def create_actor_critic_network(self):
        class MyMlpModel(keras.Model):

            def __init__(self, action_dim):
                super(MyMlpModel, self).__init__()

                initializer_lstm = Orthogonal(1, seed=1)
                initializer_mlp = Orthogonal(np.sqrt(2), seed=2)

                self.mu_0 = LSTM(256, return_sequences=True,
                                 kernel_initializer=initializer_lstm,
                                 bias_initializer=Zeros())
                self.mu_1 = LSTM(256,
                                 kernel_initializer=initializer_lstm,
                                 bias_initializer=Zeros())
                self.mu_2 = Dense(256, activation=tf.nn.relu,
                                  kernel_initializer=initializer_mlp,
                                  bias_initializer=Zeros())
                self.mu_out = Dense(action_dim, activation=None,
                                    kernel_initializer=Orthogonal(0.01),
                                    bias_initializer=Zeros())

                self.sigma = tf.Variable(initial_value=tf.zeros(action_dim), trainable=True)

                self.v_0 = LSTM(256, return_sequences=True,
                                kernel_initializer=initializer_lstm,
                                bias_initializer=Zeros())
                self.v_1 = LSTM(256,
                                kernel_initializer=initializer_lstm,
                                bias_initializer=Zeros())
                self.v_2 = Dense(256, activation=tf.nn.relu,
                                 kernel_initializer=initializer_mlp,
                                 bias_initializer=Zeros())
                self.value = Dense(1, activation=None,
                                   kernel_initializer=Orthogonal(1),
                                   bias_initializer=Zeros())

            @tf.function
            def call(self, inputs):
                x = self.mu_0(inputs)
                x = self.mu_1(x)
                x = self.mu_2(x)
                mu = self.mu_out(x)

                y = self.v_0(inputs)
                y = self.v_1(y)
                y = self.v_2(y)
                va = self.value(y)
                return mu, self.sigma, va

        return MyMlpModel(self._action_dim)
