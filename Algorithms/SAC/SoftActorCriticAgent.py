from ExperienceReplayBuffer import ExperienceReplayBuffer
import tensorflow as tf
from tensorflow import math as tfm
from tensorflow_probability import distributions as tfd
import numpy as np
import datetime


# input actions are always between (−1, 1)
def default_scaling(actions):
    return actions


# input actions are always between (−1, 1)
def multiplicative_scaling(actions, factors):
    return actions * factors


class Agent:
    """
    Soft Actor-Critic (SAC) Agent
    Based on pseudocode of OpenAI Spinning Up (2022) (https://spinningup.openai.com/en/latest/algorithms/sac.html)
    and the Paper of Haarnoja et al. (2018) (https://arxiv.org/abs/1801.01290)

    :param environment: The environment to learn from (conforming to the gymnasium specifics https://gymnasium.farama.org/)
    :param state_dim: Dimensions of the state space
    :param action_dim: Dimensions of the action space
    :param actor_network_generator: a generator function for the actor network
        signature: learning_rate, state_dim, action_dim -> tensorflow Model
    :param critic_network_generator: a generator function for the critic networks (Q-networks)
        signature: learning_rate, state_dim, action_dim -> tensorflow Model
    :param action_scaling=default_scaling: function to scale the actions form (-1, 1)
        to the range the environment requires
        signature:  (action_tensor -> scaled_action_tensor)
    :param learning_rate=0.0003: Learning rate for adam optimizer.
        The same learning rate will be used for all networks (Q-Values, Actor)
    :param gamma=0.99: discount factor
    :param tau=0.005:  Polyak averaging coefficient (between 0 and 1)
    :param alpha=0.2: Entropy regularization coefficient (between 0 and 1).
        Controlling the exploration/exploitation trade off.
        (inverse of reward scale in the original SAC paper)
    :param batch_size=256: Minibatch size for each gradient update
    :param max_replay_buffer_size=1000000: Size of the replay buffer
    :param model_path: path to the location the model is saved and loaded from
    """

    def __init__(self, environment, state_dim, action_dim,
                 actor_network_generator, critic_network_generator, action_scaling=default_scaling,
                 learning_rate=0.0003, gamma=0.99, tau=0.005, reward_scale=1, alpha=0.2,
                 batch_size=256, max_replay_buffer_size=1000000, model_path=""):
        self._environment = environment
        self._action_dim = action_dim
        self._action_scaling = action_scaling
        self._gamma = gamma
        self._tau = tau
        self._reward_scale = reward_scale
        self._alpha = alpha
        self._batch_size = batch_size
        self._mse = tf.keras.losses.MeanSquaredError()
        self._model_path = model_path
        self._reply_buffer = ExperienceReplayBuffer(state_dim, action_dim, max_replay_buffer_size, batch_size)
        self._actor = actor_network_generator(learning_rate)
        self._critic_1 = critic_network_generator(learning_rate)
        self._critic_2 = critic_network_generator(learning_rate)
        self._critic_1_t = critic_network_generator(learning_rate)
        self._critic_2_t = critic_network_generator(learning_rate)
        self._wight_init()

    def reply_buffer(self):
        return self._reply_buffer

    def environment(self):
        return self._environment

    def save_models(self, name):
        self._actor.save_weights(f"{self._model_path}actor{name}")
        self._critic_1.save_weights(f"{self._model_path}critic_1{name}")
        self._critic_2.save_weights(f"{self._model_path}critic_2{name}")
        self._critic_1_t.save_weights(f"{self._model_path}critic_1_t{name}")
        self._critic_2_t.save_weights(f"{self._model_path}critic_2_t{name}")

    def load_models(self, name):
        self._actor.load_weights(f"{self._model_path}actor{name}")
        self._critic_1.load_weights(f"{self._model_path}critic_1{name}")
        self._critic_2.load_weights(f"{self._model_path}critic_2{name}")
        self._critic_1_t.load_weights(f"{self._model_path}critic_1{name}")
        self._critic_2_t.load_weights(f"{self._model_path}critic_2{name}")

    def _wight_init(self):
        self._critic_1.set_weights(self._critic_1_t.weights)
        self._critic_2.set_weights(self._critic_2_t.weights)

    def update_target_weights(self):
        self._weight_update(self._critic_1_t, self._critic_1)
        self._weight_update(self._critic_2_t, self._critic_2)

    def _weight_update(self, target_network, network):
        new_wights = []
        for w_t, w in zip(target_network.weights, network.weights):
            new_wights.append((1 - self._tau) * w_t + self._tau * w)
        target_network.set_weights(new_wights)

    def learn(self):
        states, actions, rewards, states_prime, dones = self._reply_buffer.sample_batch()
        self.train_step_critic(states, actions, rewards, states_prime, dones)
        self.train_step_actor(states)
        self.update_target_weights()

    @tf.function
    def train_step_critic(self, states, actions, rewards, states_prime, dones):
        actions_prime, log_probs = self.sample_actions_form_policy(states_prime)
        q1 = self._critic_1_t((states_prime, actions_prime))
        q2 = self._critic_2_t((states_prime, actions_prime))
        q_r = tfm.minimum(q1, q2) - self._alpha * log_probs
        targets = self._reward_scale * rewards + self._gamma * (1 - dones) * q_r
        self._critic_update(self._critic_1, states, actions, targets)
        self._critic_update(self._critic_2, states, actions, targets)

    def _critic_update(self, critic, states, actions, targets):
        with tf.GradientTape() as tape:
            q = critic((states, actions))
            loss = 0.5 * self._mse(targets, q)
        gradients = tape.gradient(loss, critic.trainable_variables)
        critic.optimizer.apply_gradients(zip(gradients, critic.trainable_variables))

    @tf.function
    def train_step_actor(self, states):
        with tf.GradientTape() as tape:
            actions_new, log_probs = self.sample_actions_form_policy(states)
            q1 = self._critic_1((states, actions_new))
            q2 = self._critic_2((states, actions_new))
            loss = tfm.reduce_mean(self._alpha * log_probs - tfm.minimum(q1, q2))
            # equal to loss = -tfm.reduce_mean(tfm.minimum(q1, q2) - self._alpha * log_probs)
        gradients = tape.gradient(loss, self._actor.trainable_variables)
        self._actor.optimizer.apply_gradients(zip(gradients, self._actor.trainable_variables))

    @tf.function
    def sample_actions_form_policy(self, state):
        mu, sigma = self._actor(state)
        # MultivariateNormalDiag(loc=mus, scale_diag=sigmas) other option
        distribution = tfd.Normal(mu, sigma)
        actions = distribution.sample()
        log_probs = distribution.log_prob(actions)
        actions = tfm.tanh(actions)
        log_probs -= tfm.log(1 - tfm.pow(actions, 2) + 1e-6)  # + 1e-6 because log undefined for 0
        log_probs = tfm.reduce_sum(log_probs, axis=-1, keepdims=True)
        return actions, log_probs

    def act_deterministic(self, state):
        actions_prime, _ = self._actor(tf.convert_to_tensor([state], dtype=tf.float32))
        return self._act(actions_prime)

    def act_stochastic(self, state):
        actions_prime, _ = self.sample_actions_form_policy(tf.convert_to_tensor([state], dtype=tf.float32))
        return self._act(actions_prime)

    def _act(self, actions):
        scaled_actions = self._action_scaling(actions)  # scaled actions from (-1, 1) according (to environment)
        observation_prime, reward, done, truncated, _ = self._environment.step(scaled_actions[0])
        return actions, observation_prime, reward, done or truncated

    def train(self, epochs, environment_steps_before_training=1, training_steps_per_update=1,
              max_environment_steps_per_epoch=None, pre_sampling_steps=1024, save_models=False):
        """
        trains the SAC Agent
        :param epochs: Number of epochs to train.
            One epoch is finished if the agents are done ore the maximum steps are reached
        :param environment_steps_before_training=1: Number of steps the agent takes in the environment
            before the training cycle starts (the networks are updated after each step in the environment)
        :param training_steps_per_update=1: Number of times the networks are update per update cykle
        :param max_environment_steps_per_epoch=None: Maximal number of staps taken in the environment in an epoch
        :param pre_sampling_steps=1024: Number of exploration steps sampled to the replay buffer before training starts
        :param save_models=False: Determines if the models are saved per epoch
        """
        print(f"Random exploration for {pre_sampling_steps} steps!")
        observation, _ = self._environment.reset()
        ret = 0
        for _ in range(pre_sampling_steps):
            actions, observation_prime, reward, done = self.act_stochastic(observation)
            ret += reward
            self._reply_buffer.add_transition(observation, actions, reward, observation_prime, done)
            if done:
                ret = 0
                observation, _ = self._environment.reset()
            else:
                observation = observation_prime
        print("start training!")
        returns = []
        observation, _ = self._environment.reset()
        done = 0
        ret = 0
        epoch = 0
        steps = 0
        j = 0
        while True:
            i = 0
            while i < environment_steps_before_training or self._reply_buffer.size() < self._batch_size:
                if done or (max_environment_steps_per_epoch is not None and j >= max_environment_steps_per_epoch):
                    observation, _ = self._environment.reset()
                    returns.append(ret)
                    print("epoch:", epoch, "steps:", steps, "return:", ret, "avg return:", np.average(returns[-4:]))
                    if save_models:
                        self.save_models(f"SAC_{epoch}_{steps}_{ret}_{datetime.datetime.now()}")
                    ret = 0
                    epoch += 1
                    if epoch >= epochs:
                        print("training finished!")
                        return
                actions, observation_prime, reward, done = self.act_stochastic(observation)
                self._reply_buffer.add_transition(observation, actions, reward, observation_prime, done)
                observation = observation_prime
                steps += 1
                ret += reward
                i += 1
                j += 1
            for _ in range(training_steps_per_update):
                self.learn()
