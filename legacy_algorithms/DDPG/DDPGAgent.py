from ExperienceReplayBuffer import ExperienceReplayBuffer
import tensorflow as tf
from tensorflow import math as tfm
from tensorflow_probability import distributions as tfd
import numpy as np


# input should be between (−1, 1)
def default_scaling(actions):
    return actions


# input should be between (−1, 1)
def multiplicative_scaling(actions, factors):
    return actions * factors


class Agent:
    def __init__(self, environment, state_dim, action_dim,
                 actor_network_generator, critic_network_generator, action_scaling=default_scaling,
                 learning_rate=0.0003, gamma=0.99, tau=0.005,
                 exploration_noise=0.1, a_low=-1, a_high=1,
                 batch_size=256, max_replay_buffer_size=1000000):
        self._environment = environment
        self._action_dim = action_dim
        self._action_scaling = action_scaling
        self._gamma = gamma
        self._tau = tau
        self.exploration_noise = exploration_noise
        self.a_low = a_low
        self.a_high = a_high
        self._batch_size = batch_size
        self._mse = tf.keras.losses.MeanSquaredError()
        self._reply_buffer = ExperienceReplayBuffer(state_dim, action_dim, max_replay_buffer_size, batch_size)
        self._actor = actor_network_generator(learning_rate)
        self._actor_t = actor_network_generator(learning_rate)
        self._critic = critic_network_generator(learning_rate)
        self._critic_t = critic_network_generator(learning_rate)
        self._wight_init()

    def reply_buffer(self):
        return self._reply_buffer

    def environment(self):
        return self._environment

    def _wight_init(self):
        self._actor.set_weights(self._actor_t.weights)
        self._critic.set_weights(self._critic_t.weights)

    def update_target_weights(self):
        self._weight_update(self._actor_t, self._actor)
        self._weight_update(self._critic_t, self._critic)

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
        actions_prime = self._actor_t(states)
        q = self._critic_t((states_prime, actions_prime))
        targets = rewards + self._gamma * (1 - dones) * q
        with tf.GradientTape() as tape:
            q = self._critic((states, actions))
            loss = self._mse(targets, q)
        gradients = tape.gradient(loss, self._critic.trainable_variables)
        self._critic.optimizer.apply_gradients(zip(gradients, self._critic.trainable_variables))

    @tf.function
    def train_step_actor(self, states):
        with tf.GradientTape() as tape:
            actions_new = self._actor(states)
            q = self._critic((states, actions_new))
            loss = -tfm.reduce_mean(q)
        gradients = tape.gradient(loss, self._actor.trainable_variables)
        self._actor.optimizer.apply_gradients(zip(gradients, self._actor.trainable_variables))

    def _action_clipping(self, actions):
        return tf.clip_by_value(actions, self.a_low, self.a_high)

    def sample_actions_form_policy(self, state):
        actions = self._actor(state)
        # or noise from sampling form tfp normal distribution with a sigma vector to get different noise per action
        noise = tf.random.normal(actions.get_shape(), 0, self.exploration_noise)
        clip_actions = self._action_clipping(actions + noise)
        return clip_actions

    def act_deterministic(self, state):
        actions_prime = self._actor(tf.convert_to_tensor([state], dtype=tf.float32))
        return self._act(actions_prime)

    def act_stochastic(self, state):
        actions_prime = self.sample_actions_form_policy(tf.convert_to_tensor([state], dtype=tf.float32))
        return self._act(actions_prime)

    def _act(self, actions):
        scaled_actions = self._action_scaling(actions)  # scaled actions from (-1, 1) according (to environment)
        observation_prime, reward, terminated, truncated, _ = self._environment.step(scaled_actions[0])
        return actions, observation_prime, reward, terminated or truncated

    def train(self, epochs, environment_steps=1, training_steps=1, pre_sampling_steps=0):
        print(f"Random exploration for {pre_sampling_steps} steps!")
        observation, _ = self._environment.reset()
        ret = 0
        for _ in range(max(pre_sampling_steps, self._batch_size)):
            actions = tf.random.uniform((self._action_dim,), minval=self.a_low, maxval=self.a_high)
            actions = self._action_scaling(actions)
            observation_prime, reward, done, _, _ = self._environment.step(actions)
            ret += reward
            self._reply_buffer.add_transition(observation, actions, reward, observation_prime, done)
            if done:
                print("print", ret)
                ret = 0
                observation, _ = self._environment.reset()
            else:
                observation = observation_prime
        print("print", ret)

        print("start training!")
        returns = []
        observation, _ = self._environment.reset()
        done = 0
        ret = 0
        epoch = 0
        steps = 0
        while True:
            i = 0
            while i < environment_steps or self._reply_buffer.size() < self._batch_size:
                if done:
                    observation, _ = self._environment.reset()
                    returns.append(ret)
                    print("epoch:", epoch, "steps:", steps, "return:", ret, "avg return:", np.average(returns[-50:]))
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
            for _ in range(training_steps):
                self.learn()
