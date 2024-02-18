from ExperienceReplayBuffer import ExperienceReplayBuffer
import tensorflow as tf
from tensorflow import math as tfm
from tensorflow_probability import distributions as tfd
import numpy as np


class Agent:
    def __init__(self, environment, state_dim, action_dim, q_network_generator,
                 learning_rate=0.0003, gamma=0.99, tau=0.005,
                 epsilon=1, epsilon_decay=0.99, min_epsilon=0.05,
                 batch_size=256, max_replay_buffer_size=1000000):
        self._environment = environment
        self._action_dim = action_dim
        self._gamma = gamma
        self._tau = tau
        self._epsilon = epsilon
        self._epsilon_decay = epsilon_decay
        self._min_epsilon = min_epsilon
        self._batch_size = batch_size
        self._mse = tf.keras.losses.MeanSquaredError()
        self._reply_buffer = ExperienceReplayBuffer(state_dim, action_dim, max_replay_buffer_size, batch_size)
        self._q_network = q_network_generator(learning_rate)
        self._q_network_t = q_network_generator(learning_rate)
        self._wight_init()

    def reply_buffer(self):
        return self._reply_buffer

    def environment(self):
        return self._environment

    def _wight_init(self):
        self._q_network_t.set_weights(self._q_network.weights)

    def update_target_weights(self):
        self._weight_update(self._q_network_t, self._q_network)

    def _weight_update(self, target_network, network):
        new_wights = []
        for w_t, w in zip(target_network.weights, network.weights):
            new_wights.append((1 - self._tau) * w_t + self._tau * w)
        target_network.set_weights(new_wights)

    def learn(self):
        states, actions, rewards, states_prime, dones = self._reply_buffer.sample_batch()
        self.train_step(states, actions, rewards, states_prime, dones)
        self.update_target_weights()

    @tf.function
    def train_step(self, states, actions, rewards, states_prime, dones):
        q_values_prime = self._q_network_t(states_prime)
        max_q = tf.reduce_max(q_values_prime, axis=-1, keepdims=True)
        targets = rewards + self._gamma * (1 - dones) * max_q  # (1-d) : no q if done
        with tf.GradientTape() as tape:
            q_values = self._q_network(states)
            q_values_of_actions = tf.gather(q_values, actions, axis=-1, batch_dims=1)
            loss = self._mse(targets, q_values_of_actions)
        gradients = tape.gradient(loss, self._q_network.trainable_variables)
        self._q_network.optimizer.apply_gradients(zip(gradients, self._q_network.trainable_variables))

    # alternative but ugly
    def train_step2(self, state, action, rewards, state_prime, dones):
        q_values = self._q_network_t(state_prime)
        max_q = tf.reduce_max(q_values, axis=-1).numpy()
        t = rewards + self._gamma * (1 - dones) * max_q  # (1-d) : no q if done

        t_batch = self._q_network(state).numpy()
        batch_index = np.arange(self._batch_size, dtype=np.int32)

        t_batch[batch_index, action] = t
        self._q_network.train_on_batch(state, t_batch)

    def sample_actions1(self, state):  # sample with e greedy policy, alternative would be Thompson sampling
        if np.random.random() <= self._epsilon:
            actions = tf.random.uniform((1,), minval=0, maxval=self._action_dim, dtype=tf.int32)
        else:
            actions = self._deterministic_action(state)
        self.epsilon = self._epsilon * self._epsilon_decay if self._epsilon > self._min_epsilon else self._min_epsilon
        return actions

    def sample_actions(self, state):
        q_values = self._q_network(state)
        distribution = tfd.Categorical(logits=q_values)
        return distribution.sample()

    def _deterministic_action(self, state):
        return tf.argmax(self._q_network(tf.convert_to_tensor(state, dtype=tf.float32)), axis=-1)

    def act_deterministic(self, state):
        actions_prime = self._deterministic_action(tf.convert_to_tensor(state, dtype=tf.float32))
        return self._act(actions_prime)

    def act_stochastic(self, state):
        actions_prime = self.sample_actions(tf.convert_to_tensor([state], dtype=tf.float32))
        return self._act(actions_prime)

    def _act(self, actions):
        observation_prime, reward, terminated, truncated, _ = self._environment.step(actions.numpy()[0])
        return actions, observation_prime, reward, terminated or truncated

    def train(self, epochs, environment_steps=1, training_steps=1, pre_sampling_steps=0):
        print(f"Random exploration for {pre_sampling_steps} steps!")
        observation, _ = self._environment.reset()
        ret = 0
        for _ in range(max(pre_sampling_steps, self._batch_size)):
            actions = tf.random.uniform((1,), minval=0, maxval=self._action_dim, dtype=tf.int32)
            observation_prime, reward, done, _, _ = self._environment.step(actions.numpy()[0])
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
