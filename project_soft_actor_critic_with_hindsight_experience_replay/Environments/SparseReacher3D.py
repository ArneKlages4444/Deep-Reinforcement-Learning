import gym
import numpy as np
import panda_gym


class SparseReacher3D:

    def __init__(self):
        self._env = gym.make("PandaReach-v2")

    def observation_space_shape(self):
        return (9,)

    def environment(self):
        return self._env

    def _select_observations(self, observation):
        return np.concatenate((observation['observation'], observation['desired_goal']))

    def step(self, actions):
        observation, reward, done, info = self._env.step(actions.numpy())  # converting to numpy before would be better
        return self._select_observations(observation), reward, done, info

    def reset(self):
        return self._select_observations(self._env.reset())

    def achieved_goal(self, state):
        return np.array([state[0], state[1], state[2]])

    def desired_goal(self, state):
        return np.array([state[6], state[7], state[8]])

    def set_goal(self, state, goal):
        new_state = np.array(state)
        new_state[6], new_state[7], new_state[8] = goal
        return new_state

    def reward(self, state):
        return self._env.compute_reward(self.achieved_goal(state), self.desired_goal(state), {})

    def success(self, state):
        return self.reward(state) >= -0.0
