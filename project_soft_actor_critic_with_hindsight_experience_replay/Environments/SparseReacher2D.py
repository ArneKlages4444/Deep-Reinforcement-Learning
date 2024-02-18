import gym
import numpy as np


class SparseReacher2D:

    def __init__(self, delta=0.015):
        self._env = gym.make("Reacher-v4")
        self._delta = delta

    def observation_space_shape(self):
        return self._env.observation_space.shape

    def environment(self):
        return self._env

    def step(self, actions):
        return self._env.step(actions)

    def reset(self):
        return self._env.reset()

    # reacher gives the vector from fingertip to target instead of fingertip coordinates
    # we need to extract the achieved goal out of this vector
    def achieved_goal(self, state):
        x_t, y_t = self.desired_goal(state)
        x_v, y_v = state[8], state[9]  # fingertip to target vector
        x_g, y_g = x_v + x_t, y_v + y_t  # fingertip coordinates
        return x_g, y_g

    def desired_goal(self, state):
        return state[4], state[5]

    def set_goal(self, state, goal):
        x_g, y_g = goal
        x, y = self.achieved_goal(state)
        new_state = np.array(state)
        new_state[4], new_state[5] = x_g, y_g
        new_state[8], new_state[9] = x - x_g, y - y_g  # create fingertip to goal vector
        return new_state

    # check if distance between goal and fingertip is lower than epsilon
    def reward(self, state):
        return -1 if np.linalg.norm([state[8], state[9]]) > self._delta else 0

    def success(self, state):
        return self.reward(state) >= -0.0
