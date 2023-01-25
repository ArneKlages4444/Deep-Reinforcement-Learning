from functools import partial
import gym
import tensorflow as tf
# import pybullet_envs

from Networks.InvertedPendulumNetwork import create_policy_network
from ReinforceAgent import Agent

if __name__ == '__main__':
    tf.keras.backend.clear_session()
    env = gym.make("InvertedPendulum-v4")
    print("state_dim=", env.observation_space.shape, "action_dim=", env.action_space.shape[0], "action_scaling:",
          env.action_space.high)

    agent = Agent(environment=env,
                  policy_network_generator=partial(create_policy_network, state_dim=env.observation_space.shape[0],
                                                   action_dim=env.action_space.shape[0]))
    agent.train(10000)
