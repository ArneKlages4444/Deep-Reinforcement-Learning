from functools import partial
import gym
import tensorflow as tf

from Networks.LunaLanderNetwork import create_q_network
from DQNAgent import Agent

if __name__ == '__main__':
    tf.keras.backend.clear_session()
    env = gym.make("LunarLander-v2")
    print("state_dim=", env.observation_space.shape, "action_dim=", env.action_space)

    agent = Agent(environment=env, state_dim=env.observation_space.shape, action_dim=4,
                  q_network_generator=partial(create_q_network, state_dim=env.observation_space.shape[0], action_dim=4),
                  batch_size=256)
    agent.train(10000)
