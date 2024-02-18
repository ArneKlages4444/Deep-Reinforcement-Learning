from functools import partial
import gym
import tensorflow as tf

from GenericMLPs1D import create_policy_network, create_q_network
from DDPGAgent import Agent, multiplicative_scaling

if __name__ == '__main__':
    tf.keras.backend.clear_session()
    env = gym.make('InvertedPendulum-v4')
    print("state_dim=", env.observation_space.shape, "action_dim=", env.action_space.shape[0], "action_scaling:",
          env.action_space.high)

    agent = Agent(environment=env, state_dim=env.observation_space.shape, action_dim=env.action_space.shape[0],
                  action_scaling=partial(multiplicative_scaling, factors=env.action_space.high),
                  actor_network_generator=partial(create_policy_network, state_dim=env.observation_space.shape[0],
                                                  action_dim=env.action_space.shape[0]),
                  critic_network_generator=partial(create_q_network, state_dim=env.observation_space.shape[0],
                                                   action_dim=env.action_space.shape[0]))
    agent.train(10000)
