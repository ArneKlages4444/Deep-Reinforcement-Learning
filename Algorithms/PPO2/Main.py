import tensorflow as tf
from Agent import Agent
from GenericMLPs1D import create_policy_network, create_value_network
import gym
from functools import partial

if __name__ == '__main__':
    tf.keras.backend.clear_session()
    env = gym.make('InvertedPendulum-v4')
    print("state_dim=", env.observation_space.shape, "action_dim=", env.action_space.shape[0], "action_scaling:",
          env.action_space.high)
    agent = Agent(environment=env,
                  actor_network_generator=partial(create_policy_network, state_dim=env.observation_space.shape[0],
                                                  action_dim=env.action_space.shape[0]),
                  critic_network_generator=partial(create_value_network, state_dim=env.observation_space.shape))
    agent.train(epochs=1000, max_steps_per_episode=200)
