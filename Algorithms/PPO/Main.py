from Agent import Agent
import gymnasium as gym
from functools import partial
import tensorflow as tf
from gymnasium.wrappers import NormalizeObservation

# from generic_LSTMs import create_policy_network, create_value_network
from GenericMLPs1D import create_policy_network, create_value_network

env = gym.make("InvertedPendulum-v4")

env = NormalizeObservation(env)
if __name__ == '__main__':
    tf.keras.backend.clear_session()
    env = gym.make('InvertedPendulum-v4')
    norm_obs = False
    if norm_obs:
        env = NormalizeObservation(env)
    print("state_dim=", env.observation_space.shape, "action_dim=", env.action_space.shape[0], "action_scaling:",
          env.action_space.high)
    agent = Agent(environments=env,
                  actor_network_generator=partial(create_policy_network, state_dim=env.observation_space.shape,
                                                  action_dim=env.action_space.shape[0]),
                  critic_network_generator=partial(create_value_network, state_dim=env.observation_space.shape),
                  window_size=None)

    agent.train(epochs=1000, batch_size=64, sub_epochs=4, steps_per_trajectory=640)
