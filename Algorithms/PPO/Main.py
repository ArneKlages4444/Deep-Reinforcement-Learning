import datetime
from functools import partial

import gymnasium as gym
import tensorflow as tf
from gymnasium.wrappers import FrameStack

from Agent import Agent
import networks.GenericMLPs1D as mlp
import networks.GenericLSTMs as lstm

env_name = "InvertedPendulum-v4"
num_envs = 4
window_size = None  # 8
log_dir = f'logs/{env_name}/PPO_{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}'


def main():
    tf.keras.backend.clear_session()

    if window_size is not None:
        envs = [lambda: FrameStack(gym.make(env_name), window_size) for _ in range(num_envs)]
        env = gym.vector.SyncVectorEnv(envs)  # oder: env = gym.vector.AsyncVectorEnv(envs)
        policy_network = lstm.create_policy_network
        value_network = lstm.create_value_network
    else:
        env = gym.vector.make(env_name, num_envs=num_envs, asynchronous=False)
        policy_network = mlp.create_policy_network
        value_network = mlp.create_value_network

    actor_network_generator = partial(
        policy_network,
        state_dim=env.single_observation_space.shape,
        action_dim=env.single_action_space.shape[0],
    )
    critic_network_generator = partial(value_network, state_dim=env.single_observation_space.shape)

    agent = Agent(
        environments=env,
        actor_network_generator=actor_network_generator,
        critic_network_generator=critic_network_generator,
        log_dir=log_dir,
        verbose=True,
        num_envs=num_envs,
        batch_size=256,
        data_set_repeats=4,
        steps_per_epoch=2048
    )
    agent.train(epochs=100)


if __name__ == "__main__":
    main()
