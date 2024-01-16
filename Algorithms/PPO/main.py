import datetime
import gymnasium as gym
import tensorflow as tf
from gymnasium.wrappers import FrameStack

from agent import Agent
from policies.default_policies import MlpGaussianActorCriticPolicy

env_name = "InvertedPendulum-v4"
num_envs = 4
window_size = None  # 8
network_type = "mlp"
log_dir = f'logs/{env_name}/PPO_{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}'


def main():
    tf.keras.backend.clear_session()

    if window_size is not None:
        envs = [lambda: FrameStack(gym.make(env_name), window_size) for _ in range(num_envs)]
        env = gym.vector.SyncVectorEnv(envs)  # oder: env = gym.vector.AsyncVectorEnv(envs)
    else:
        env = gym.vector.make(env_name, num_envs=num_envs, asynchronous=False)

    if network_type == "cnn":
        raise Exception(f"Unknown network type {network_type}")
    elif network_type == "rnn":
        raise Exception(f"Unknown network type {network_type}")
    elif network_type == "mlp":
        policy = MlpGaussianActorCriticPolicy(action_dim=env.single_action_space.shape[0],
                                              state_dim=env.single_observation_space.shape)
    else:
        raise Exception(f"Unknown network type {network_type}")

    agent = Agent(
        environments=env,
        policy=policy,
        normalize_adv=False,
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
