from functools import partial
import tensorflow as tf

from final_project.Networks.GenericMLPs1D import create_policy_network, create_q_network
from SoftActorCriticAgent import Agent, multiplicative_scaling
from Environments.SparseReacher3D import SparseReacher3D
from HER import HindsightExperienceReplayBuffer, future_goal_sampling_strategy

if __name__ == '__main__':
    tf.keras.backend.clear_session()
    environment = SparseReacher3D()
    env = environment.environment()
    state_dim = environment.observation_space_shape()
    action_dim = env.action_space.shape[0]
    action_scaling = env.action_space.high
    print("state_dim=", state_dim, "action_dim=", action_dim, "action_scaling:", action_scaling)
    agent = Agent(environment=environment, state_dim=state_dim, action_dim=action_dim, alpha=0.05,
                  action_scaling=partial(multiplicative_scaling, factors=action_scaling),
                  actor_network_generator=partial(create_policy_network, state_dim=state_dim[0], action_dim=action_dim),
                  critic_network_generator=partial(create_q_network, state_dim=state_dim[0], action_dim=action_dim))
    her = HindsightExperienceReplayBuffer(agent, goal_sampling_strategy=future_goal_sampling_strategy)
    her.train(epochs=40)
