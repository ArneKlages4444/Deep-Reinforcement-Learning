import numpy as np
import pandas as pd


def final_goal_sampling_strategy(trajectory, current_index, environment):
    _, _, s_p, _ = trajectory[-1]
    g = environment.achieved_goal(s_p)
    return [g]


def k_final_goal_sampling_strategy(trajectory, current_index, environment, k=4):
    return final_goal_sampling_strategy(trajectory, current_index, environment) * k


def future_goal_sampling_strategy(trajectory, current_index, environment, k=4):
    goals = []
    for _ in range(k):
        i = np.random.randint(current_index, len(trajectory))
        _, _, s_p, _ = trajectory[i]
        goals.append(environment.achieved_goal(s_p))
    return goals


def no_goal_sampling_strategy(trajectory, current_index, environment):
    return []


class HindsightExperienceReplayBuffer:

    def __init__(self, agent, goal_sampling_strategy=final_goal_sampling_strategy):
        self._agent = agent
        self._goal_sampling_strategy = goal_sampling_strategy
        self._replay_buffer = agent.reply_buffer()
        self._environment = self._agent.environment()

    def evaluate(self, steps, epoch, successes, avg_returns):
        success_cnt = 0
        rets = []
        for _ in range(steps):
            state = self._environment.reset()
            done = False
            ret = 0
            while not done:
                _, state, reward, done = self._agent.act_deterministic(state)
                ret += self._environment.reward(state)
                if self._environment.success(state):
                    success_cnt += 1
                    done = True
            rets.append(ret)
        avg_return = np.average(rets)
        success_rate = success_cnt / steps
        successes.append(success_rate)
        avg_returns.append(avg_return)
        print(f"epoch {epoch}: avg return={avg_return}, success rate={success_rate} (with {steps} evaluation steps)")

    def train(self, epochs=200, cycles=50, episodes=16, n=40, t=1000,
              eval_steps=1000, save_eval=False, eval_name='evaluation'):
        successes = []
        avg_returns = []
        self.evaluate(eval_steps, 0, successes, avg_returns)
        for e in range(1, epochs + 1):
            for _ in range(cycles):
                for _ in range(episodes):
                    state = self._environment.reset()
                    trajectory = []
                    dones = 0
                    j = 0
                    while not dones and j < t:
                        actions, state_prime, r, dones = self._agent.act_stochastic(state)
                        trajectory.append((state, actions, state_prime, dones))
                        state = state_prime
                        j += 1
                        if self._environment.success(state):
                            dones = True
                    for i, (state, actions, state_prime, dones) in enumerate(trajectory):
                        reward = self._environment.reward(state_prime)
                        self._replay_buffer.add_transition(state, actions, reward, state_prime, dones)
                        goals = self._goal_sampling_strategy(trajectory, i, self._environment)
                        for g in goals:
                            state_new = self._environment.set_goal(state, g)
                            state_prime_new = self._environment.set_goal(state_prime, g)
                            reward_new = self._environment.reward(state_prime_new)
                            self._replay_buffer.add_transition(state_new, actions, reward_new, state_prime_new, dones)
                if self._replay_buffer.ready():
                    for i in range(n):
                        self._agent.learn()
            self.evaluate(eval_steps, e, successes, avg_returns)
        if save_eval:
            data = {'epoch': range(epochs + 1), 'success rate': successes, 'average return': avg_returns}
            df = pd.DataFrame.from_dict(data)
            df.to_csv(f'{eval_name}.csv')
