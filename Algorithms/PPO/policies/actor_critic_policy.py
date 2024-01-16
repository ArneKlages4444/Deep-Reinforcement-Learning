class ActorCriticPolicy:
    def __init__(self, state_dim, action_dim):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.network = self.create_actor_critic_network()

    # return prob_current_policy, entropy, value
    def __call__(self, s, a):
        pass

    # return value
    def get_value(self, state):
        pass

    # return actions_prime, log_probs, value
    def act_stochastic(self, state):
        pass

    # return actions
    def act_deterministic(self, state):
        pass
