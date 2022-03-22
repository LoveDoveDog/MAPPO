import numpy as np

# a simple example, three agents have reward equaling the minus of their observations
# the goal is to maximize the sum rewards.

class bigenv(object):
    def __init__(self, args):
        self.agent_num = args.agent_num
        self.state = np.array([0,0,0]) # the state has to be a horizontial numpy vector
        self.state_dim_critic = 3 # dim for critic is also the dim for state
        self.state_dim_actor = 1 # dim for actor is also the dim for observation
        self.action_dim = 2

    def step(self, action):
        reward_list = np.zeros(self.agent_num+1) # reward_list has to be a horizontial numpy vector
        reward_list[0:-1] = -self.state
        reward_list[-1] = -np.sum(self.state) # the last term is the sum reward among agents

        for i in np.arange(self.agent_num):
            if action[i] == 0:
                self.state[i]+=1
            else:
                self.state[i]=0

        return self.state, reward_list

    def reset(self):
        self.state = np.array([0,0,0])
        return self.state
