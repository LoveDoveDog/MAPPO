import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical

# Do not modify this program

################################## PPO Policy ##################################
class RolloutBuffer:
    def __init__(self):
        self.actions = []
        self.states_for_ppo = []
        self.logprobs = []
        self.rewards = []
        self.states_for_common = []
    
    def clear(self):
        del self.actions[:]
        del self.states_for_ppo[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.states_for_common[:]


class ActorCritic(nn.Module):
    def __init__(self, state_dim_critic, state_dim_actor, action_dim, actor_node, critic_node):
        super(ActorCritic, self).__init__()
        self.action_dim = action_dim
        self.actor = nn.Sequential(
                        nn.Linear(state_dim_actor, actor_node),
                        nn.Tanh(),
                        nn.Linear(actor_node, actor_node),
                        nn.Tanh(),
                        nn.Linear(actor_node, action_dim),
                        nn.Softmax(dim=-1)
                    )
        self.critic = nn.Sequential(
                        nn.Linear(state_dim_critic, critic_node),
                        nn.Tanh(),
                        nn.Linear(critic_node, critic_node),
                        nn.Tanh(),
                        nn.Linear(critic_node, 1)
                    )
    
    def act(self, state_for_ppo):
        action_probs = self.actor(state_for_ppo)
        dist = Categorical(action_probs)
        action = dist.sample()
        action_logprob = dist.log_prob(action)
        return action.detach(), action_logprob.detach()
    
    def evaluate(self, state_for_ppo, state_for_common, action):
        action_probs = self.actor(state_for_ppo)
        dist = Categorical(action_probs)
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_for_common_values = self.critic(state_for_common)
        return action_logprobs, state_for_common_values, dist_entropy


class PPO:
    def __init__(self, state_dim_critic, state_dim_actor, action_dim, actor_node, critic_node, lr_actor, lr_critic, gamma, K_epochs, eps_clip, device):
        self.device = device
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs        
        self.buffer = RolloutBuffer()
        self.policy = ActorCritic(state_dim_critic, state_dim_actor, action_dim, actor_node, critic_node).to(self.device)
        self.optimizer = torch.optim.Adam([
                        {'params': self.policy.actor.parameters(), 'lr': lr_actor},
                        {'params': self.policy.critic.parameters(), 'lr': lr_critic}
                    ])
        self.policy_old = ActorCritic(state_dim_critic, state_dim_actor, action_dim, actor_node, critic_node).to(self.device)
        self.policy_old.load_state_dict(self.policy.state_dict())       
        self.MseLoss = nn.MSELoss()

    def select_action(self, state_for_ppo):
        with torch.no_grad():
            state_for_ppo = torch.FloatTensor(state_for_ppo).to(self.device)
            action, action_logprob = self.policy_old.act(state_for_ppo)
        action = action.to(self.device)
        action_logprob = action_logprob.to(self.device)            
        self.buffer.states_for_ppo.append(state_for_ppo)
        self.buffer.actions.append(action)
        self.buffer.logprobs.append(action_logprob)
        return action.item()

    def update(self):
        # Monte Carlo estimate of returns
        rewards = []
        discounted_reward = 0
        for reward in reversed(self.buffer.rewards):
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)           
        # Normalizing the rewards
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)
        # convert list to tensor
        old_states_for_ppo = torch.stack(self.buffer.states_for_ppo, dim=0).detach().to(self.device)
        old_states_for_common = torch.stack(self.buffer.states_for_common, dim=0).detach().to(self.device)
        old_actions = torch.squeeze(torch.stack(self.buffer.actions, dim=0)).detach().to(self.device)
        old_logprobs = torch.squeeze(torch.stack(self.buffer.logprobs, dim=0)).detach().to(self.device)
        # Optimize policy for K epochs
        for _ in range(self.K_epochs):
            # Evaluating old actions and values
            logprobs, states_for_common_values, dist_entropy = self.policy.evaluate(old_states_for_ppo, old_states_for_common, old_actions)
            # match states_for_common_values tensor dimensions with rewards tensor
            states_for_common_values = torch.squeeze(states_for_common_values)           
            # Finding the ratio (pi_theta / pi_theta__old)
            ratios = torch.exp(logprobs - old_logprobs.detach())
            # Finding Surrogate Loss
            advantages = rewards - states_for_common_values.detach()   
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages
            # final loss of clipped objective PPO
            loss = -torch.min(surr1, surr2) + 0.5*self.MseLoss(states_for_common_values, rewards) - 0.01*dist_entropy            
            # take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()            
        # Copy new weights into old policy
        self.policy_old.load_state_dict(self.policy.state_dict())
        # clear buffer
        self.buffer.clear()
    
    def save(self, model_path_and_name):
        torch.save(self.policy_old.state_dict(), model_path_and_name)
   
    def load(self, model_path_and_name):
        self.policy_old.load_state_dict(torch.load(model_path_and_name, map_location=lambda storage, loc: storage))
        self.policy.load_state_dict(torch.load(model_path_and_name, map_location=lambda storage, loc: storage))
        