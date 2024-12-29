'''
The code is based on Phil Tabor Repository
https://youtu.be/hlv79rcHws0?si=NORRlRX8w3rPztC_
https://github.com/philtabor/Youtube-Code-Repository/blob/master/ReinforcementLearning/PolicyGradient/PPO/torch/ppo_torch.py
https://www.neuralnet.ai/a-crash-course-in-proximal-policy-optimization/

The Original paper from openAI
https://openai.com/index/openai-baselines-ppo/
https://arxiv.org/pdf/1707.06347
'''

import os
import numpy as np
import torch as T
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
import statistics as stat


class PPOMemory:
    def __init__(self, batch_size):
        self.states = []
        self.log_probs = []
        self.vals = []
        self.actions = []
        self.rewards = []
        self.dones = []

        self.batch_size = batch_size

    def generate_batches(self):
        n_states = len(self.states)
        batch_start = np.arange(0, n_states, self.batch_size)
        indices = np.arange(n_states, dtype=np.int64)
        np.random.shuffle(indices)
        batches = [indices[i:i+self.batch_size] for i in batch_start]

        return batches
    
    def get_arrays(self):
        return np.array(self.states),\
                np.array(self.actions),\
                np.array(self.log_probs),\
                np.array(self.vals),\
                np.array(self.rewards),\
                np.array(self.dones)
        

    def store_memory(self, state, action, log_probs, vals, reward, done):
        self.states.append(state)
        self.actions.append(action)
        self.log_probs.append(log_probs)
        self.vals.append(vals)
        self.rewards.append(reward)
        self.dones.append(done)

    def clear_memory(self):
        self.states = []
        self.log_probs = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.vals = []

class ActorNetwork(nn.Module):
    def __init__(self, input_dims, n_actions, lr, fc1_dims=256, fc2_dims=1024, chkpt=1, optim_step = 100, optim_gamma = 0.9):
        super(ActorNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dims, fc1_dims)
        self.fc2 = nn.Linear(fc1_dims, fc2_dims)
        self.fc3 = nn.Linear(fc2_dims,fc1_dims )
        self.fc4 = nn.Linear(fc1_dims, n_actions)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)
        self.checkpoint_file = f'Data/Actor{chkpt}.pth'
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=optim_step, gamma=optim_gamma)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        x = self.fc1(state)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.relu(x)
        x = self.fc4(x)
        dist = Categorical(logits=x)
        return dist

    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))

class CriticNetwork(nn.Module):
    def __init__(self, input_dims, lr, fc1_dims=256, fc2_dims=512, chkpt=1, optim_step = 100, optim_gamma = 0.9):
        super(CriticNetwork, self).__init__()

        self.checkpoint_file = f'Data/Critic{chkpt}.pth'
        self.fc1 = nn.Linear(input_dims, fc1_dims)
        self.fc2 = nn.Linear(fc1_dims, fc2_dims)
        self.fc3 = nn.Linear(fc2_dims, fc2_dims)
        self.fc4 = nn.Linear(fc2_dims, 1)
        self.relu = nn.ReLU()  
        
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=optim_step, gamma=optim_gamma)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        x = self.fc1(state)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.relu(x)
        value = self.fc4(x)
        return value

    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))

class PPO_Agent:
    def __init__(self, chkpt, input_dims=119, n_actions=4, logger=None):
        self.gamma = 0.99
        self.policy_clip = 0.2
        self.value_clip = 1  
        self.n_epochs = 5
        self.gae_lambda = 0.90
        self.entropy_coefficient = 0.05  
        self.max_grad_norm = 0.5  
        self.batch_size = 64
        self.lr_actor = 0.001
        self.lr_critic = 0.001
        self.optim_step = 5000
        self.optim_gamma = 0.9
        self.logger = logger
        
        self.actor = ActorNetwork(input_dims, n_actions, self.lr_actor, chkpt=chkpt, optim_step=self.optim_step, optim_gamma=self.optim_gamma)
        self.critic = CriticNetwork(input_dims, self.lr_critic, chkpt=chkpt, optim_step=self.optim_step, optim_gamma=self.optim_gamma)
        self.memory = PPOMemory(self.batch_size)
        
    def remember(self, state, action, probs, vals, reward, done):
        self.memory.store_memory(state, action, probs, vals, reward, done)

    def save_models(self):
        print('... saving models ...')
        self.actor.save_checkpoint()
        self.critic.save_checkpoint()

    def load_models(self):
        print('... loading models ...')
        self.actor.load_checkpoint()
        self.critic.load_checkpoint()

    def choose_action(self, state):
        state = state.to(self.actor.device)
        with T.no_grad():
            dist = self.actor(state)
            value = self.critic(state)
        action = dist.sample().item()
        # prob = dist.probs[action].item()
        log_prob = dist.log_prob(T.tensor(action, device=self.actor.device)).item()
        value = value.item()

        return action, log_prob, value

    def calculate_advantage (self, reward_arr, val_arr, done_arr):
        advantage = np.zeros(len(reward_arr), dtype=np.float32)

        future_advantage = 0
        for t in reversed(range(len(reward_arr))):
            if t == len(reward_arr) - 1:  
                td_error = reward_arr[t] - val_arr[t]  # No next value for last step
            else:
                td_error = reward_arr[t] + self.gamma * val_arr[t+1] * (1 - int(done_arr[t])) - val_arr[t]
                
            future_advantage = td_error + self.gamma * self.gae_lambda * future_advantage * (1 - int(done_arr[t]))
            advantage[t] = future_advantage
        
        advantage = T.tensor(advantage).to(self.actor.device)

        # Normalization (optional)
        try:
            advantage_norm = (advantage - advantage.mean()) / (advantage.std() + 1e-8)
        except:
            self.log('advantage', advantage)
            self.log('reward_arr', reward_arr)
            self.log('val_arr', val_arr)
            self.logger.save()
            raise 
        
        
        self.logger.log('advantage_mean',advantage.mean().item())
        self.logger.log('advantage_std',advantage.std().item())
        
        return advantage_norm
        
    def learn(self, epoch):
        actor_losses = []  # for logging
        critic_losses = []  # for logging
        total_losses = [] # for logging
        
        state_arr, action_arr, old_log_probs_arr, val_arr, reward_arr, done_arr = self.memory.get_arrays()
        # Normalize rewards
        # reward_arr = (reward_arr - np.mean(reward_arr)) / (np.std(reward_arr) + 1e-8)

        advantage = self.calculate_advantage(reward_arr, val_arr, done_arr)
        values = T.tensor(val_arr).to(self.actor.device)

        for i in range(self.n_epochs):
            batches = self.memory.generate_batches()
        
            for batch in batches:
                states = T.tensor(state_arr[batch], dtype=T.float).to(self.actor.device)
                old_log_probs = T.tensor(old_log_probs_arr[batch]).to(self.actor.device)
                actions = T.tensor(action_arr[batch]).to(self.actor.device)

                dist = self.actor(states)
                critic_value = self.critic(states)
                critic_value = T.squeeze(critic_value)

                new_log_probs = dist.log_prob(actions)

                # Ratio of new and old probabilities (exp(log-probs))
                prob_ratio = T.exp(new_log_probs - old_log_probs)

                weighted_probs = advantage[batch] * prob_ratio
                weighted_clipped_probs = T.clamp(prob_ratio, 1 - self.policy_clip, 1 + self.policy_clip) * advantage[batch]
                actor_loss = -T.min(weighted_probs, weighted_clipped_probs).mean()

                # Calculate returns with value clipping
                returns = advantage[batch] + values[batch]
                value_clipped = values[batch] + T.clamp(critic_value - values[batch], -self.value_clip, self.value_clip)
                critic_loss1 = (returns - critic_value) ** 2
                critic_loss2 = (returns - value_clipped) ** 2
                critic_loss = 0.5 * T.max(critic_loss1, critic_loss2).mean()

                # Add entropy bonus for exploration
                dist_entropy = dist.entropy().mean()

                # Combine all losses
                total_loss = actor_loss + 0.5 * critic_loss - self.entropy_coefficient * dist_entropy

                # logging loss
                critic_losses.append(critic_loss.item())
                actor_losses.append(actor_loss.item()) 
                total_losses.append(total_loss.item())

                self.actor.optimizer.zero_grad()
                self.critic.optimizer.zero_grad()
                total_loss.backward()

                self.logger.log('max_actor_grad', max(p.grad.abs().max().item() for p in self.actor.parameters() if p.grad is not None))
                self.logger.log('max_critic_grad', max(p.grad.abs().max().item() for p in self.critic.parameters() if p.grad is not None))
                
                # Clip gradients for stability
                T.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                T.nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)

                self.actor.optimizer.step()
                self.critic.optimizer.step()

            self.critic.scheduler.step()
            self.actor.scheduler.step()


        self.memory.clear_memory()
        self.actor_loss = stat.mean(actor_losses)
        self.critic_loss = stat.mean(critic_losses)
        self.total_loss = stat.mean(total_losses)
        