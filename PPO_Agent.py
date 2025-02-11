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
    def __init__(self, input_dims, n_actions, lr, fc1_dims=256, fc2_dims=512, chkpt=1, optim_step = 100, optim_gamma = 0.9, logger = None, weight_decay = 1e-4):
        super(ActorNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dims, fc1_dims)
        self.fc2 = nn.Linear(fc1_dims, fc2_dims)
        # self.fc3 = nn.Linear(fc2_dims,fc2_dims )
        self.fc4 = nn.Linear(fc2_dims,fc1_dims )
        self.fc5 = nn.Linear(fc1_dims, n_actions)
        self.relu = nn.ReLU()
        self.leaky_relu = nn.LeakyReLU()
        self.softmax = nn.Softmax(dim=-1)
        self.checkpoint_file = f'Data/Actor{chkpt}.pth'
        self.optimizer = optim.Adam(self.parameters(), lr=lr)       # without weight_decay
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=optim_step, gamma=optim_gamma)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)
        self.logger = logger

    def forward(self, state):
        x = self.fc1(state)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        # x = self.fc3(x)
        # x = self.relu(x)
        x = self.fc4(x)
        x = self.relu(x)
        x = self.fc5(x)
        dist = Categorical(logits=x)
        return dist

    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file,weights_only=True))

    def get_all_params_as_list(self):
        params = [p.data.cpu().numpy().flatten() for p in self.parameters()]
        return [param for sublist in params for param in sublist]  # Flatten the nested lists

class CriticNetwork(nn.Module):
    def __init__(self, input_dims, lr, fc1_dims=256, fc2_dims=512, chkpt=1, optim_step = 100, optim_gamma = 0.9, weight_decay = 1e-4):
        super(CriticNetwork, self).__init__()

        self.checkpoint_file = f'Data/Critic{chkpt}.pth'
        self.fc1 = nn.Linear(input_dims, fc1_dims)
        self.fc2 = nn.Linear(fc1_dims, fc2_dims)
        # self.fc3 = nn.Linear(fc2_dims, fc2_dims)
        self.fc4 = nn.Linear(fc2_dims, fc1_dims)
        self.fc5 = nn.Linear(fc1_dims, 1)
        self.relu = nn.ReLU()
        self.leaky_relu = nn.LeakyReLU()  
        
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=optim_step, gamma=optim_gamma)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        x = self.fc1(state)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        # x = self.fc3(x)
        # x = self.relu(x)
        x = self.fc4(x)
        x = self.relu(x)
        x = self.fc5(x)
        return x

    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file,weights_only=True))

class PPO_Agent:
    def __init__(self, chkpt, input_dims=185, n_actions=4, logger=None, wandb = None):
        self.gamma = 0.995
        self.policy_clip = 0.2
        self.value_clip = 0.2  
        self.n_epochs = 4
        self.gae_lambda = 0.99
        self.max_grad_norm = 0.5  
        self.batch_size = 128
        self.lr_actor = 1e-3
        self.lr_critic = 1e-3
        self.weight_decay = 0.1
        self.optim_step = 10000
        self.optim_gamma = 0.95
        self.critic_actor_ratio = 0.5
        self.logger = logger
        self.wandb = None   # will be updated by Trainer
        self.frame_skip = 0 # number of frame to skip
        self.skip = 0   # counter for skipping memmory
        self.learn_step = 0 # counter for number of learning
        self.entropy_coefficient = 0.1
        self.entropy_coe_min = 0.001
        self.entropy_decay = 0.95         # Slower decay
        self.entropy_decay_steps = 500    # Less frequent decay

        self.actor = ActorNetwork(input_dims, n_actions, self.lr_actor, chkpt=chkpt, optim_step=self.optim_step, 
                                  optim_gamma=self.optim_gamma, logger=self.logger, weight_decay=self.weight_decay)
        self.critic = CriticNetwork(input_dims, self.lr_critic, chkpt=chkpt, optim_step=self.optim_step, 
                                    optim_gamma=self.optim_gamma, weight_decay=self.weight_decay)
        self.memory = PPOMemory(self.batch_size)
       

    def remember(self, state, action, probs, vals, reward, done):
        # if (reward == 0 or reward == -0.05) and self.skip < self.frame_skip:      # no skipping
        #     self.skip +=1
        #     return
        # self.skip = 0
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

    def calculate_advantage_and_returns (self, reward_arr, val_arr, done_arr, next_val):
        '''
        last step is always end of game. last done must be true
        '''
        
        advantage = np.zeros_like(reward_arr, dtype=np.float32)
        returns = np.zeros_like(reward_arr, dtype=np.float32)
                
        future_advantage = 0.0
        
        for t in reversed(range(len(reward_arr))):

            # For the last time step, use next_val for bootstrapping if not done
            if t == len(reward_arr) - 1:
                next_value = next_val * (1 - int(done_arr[t]))
            else:
                next_value = val_arr[t+1] * (1 - int(done_arr[t]))
            
            td_error = reward_arr[t] + self.gamma * next_value - val_arr[t]
            
            # GAE advantage calculation
            if done_arr[t]:
                future_advantage = td_error
            else:
                future_advantage = td_error + self.gamma * self.gae_lambda * future_advantage 
            
            advantage[t] = future_advantage
            
            #Return = V(s_t) + advantage
            returns[t] = advantage[t] + val_arr[t]
            
            
        # Convert to tensors and move to device
        advantage = T.tensor(advantage).to(self.actor.device)
        returns = T.tensor(returns).to(self.actor.device)

        # Normalization (optional)
        # advantage_norm = (advantage - advantage.mean()) / (advantage.std() + 1e-8)
        advantage_norm = advantage
        
        # Log for debugging and monitoring
        self.advantage_mean = advantage.mean().item()
        self.advantage_std = advantage.std().item()
        self.advantage_norm = advantage_norm.mean()
        self.logger.log('advantage_mean',self.advantage_mean)
        self.logger.log('advantage_std', self.advantage_std)
        self.logger.log('advantage_norm', self.advantage_norm)
        
        return advantage_norm, returns
        
    def learn(self, next_val):
        actor_losses = []   # for logging
        critic_losses = []  # for logging
        total_losses = []   # for logging
        entropy = []        # for logging

        # decay entropy coefficient
        self.learn_step += 1
        if self.learn_step % self.entropy_decay_steps == 0:
            self.entropy_coefficient = max(self.entropy_coefficient * self.entropy_decay, 
                                           self.entropy_coe_min)

        state_arr, action_arr, old_log_probs_arr, val_arr, reward_arr, done_arr = self.memory.get_arrays()
        
        # skipping learning if too few samples
        # if len(state_arr) < 2:          
        #     print(f"Skipping learning: only {len(state_arr)} samples, need at least 2")
        #     self.memory.clear_memory()
        #     return
        
        # Normalize rewards
        # reward_arr = (reward_arr - np.mean(reward_arr)) / (np.std(reward_arr) + 1e-8)

        # Compute advantage and returns
        advantage, returns = self.calculate_advantage_and_returns(reward_arr, val_arr, done_arr, next_val)

        for i in range(self.n_epochs):
            batches = self.memory.generate_batches()
        
            for batch in batches:
                states = T.tensor(state_arr[batch], dtype=T.float).to(self.actor.device)
                old_log_probs = T.tensor(old_log_probs_arr[batch]).to(self.actor.device)
                actions = T.tensor(action_arr[batch]).to(self.actor.device)
                old_critic_values = T.tensor(val_arr[batch], dtype=T.float).to(self.actor.device)
                dist = self.actor(states)
                new_log_probs = dist.log_prob(actions)

                critic_value = self.critic(states)
                critic_value = T.squeeze(critic_value)
                
                # Ratio of new and old probabilities (exp(log-probs))
                prob_ratio = T.exp(new_log_probs - old_log_probs)
                
                # Calculate actor loss
                weighted_probs = advantage[batch] * prob_ratio
                weighted_clipped_probs = T.clamp(prob_ratio, 1 - self.policy_clip, 1 + self.policy_clip) * advantage[batch]
                actor_loss = -T.min(weighted_probs, weighted_clipped_probs).mean()

                # Calculate critic loss
                # value_clipped = old_critic_values + T.clamp(critic_value - old_critic_values, -self.value_clip, self.value_clip)
                # critic_loss1 = (returns[batch] - critic_value) ** 2
                # critic_loss2 = (returns[batch] - value_clipped) ** 2
                # critic_loss = T.max(critic_loss1, critic_loss2).mean()
                
                # Calculate critic loss without clipping
                critic_loss = ((returns[batch] - critic_value) ** 2).mean()


                # Add entropy bonus for exploration
                dist_entropy = dist.entropy().mean()

                # Combine all losses
                total_loss = actor_loss + self.critic_actor_ratio * critic_loss - self.entropy_coefficient * dist_entropy

                # logging loss
                critic_losses.append(critic_loss.item())
                actor_losses.append(actor_loss.item()) 
                total_losses.append(total_loss.item())
                entropy.append(dist_entropy.item())
                self.wandb(val_arr_mean = val_arr.mean(), returns = returns.mean() )
                
                # Perform optimization step
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
        self.entropy = stat.mean(entropy)
