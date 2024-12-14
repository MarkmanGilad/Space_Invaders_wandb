import torch
import torch.nn as nn


# Define the Actor-Critic Network (same as before)
class ActorCriticNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        
        #shared Layers
        self.linear1 = nn.Linear(state_dim, 128)
        self.LeakyRelu = nn.LeakyReLU()
        self.linear2 = nn.Linear(128, 256)
        self.linear3 = nn.Linear(256, 128)
                
        # Actor head
        self.actor_layer = nn.Linear(128, action_dim)
        self.softmax = nn.Softmax(dim=-1)
        
        # Critic head
        self.critic_layer = nn.Linear(128, 1)
    
    def forward(self, state):
        x = self.linear1(state)
        x = self.LeakyRelu(x)
        x = self.linear2(x)
        x = self.LeakyRelu(x)
        x = self.linear3(x)
        x = self.LeakyRelu(x)

        # Actor (policy) output
        action_values = self.actor_layer(x)
        action_probs = self.softmax(action_values)
        
        # Critic (value) output
        value = self.critic_layer(x)
        
        return action_probs, value

    def load_params(self, path):
        self.load_state_dict(torch.load(path, weights_only=False))

    def save_params(self, path):
        torch.save(self.state_dict(), path)
    
    def __call__(self, state):
        return self.forward(state)


# Define the Actor-Critic Agent (same as before)
class ActorCriticAgent:
    def __init__(self, player=1, state_dim= 88, action_dim=4, path=None):
        self.player = player
        self.policy_value : ActorCriticNetwork = ActorCriticNetwork(state_dim, action_dim)
        if path:
            self.policy_value.load_params(path=path)

    def get_Action(self, state, events=None, epoch=None, train=True):
        state_tensor = state
        action_probs, _ = self.policy_value(state_tensor)
        action_index = torch.argmax(action_probs) 
        if train:
            action_index = torch.multinomial(action_probs, 1).item()
            return action_index
        else:
            action_index = torch.argmax(action_probs)
            return action_index

    def get_action_and_value(self, state):
        state_tensor = state
        action_probs, value = self.policy_value(state_tensor)
        action_index = torch.multinomial(action_probs, 1).item()
        return action_index, action_probs[action_index], value
   
    def __call__(self, state, train=False, events=None):
        return self.get_action(state, train=train)

            