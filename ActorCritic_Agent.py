import torch
import torch.nn as nn

# Define the Actor-Critic Network with CUDA support
class ActorCriticNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, use_cuda=True):
        """
        Initialize the Actor-Critic Network.

        Args:
            state_dim (int): Dimension of the state space.
            action_dim (int): Dimension of the action space.
            use_cuda (bool): Whether to use CUDA if available.
        """
        super().__init__()
        
        # Determine device
        if use_cuda and torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
        print(f"Using device: {self.device}")
        
        # Shared Layers
        self.linear1 = nn.Linear(state_dim, 128)
        self.LeakyRelu = nn.LeakyReLU()
        self.linear2 = nn.Linear(128, 256)
        self.linear3 = nn.Linear(256, 128)
                
        # Actor head
        self.actor_layer = nn.Linear(128, action_dim)
        self.softmax = nn.Softmax(dim=-1)
        
        # Critic head
        self.critic_layer = nn.Linear(128, 1)

        # Move the model to the specified device
        self.to(self.device)

    def forward(self, state):
        """
        Forward pass through the network.

        Args:
            state (torch.Tensor): Input state tensor.

        Returns:
            action_probs (torch.Tensor): Probability distribution over actions.
            value (torch.Tensor): Estimated value of the state.
        """
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
        """
        Load network parameters from a file.

        Args:
            path (str): Path to the saved model parameters.
        """
        self.load_state_dict(torch.load(path, map_location=self.device))

    def save_params(self, path):
        """
        Save network parameters to a file.

        Args:
            path (str): Path to save the model parameters.
        """
        torch.save(self.state_dict(), path)

    def __call__(self, state):
        """
        Call the forward method of the network.

        Args:
            state (torch.Tensor): Input state tensor.

        Returns:
            tuple: Action probabilities and state value.
        """
        return self.forward(state)

# Define the Actor-Critic Agent with CUDA support
class ActorCriticAgent:
    def __init__(self, player=1, state_dim=88, action_dim=4, path=None, use_cuda=True):
        """
        Initialize the Actor-Critic Agent.

        Args:
            player (int): Identifier for the player.
            state_dim (int): Dimension of the state space.
            action_dim (int): Dimension of the action space.
            path (str, optional): Path to load pre-trained model parameters.
            use_cuda (bool): Whether to use CUDA if available.
        """
        self.player = player
        self.policy_value: ActorCriticNetwork = ActorCriticNetwork(state_dim, action_dim, use_cuda=use_cuda)
        self.device = self.policy_value.device
        self.policy_value = self.policy_value.to(self.device)
        if path:
            self.policy_value.load_params(path=path)

    def get_Action(self, state, events=None, epoch=None, train=True):
        """
        Get the action for the given state.

        Args:
            state (list or np.ndarray): Current state of the environment.
            events (optional): Additional events (unused).
            epoch (optional): Current training epoch (unused).
            train (bool): Whether to use training mode for sampling actions.

        Returns:
            int: Selected action index.
        """
        state_tensor = torch.tensor(state, dtype=torch.float32).to(self.device)  # Move state to device
        action_probs, _ = self.policy_value(state_tensor)
        if train:
            action_index = torch.multinomial(action_probs, 1).item()
        else:
            action_index = torch.argmax(action_probs).item()
        return action_index

    def get_action_and_value(self, state):
        """
        Get the action and value for the given state.

        Args:
            state (Tensor): Current state of the environment.

        Returns:
            tuple: Selected action index, action probability, and state value.
        """
        state = state.to(self.device)  # Move state to device
        action_probs, value = self.policy_value(state)
        action_index = torch.multinomial(action_probs, 1).item()
        return action_index, action_probs[action_index], value

    def __call__(self, state, train=False, events=None):
        """
        Call the get_Action method of the agent.

        Args:
            state (list or np.ndarray): Current state of the environment.
            train (bool): Whether to use training mode for sampling actions.
            events (optional): Additional events (unused).

        Returns:
            int: Selected action index.
        """
        return self.get_Action(state, train=train)
