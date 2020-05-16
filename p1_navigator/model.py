import torch
import torch.nn as nn
import torch.nn.functional as F

class QNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
        """
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        
        hidden_layer1 = 128
        hidden_layer2 = 64
        self.fc1 = nn.Linear(state_size, hidden_layer1)
        self.fc2 = nn.Linear(hidden_layer1,  hidden_layer2)
        self.fc3 = nn.Linear(hidden_layer2,  action_size)
        
        #self.dropout1 = nn.Dropout(p=0.1)

    def forward(self, state):
        """Build a network that maps state -> action values."""
        
        state = F.relu(self.fc1(state))
        #state = self.dropout1(state)
        
        state = F.relu(self.fc2(state))
        
        state = F.relu(self.fc3(state))
        #state = F.softmax(state, dim=1)
        
        return state
