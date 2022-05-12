import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal


class Actor(nn.Module):
    """ Gaussian policy with reparameterization tricks. """
    def __init__(self, state_size, action_size, hidden_size, log_std_min=-10, log_std_max=2, init_w=3e-3): #log_std_min : [old -10 new -20]
        super(Actor, self).__init__()

        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        # dense layer
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc1.weight.data.uniform_(-init_w, init_w)
        self.fc1.bias.data.uniform_(-init_w, init_w)

        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc2.weight.data.uniform_(-init_w, init_w)
        self.fc2.bias.data.uniform_(-init_w, init_w)

        # last layer to the mean of gaussian
        self.mean_linear = nn.Linear(hidden_size, action_size)
        self.mean_linear.weight.data.uniform_(-init_w, init_w)
        self.mean_linear.bias.data.uniform_(-init_w, init_w)

        # last layer to the log(std) of gaussian
        self.log_std_linear = nn.Linear(hidden_size, action_size)
        self.log_std_linear.weight.data.uniform_(-init_w, init_w)
        self.log_std_linear.bias.data.uniform_(-init_w, init_w)

        
    def forward(self, state):
        # pass forward
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        
        mean = self.mean_linear(x) # mean of gaussian
        log_std = self.log_std_linear(x)  # log(std) of gaussian
        # clip the log(std)
        log_std = torch.clamp(log_std, min=self.log_std_min, max=self.log_std_max)
        std = torch.exp(log_std)

        return mean, std