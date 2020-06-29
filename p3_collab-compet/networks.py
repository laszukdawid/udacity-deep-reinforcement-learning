import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn.modules import activation
from torch.nn.modules.linear import Linear

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def layer_init(layer, w_scale=1.0):
    nn.init.orthogonal_(layer.weight.data)
    layer.weight.data.mul_(w_scale)
    nn.init.constant_(layer.bias.data, 0)
    return layer

def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)

class ActorBody(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_layers=(64,64)):
        super(ActorBody, self).__init__()

        layers = [ nn.Linear(input_dim, hidden_layers[0]) ]
        layers += [ nn.Linear(dim_in, dim_out) for dim_in, dim_out in zip(hidden_layers[:-1], hidden_layers[1:]) ]
        layers += [ nn.Linear(hidden_layers[-1], output_dim) ]

        layers = [layer_init(layer) for layer in layers]
        # self.fc1 = nn.Linear(input_dim, hidden_layers[0])
        # self.fc2 = nn.Linear(hidden_layers[0], hidden_layers[1])
        # self.fc3 = nn.Linear(hidden_layers[1], output_dim)

        # self.dropout = nn.Dropout(0.25)

        self.gate = F.relu
        self.gate_out = torch.tanh

        # self.reset_parameters()
        self.layers = nn.ModuleList(layers)

    def reset_parameters(self):
        for layer in self.layers:
            layer_init(layer)

    def forward(self, x):
        # return a vector of the force
        for idx, layer in enumerate(self.layers[:-1]):
            x = layer(x)
            # if idx == 0:
            #     x = self.dropout(x)
            x = self.gate(x)
        # return self.gate_out(self.layers[-1](x))
        # return self.layers[-1](x)
        # return self.layers[-1](x).clamp(-1, 1)
        return F.normalize(self.layers[-1](x), dim=-1)

class CriticBody(nn.Module):
    def __init__(self, input_dim, action_dim, hidden_layers=(64,64), actor=False):
        super(CriticBody, self).__init__()

        """self.input_norm = nn.BatchNorm1d(input_dim)
        self.input_norm.weight.data.fill_(1)
        self.input_norm.bias.data.fill_(0)"""


        layers = [ nn.Linear(input_dim, hidden_layers[0]) ]
        if len(hidden_layers) > 1:
            layers += [ nn.Linear(hidden_layers[0]+action_dim, hidden_layers[1]) ]
            layers += [ nn.Linear(dim_in, dim_out) for dim_in, dim_out in zip(hidden_layers[1:-1], hidden_layers[2:]) ]
            layers += [ nn.Linear(hidden_layers[-1], 1) ]
        else:
            layers += [ nn.Linear(hidden_layers[0]+action_dim, 1) ]

        layers = [layer_init(layer) for layer in layers]

        self.gate = F.relu
        self.gate_out = torch.tanh

        self.actor = actor
        # self.reset_parameters()
        self.layers = nn.ModuleList(layers)

    def reset_parameters(self):
        for layer in self.layers:
            layer_init(layer)

    def forward(self, x, actions):
        # critic network simply outputs a number
        for idx, layer in enumerate(self.layers[:-1]):
            if idx == 1:
                x = self.gate(layer(torch.cat((x, actions), dim=-1)))
            else:
                x = self.gate(layer(x))
        return self.gate_out(self.layers[-1](x))
