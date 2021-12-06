import torch
import torch.nn as nn
import torch.nn.functional as F
from IPython import embed


class Brain(nn.Module):
    def __init__(self, input_size, hidden_size, depth, output_size):
        super(Brain, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.depth = depth
        self.output_size = output_size

        linears = []
        lastsize = input_size
        for i in range(depth+1):
            linears += [nn.Linear(lastsize, hidden_size)]
            lastsize = hidden_size
        linears += [nn.Linear(hidden_size, output_size)]
        self.linears = nn.ModuleList(linears)

    def forward(self, x):
        for linear in self.linears:
            x = linear(x)
            x = torch.sigmoid(x)
        return x

    def mutate(self, probability, ammount):
        for linear in self.linears:

            # mutate weight
            shape = linear.weight.data.shape
            to_mutate = (torch.rand(size=shape) < probability).int()
            delta = ((torch.rand(size=shape)*2) - 1) * ammount
            delta = delta*to_mutate
            linear.weight.data = linear.weight.data + delta

            # mutate bias
            shape = linear.bias.data.shape
            to_mutate = (torch.rand(size=shape) < probability).int()
            delta = ((torch.rand(size=shape)*2) - 1) * ammount
            delta = delta*to_mutate
            linear.bias.data = linear.bias.data + delta

    def merge(self, partner):
        son = Brain(self.input_size, self.hidden_size, self.depth, self.output_size)

        for i in range(len(self.linears)):
            plinear = self.linears[i]
            mlinear = partner.linears[i]
            slinear = son.linears[i]

            # merge weight
            shape = plinear.weight.data.shape
            selector = torch.rand(size=shape)
            slinear.weight.data = (selector*plinear.weight.data) + ((1-selector)*mlinear.weight.data)

            # merge bias
            shape = plinear.bias.data.shape
            selector = torch.rand(size=shape)
            slinear.bias.data = (selector*plinear.bias.data) + ((1-selector)*mlinear.bias.data)

