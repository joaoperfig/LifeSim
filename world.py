import torch
import torch.nn as nn
import torch.nn.functional as F
from IPython import embed
from being import Brain

class World:

    def __init__(self):

        # Being parameters
        input_size = 4
        hidden_size = 5
        depth = 2
        output_size = 4
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.depth = depth
        self.output_size = output_size

        # World parameters
        startbeingnum = 5
        maxbeingnum = 10
        spawnsize = 100
        timescale = 0.0001
        lineardrag = 1
        maxvelocity = 2
        self.maxbeingnum = maxbeingnum
        self.spawnsize = spawnsize
        self.timescale = timescale
        self.lineardrag = lineardrag
        self.maxvelocity = maxvelocity

        # world state
        self.beings = []
        self.beingpositions = torch.zeros((maxbeingnum, 2))
        self.alivemask = torch.zeros((maxbeingnum,))
        self.beingvelocities = torch.zeros((maxbeingnum, 2))

        # initialize beings
        for i in range(startbeingnum):
            position = torch.rand((2,)) * self.spawnsize
            being = Brain(input_size, hidden_size, depth, output_size)
            self.beings += [being]
            self.alivemask[i] = 1
            self.beingpositions[i] = position

        for i in range(maxbeingnum-startbeingnum):
            self.beings += [None]
            self.alivemask[startbeingnum+i] = 0

    def time_step(self):
        inputs = self.get_inputs()
        decisions = self.compute_decisions(inputs)
        self.act_decisions(decisions)
        self.physics_step(self.timescale)

    def get_inputs(self):
        inputs = torch.zeros((self.maxbeingnum, self.input_size))
        return inputs

    def compute_decisions(self, inputs):
        decisions = []
        for i, input in enumerate(inputs):
            if self.alivemask[i]:
                decisions += [self.beings[i](input)]
            else:
                decisions += [torch.zeros((self.output_size,))]
        return torch.stack(decisions)
        
    def act_decisions(self, decisions):
        return

    def physics_step(self, timescale):
        # update position
        self.beingpositions = self.beingpositions + ((self.beingvelocities * torch.unsqueeze(self.alivemask, dim=1)) * timescale)
        
        # update speed
        sign = torch.sign(self.beingvelocities)
        value = torch.abs(self.beingvelocities) - (torch.ones(self.beingvelocities.shape) * self.lineardrag)
        value = torch.clamp(value ,min=0, max=self.maxvelocity)
        self.beingvelocities = sign * value



    


if __name__ == "__main__":
    world = World()
    embed()