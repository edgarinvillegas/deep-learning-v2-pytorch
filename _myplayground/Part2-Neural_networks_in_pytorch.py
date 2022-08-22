import torch
from torch import nn
import torch.nn.functional as F
from collections import OrderedDict

def resetWeight(layer):
    layer.weight.data.fill_(0)
    layer.bias.data.fill_(0)

def resetWeights(model):
    resetWeight(model.fc1)
    resetWeight(model.fc2)
    resetWeight(model.fc3)

## Your solution here
class FourLayerNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.softmax(self.fc3(x), dim=1)
        return x
flNetwork = FourLayerNetwork()
resetWeights(flNetwork)

flsNetwork = nn.Sequential(OrderedDict([
    ('fc1', nn.Linear(784, 128)),
    ('relu1', nn.ReLU()),
    ('fc2', nn.Linear(128, 64)),
    ('relu2', nn.ReLU()),
    ('fc3', nn.Linear(64, 10)),
    ('softmax' , nn.Softmax(dim=1))
]))
resetWeights(flsNetwork)

x = torch.randn((64, 784))
outputCls = flNetwork.forward(x)
outputSeq = flsNetwork.forward(x)

#print('outputCls: ', outputCls)
#print('outputSeq: ', outputSeq)

print(outputCls.shape)
print(outputSeq.shape)

# Check that we get same output in both types of networks
assert (outputCls != outputSeq).sum() == 0
print('Success!')