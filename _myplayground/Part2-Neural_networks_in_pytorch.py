import torch
from torch import nn
import torch.nn.functional as F

def resetWeight(layer):
    layer.weight.data.fill_(0)
    layer.bias.data.fill_(0)

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

resetWeight(flNetwork.fc1)
resetWeight(flNetwork.fc2)
resetWeight(flNetwork.fc3)

flsNetwork = nn.Sequential(
    nn.Linear(784, 128), nn.ReLU(),
    nn.Linear(128, 64), nn.ReLU(),
    nn.Linear(64, 10), nn.Softmax(dim=1)
);
resetWeight(flsNetwork[0])
resetWeight(flsNetwork[2])
resetWeight(flsNetwork[4])

x = torch.randn((64, 784))
outputCls = flNetwork.forward(x)
outputSeq = flsNetwork.forward(x)

print(outputCls.shape)
print(outputSeq.shape)

# Check that we get same output in both types of networks
assert (outputCls != outputSeq).sum() == 0