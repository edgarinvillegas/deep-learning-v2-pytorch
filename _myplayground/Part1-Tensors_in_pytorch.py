import torch

def activation(x):
    return 1 / (1 + torch.exp(-x))

torch.manual_seed(7)
features = torch.randn((1, 5))
weigths = torch.randn_like(features)
bias = torch.rand((1, 1))

h = torch.mm(features, weigths.t()) + bias
print(h)
y = activation(h)