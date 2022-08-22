import torch


## Part 1
def activation(x):
    return 1 / (1 + torch.exp(-x))

# torch.manual_seed(7)
# features = torch.randn((1, 5))
# weigths = torch.randn_like(features)
# bias = torch.rand((1, 1))
#
# h = torch.mm(features, weigths.t()) + bias
# print(h)
# y = activation(h)

## Part 2
torch.manual_seed(7)

x = torch.randn((1,3))
# print(x)

n_input = x.shape[1]
n_hidden = 2
n_output = 1

W1 = torch.randn(n_input, n_hidden)
W2 = torch.randn((n_hidden, n_output))
b1 = torch.randn((1, n_hidden))
b2 = torch.randn((1, n_output))

h1 = torch.mm(x, W1) + b1
fh1 = activation(h1)

h2 = torch.mm(fh1, W2) + b2
y = activation(h2)

print(y)