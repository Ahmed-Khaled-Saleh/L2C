
from torch import nn
import torch.nn.functional as F
import torch
from utils import compute_mixing_weights

class CNNCifar(nn.Module):
    def __init__(self):
        super(CNNCifar, self).__init__()
        self.pool = nn.MaxPool2d(2, 2)

        self.conv1_weight = torch.randn(6, 3, 5, 5, requires_grad=True)
        self.conv1_bias = torch.randn(6, requires_grad=True)

        self.conv2_weight = torch.randn(16, 6, 5, 5, requires_grad=True)
        self.conv2_bias = torch.randn(16, requires_grad=True)

        self.fc1_weight = torch.randn(120, 16 * 5 * 5, requires_grad=True)
        self.fc1_bias = torch.randn(120, requires_grad=True)

        self.fc2_weight = torch.randn(84, 120, requires_grad=True)
        self.fc2_bias = torch.randn(84, requires_grad=True)

        self.fc3_weight = torch.randn(10, 84, requires_grad=True)
        self.fc3_bias = torch.randn(10, requires_grad=True)

        self.alpha = nn.Parameter(torch.randn(100, 100), requires_grad=True)
        self.w = nn.Parameter(torch.randn(100, 100), requires_grad=True)


    def forward(self, x):
        x = self.pool(F.relu(F.conv2d(x, self.conv1_weight, self.conv1_bias)))
        x = self.pool(F.relu(F.conv2d(x, self.conv2_weight, self.conv2_bias)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(torch.nn.functional.linear(x, self.fc1_weight, self.fc1_bias))
        x = F.relu(torch.nn.functional.linear(x, self.fc2_weight, self.fc2_bias))
        x = torch.nn.functional.linear(x, self.fc3_weight, self.fc3_bias)
        return F.log_softmax(x, dim=1)


