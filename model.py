
from torch import nn
import torch.nn.functional as F
import torch
from utils import compute_mixing_weights

class CNNCifar(nn.Module):
    def __init__(self, neighbour_sets):
        super(CNNCifar, self).__init__()
        self.neighbour_sets = neighbour_sets
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        self.alpha = nn.Parameter(torch.randn(100, 100), requires_grad=True)
        self.w = nn.Parameter(torch.randn(100, 100), requires_grad=True)


    def update_mixing_weights(self, alpha, neighbour_sets):
        with torch.no_grad():
            for i in range(len(neighbour_sets)):
                w = compute_mixing_weights(alpha[i], neighbour_sets[i])
                self.w.data[i] = w

    def forward(self, x, val= False):
        if val:
            self.update_mixing_weights(self.alpha, self.neighbour_sets)
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)



