
from torch import nn
import torch.nn.functional as F
import torch
from utils import compute_mixing_weights

class CNNCifar(nn.Module):
    def __init__(self):
        super(CNNCifar, self).__init__()

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

        self.params_list = [self.conv1_weight, self.conv1_bias, self.conv2_weight, self.conv2_bias, self.fc1_weight, self.fc1_bias, self.fc2_weight, self.fc2_bias, self.fc3_weight, self.fc3_bias]

    def compute_mixing_weights(self, neighbour_set, i, w):
        for j in neighbour_set[i]:
            if j != 0:
                w[i][j] = torch.exp(self.alpha[i][j])
        w[i] /= w[i].sum()
        self.w[i].data = w[i]

    def aggregat(self, neighbour_set, i, w, delta_theta):
        for j in neighbour_set[i]:
            if j != 0:
                self.conv1_weight = self.conv1_weight - self.w[i][j] * delta_theta.conv1_weight.clone()
                self.conv1_bias = self.conv1_bias - self.w[i][j] * delta_theta.conv1_bias.clone()
                self.conv2_weight = self.conv2_weight - self.w[i][j] * delta_theta.conv2_weight.clone()
                self.conv2_bias = self.conv2_bias - self.w[i][j] * delta_theta.conv2_bias.clone()
                self.fc1_weight = self.fc1_weight - self.w[i][j] * delta_theta.fc1_weight.clone()
                self.fc1_bias = self.fc1_bias - self.w[i][j] * delta_theta.fc1_bias.clone()
                self.fc2_weight = self.fc2_weight - self.w[i][j] * delta_theta.fc2_weight.clone()
                self.fc2_bias = self.fc2_bias - self.w[i][j] * delta_theta.fc2_bias.clone()
                self.fc3_weight = self.fc3_weight - self.w[i][j] * delta_theta.fc3_weight.clone()
                self.fc3_bias = self.fc3_bias - self.w[i][j] * delta_theta.fc3_bias.clone()


    def forward(self, x, val=False, neighbour_set=None, i=None, w=None, delta_theta=None):

        if val:
            self.compute_mixing_weights(neighbour_set, i, w)
            self.aggregat(neighbour_set, i, w, delta_theta)

            
        x = F.max_pool2d(F.relu(F.conv2d(x, self.conv1_weight, self.conv1_bias)), 2, 2)
        x = F.max_pool2d(F.relu(F.conv2d(x, self.conv2_weight, self.conv2_bias)), 2, 2)
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(F.linear(x, self.fc1_weight, self.fc1_bias))
        x = F.relu(F.linear(x, self.fc2_weight, self.fc2_bias))
        x = F.linear(x, self.fc3_weight, self.fc3_bias)
        return F.log_softmax(x, dim=1)


