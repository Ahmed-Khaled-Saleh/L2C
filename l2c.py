import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import torch
from torch import nn
import torch.optim as optim
from model import CNNCifar
from utils import local_sgd, compute_mixing_weights, compute_test_acc
from tqdm import tqdm
import log
torch.manual_seed(42)
np.random.seed(42)


def L2C(beta, neighbour_sets, train_loaders, val_loaders, S, T, T_0, K_0):
    
    k = len(neighbour_sets)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNNCifar().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1)

    
    test_accuracies = [[] for _ in range(k)]

    theta = [model.state_dict().copy() for _ in range(k)]
    theta_half = [model.state_dict().copy() for _ in range(k)]
    alpha = torch.zeros(k, k, requires_grad=True)
    w = torch.zeros(k, k)
    delta_theta = [model.state_dict().copy() for _ in range(k)]

    for t in tqdm(range(T)):
        for i in range(k):
            # Local SGD step
            new_state = local_sgd(i, model, theta[i], S, train_loaders[i], optimizer, criterion, device)
            model.load_state_dict(new_state)

            # Change capturing
            log.info(f'Computing change capturing at node {i + 1}')
            for name, param in model.named_parameters():
                delta_theta[i][name] = theta[i][name] - theta_half[i][name]

            log.info(f'Computing mixing weights at node {i + 1}')
            # Mixing weights calculation
            w[i] = compute_mixing_weights(alpha, neighbour_sets[i])

            # Aggregation
            log.info(f'Aggergating at node {i + 1}')
            theta_next = {}
            for name, param in model.named_parameters():
                theta_next[name] = theta[i][name].clone()

            for j in neighbour_sets[i]:
                for name, param in model.named_parameters():
                    theta_next[name] -= w[i][j].item() * delta_theta[i][name][j].clone()

            # Update L2C
            log.info(f'Updating L2C at node {i + 1}')
            model.load_state_dict(theta_next)
            for _, data in enumerate(val_loaders[i]):
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
            import pdb
            pdb.set_trace()
            alpha[i] -= beta * alpha[i].grad

            # Remove edges for sparse topology
            # if t == T_0:
            #     for _ in range(K_0):
            #         j = min(neighbour_sets[i], key=lambda x: w[i][x])
            #         neighbour_sets[i].delete(j)

            theta[i] = model.state_dict().copy()
            theta_half[i] = model.state_dict().copy()

            # Compute test accuracy for each local model
            test_accuracies = compute_test_acc(model, val_loaders[i], device, test_accuracies, i)
        
        log.info(f'Test accuracies atiteration at Comm_round {k} =  {sum(test_accuracies) / k}')
    
    return theta, test_accuracies