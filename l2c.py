import torch
from torch import nn
import torch.optim as optim
from model import CNNCifar
from utils import (local_sgd, 
                   compute_mixing_weights, 
                   compute_test_acc, 
                   tqdm_output)

from tqdm import tqdm
import log
from copy import deepcopy


def L2C(beta, neighbour_set, train_loaders, val_loaders, test_loaders, S, T, T_0, K_0):
    
    k = len(neighbour_set)
    device = torch.device("cuda" if not torch.cuda.is_available() else "cpu")
    model = CNNCifar().to(device)
    criterion = nn.CrossEntropyLoss()
    # optimizer for all params except alpha and w
    params = [model.conv1_weight, model.conv1_bias, model.conv2_weight, model.conv2_bias, model.fc1_weight, model.fc1_bias, model.fc2_weight, model.fc2_bias, model.fc3_weight, model.fc3_bias]
    optimizer = optim.SGD(params, lr=0.01, momentum=0.9, weight_decay=0.01)
    l2c_optimizer = optim.Adam([model.alpha], lr=beta, weight_decay=0.01)

    test_accuracies = [[] for _ in range(k)]

    theta = [CNNCifar().to(device) for _ in range(k)]
    theta_half = [CNNCifar().to(device) for _ in range(k)]
    delta_theta = [CNNCifar().to(device) for _ in range(k)]
    theta_next = [CNNCifar().to(device) for _ in range(k)]

    w = torch.zeros(k, k, dtype=theta[0].alpha.dtype, device=theta[0].alpha.device)

    with tqdm(range(T)) as trange:
        for t in trange:
            for i in range(k):
                # Local SGD step
                log.info(f'Started training a Local SGD at node {i + 1}')

                # theta_half[i] = theta[i]
                # for m in range(S):
                #     for _, data in enumerate(train_loaders[i]):
                #         inputs, labels = data
                #         inputs, labels = inputs.to(device), labels.to(device)
                #         optimizer.zero_grad()
                #         outputs = theta_half[i](inputs)
                #         loss = criterion(outputs, labels)
                #         loss.backward()
                #         optimizer.step()

                log.info(f'Finished training a Local SGD at node {i + 1}')

                # Change capturing
                log.info(f'Computing change capturing at node {i + 1}')
                # import pdb; pdb.set_trace()
                delta_theta[i].fc1_weight = theta[i].fc1_weight - theta_half[i].fc1_weight.clone()
                delta_theta[i].fc1_bias = theta[i].fc1_bias - theta_half[i].fc1_bias.clone()
                delta_theta[i].fc2_weight = theta[i].fc2_weight - theta_half[i].fc2_weight.clone()
                delta_theta[i].fc2_bias = theta[i].fc2_bias - theta_half[i].fc2_bias.clone()
                delta_theta[i].fc3_weight = theta[i].fc3_weight - theta_half[i].fc3_weight.clone()
                delta_theta[i].fc3_bias = theta[i].fc3_bias - theta_half[i].fc3_bias.clone()
                delta_theta[i].conv1_weight = theta[i].conv1_weight - theta_half[i].conv1_weight.clone()
                delta_theta[i].conv1_bias = theta[i].conv1_bias - theta_half[i].conv1_bias.clone()
                delta_theta[i].conv2_weight = theta[i].conv2_weight - theta_half[i].conv2_weight.clone()
                delta_theta[i].conv2_bias = theta[i].conv2_bias - theta_half[i].conv2_bias.clone()
                               
                log.info(f'Computing mixing weights at node {i + 1}')               
                # Update L2C
                log.info(f'Updating L2C at node {i + 1}')
                # a training loop to find alpha that minimizes the validation loss
                for _, data in enumerate(val_loaders[i]):
                    inputs, labels = data
                    inputs, labels = inputs.to(device), labels.to(device)
                    
                    l2c_optimizer.zero_grad()
                    model.alpha.requires_grad_(True)
                    outputs = theta_next[i](inputs,
                                            val=True,
                                            neighbour_set=neighbour_set,
                                            i=i,
                                            w=w,
                                            delta_theta=delta_theta[i])
                    
                    loss = criterion(outputs, labels)
                    loss.backward()
                    print(f'gradient of alpha is {theta_next[i].alpha.grad}')
                    l2c_optimizer.step()

                # Remove edges for sparse topology
                if t == T_0:
                    for _ in range(K_0):
                        j = min(neighbour_set[i], key=lambda x: w[i][x])
                        neighbour_set[i].delete(j)

                # theta[i] = model.state_dict().copy()
                # theta_half[i] = model.state_dict().copy()

                # Compute test accuracy for each local model
                test_accuracies = compute_test_acc(model, test_loaders[i], device, test_accuracies, i)
            
        log.info(f'Test accuracies atiteration at Comm_round {t} =  {sum(test_accuracies) / k}')
    
    return theta, test_accuracies