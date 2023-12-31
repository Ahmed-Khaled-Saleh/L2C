import torch
from torch.utils.data import(
    Subset, 
    ConcatDataset, 
    SubsetRandomSampler, 
    DataLoader,
    random_split,
)

import numpy as np
import log

def local_sgd(node_number, 
              model, 
              theta, 
              S, 
              train_loader, 
              optimizer, 
              criterion, 
              device):
    
    log.info(f'Started training a Local SGD at node {node_number + 1}')

    model.load_state_dict(theta)
    for m in range(S):
        for _, data in enumerate(train_loader):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

    log.info(f'Finished training a Local SGD at node {node_number + 1}')
    return model.state_dict().copy()


def compute_mixing_weights(alpha, neighbour_set):
    '''
        \highlightcyan{$w_{i,j}=\frac{\exp(\alpha_{i,j})}{\sum_{\ell\in i\cup N(i)}\exp(\alpha_{i,\ell})}$}
        return the mixing weight where each element is computed as above
    '''
    w = torch.zeros(len(neighbour_set), dtype=alpha.dtype, device=alpha.device)
    for i, j in enumerate(neighbour_set):
        w[i] = torch.exp(alpha[j])
    w /= w.sum()
    return w

def compute_test_acc(model, test_loader, device, test_accuracies, i):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    test_accuracies[i].append(accuracy)
    return test_accuracies



def get_loaders(train_set, test_set, indxs, tasks):

    train_loaders = []
    val_loaders = []
    test_loaders = []

    for i in range(len(tasks)):
        subset_1 = Subset(train_set, indxs[tasks[i][0]])
        subset_2 = Subset(train_set, indxs[tasks[i][1]])

        concatenated_dataset = ConcatDataset([subset_1, subset_2])
        num_samples = len(concatenated_dataset)

        train_size = int(0.8 * num_samples)  # 80% for training
        val_size = len(concatenated_dataset) - train_size 

        train_dataset, val_dataset = random_split(concatenated_dataset, [train_size, val_size])

        train_indices = train_dataset.indices
        val_indices = val_dataset.indices

        # Use SubsetRandomSampler for shuffling
        train_sampler = SubsetRandomSampler(train_indices)
        val_sampler = SubsetRandomSampler(val_indices)


        # Create a dataloader with shuffled samples
        train_loader = DataLoader(train_dataset, shuffle= True)
        val_loader = DataLoader(val_dataset, shuffle= True)

        # create the test loader
        target_instances = [data for data in test_set if data[1] == tasks[i][0] or data[1] == tasks[i][1]]
        subset_3 = Subset(target_instances, range(len(target_instances)))
        test_loader = DataLoader(subset_3, batch_size=100, shuffle=True, num_workers=2)

        train_loaders.append(train_loader)
        val_loaders.append(val_loader)
        test_loaders.append(test_loader)

    return train_loaders, val_loaders, test_loaders



def plot_test_accuracies(test_accuracies, T, k, title):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 10))
    for i in range(k):
        plt.plot(range(T), test_accuracies[i], label='Client {}'.format(i))
    plt.xlabel('Communication rounds')
    plt.ylabel('Test accuracy')
    plt.legend()
    plt.title(title)
    plt.show()


import sys
from unittest.mock import patch
from contextlib import contextmanager

@contextmanager
def tqdm_output(tqdm, write=sys.stderr.write):
    def wrapper(message):
        if message != '\n':
            tqdm.clear()
        write(message)
        if '\n' in message:
            tqdm.display()

    with patch('sys.stdout', sys.stderr), patch('sys.stderr.write', wrapper):
        yield tqdm


