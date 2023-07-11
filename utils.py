import torch
from torch.utils.data import Subset, ConcatDataset, SubsetRandomSampler, DataLoader
import numpy as np
import log

def local_sgd(node_number, model, theta, S, train_loader, optimizer, criterion, device):
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
        return the mixing weight where eeach element is computed as above
    '''
    w = torch.zeros(len(neighbour_set))
    for i, j in enumerate(neighbour_set):
        w[i] = torch.exp(alpha[i][j])
    w /= torch.sum(w)
    return w

def compute_test_acc(model, val_loader, device, test_accuracies, i):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data in val_loader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    test_accuracies[i].append(accuracy)
    return test_accuracies



def get_loaders(data_set, indxs, tasks):

    loaders = []
    for i in range(len(tasks)):
        subset_1 = Subset(data_set, indxs[tasks[i][0]])
        subset_2 = Subset(data_set, indxs[tasks[i][1]])

        concatenated_dataset = ConcatDataset([subset_1, subset_2])
        num_samples = len(concatenated_dataset)

        indices = np.random.permutation(num_samples)

        # Use SubsetRandomSampler for shuffling
        sampler = SubsetRandomSampler(indices)

        # Create a dataloader with shuffled samples
        loader = DataLoader(concatenated_dataset, sampler=sampler)
        loaders.append(loader)

    return loaders



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

