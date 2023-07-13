import networkx as nx
import numpy as np
import torch
np.random.seed(42)
torch.manual_seed(42)

from l2c import L2C
from utils import get_loaders, plot_test_accuracies
from data import train_set, test_set
from non_iid import generate_non_iid_tasks
from topolgy import Topolgy

beta = 0.01
k = 10
S = 10
T = 10
T_0 = 10
K_0 = 10


classes_per_task, indexs = generate_non_iid_tasks(train_set)
# import pdb; pdb.set_trace()
train_loaders, val_loaders, test_loaders = get_loaders(train_set, 
                                                       test_set, 
                                                       indexs, 
                                                       classes_per_task)

topology = Topolgy()

topology.generate_graph(params = 0.2)
neighbour_set =  nx.adjacency_matrix(topology.G).toarray()

theta, test_accs = L2C(beta, 
                       neighbour_set, 
                       train_loaders, 
                       val_loaders,
                       test_loaders, 
                       S, 
                       T, 
                       T_0, 
                       K_0)


plot_test_accuracies(test_accs, 100, k, 'L2C')
