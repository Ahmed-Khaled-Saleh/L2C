from torch.utils.data import Dataset, DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

train_set = datasets.CIFAR10(
    root='data',
    train= True,
    transform = ToTensor(),
    download= True
)

test_set = datasets.CIFAR10(
    root='data',
    train= False,
    transform = ToTensor(),
    download= True
)


