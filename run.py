from dataset import CIFAR10Dataset
from torch.utils.data import DataLoader
from training import train
from model import SimpleCNN

cifar_dataset = CIFAR10Dataset(directory_path='/Users/tomplacidi/env/datasets/cifar_10')
dataloader = DataLoader(cifar_dataset, batch_size=100, shuffle=True)

model = SimpleCNN()

train(model=model, dataloader=dataloader, num_epochs=50)


