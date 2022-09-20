from dataset import CIFAR10Dataset
from torch.utils.data import DataLoader
from training import train
from model import SimpleCNN


train_dataloader = DataLoader(CIFAR10Dataset(sample_set='train'), batch_size=100, shuffle=True)
test_dataloader = DataLoader(CIFAR10Dataset(sample_set='test'), batch_size=100, shuffle=True)

model = SimpleCNN()

train(model=model, train_dataloader=train_dataloader, test_dataloader=test_dataloader, num_epochs=20)


