from dataset import CIFAR10Dataset
from torch.utils.data import DataLoader

cifar_dataset = CIFAR10Dataset()
dataloader = DataLoader(cifar_dataset, batch_size=10, shuffle=True)

for batch in dataloader:
	images, labels = batch
	print(labels)
	print(images.shape)
	break

