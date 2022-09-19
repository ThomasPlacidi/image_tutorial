# File for creating a loading data
import torch
import os
import pickle
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import matplotlib.pyplot as plt
import PIL


class CIFAR10Dataset(Dataset):
	def __init__(self, directory_path='/Users/joshua/env/datasets/cifar_10'):
		labels_tensor_list = []
		image_tensor_list = []

		# define image transforms
		self.horizontal_flip = T.RandomHorizontalFlip(p=0.5)
		self.vertical_flip = T.RandomVerticalFlip(p=0.2)
		#self.jitter = T.ColorJitter(brightness=0.9, hue=0.5)
		#self.blurrer = T.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5))
		#self.resize_cropper = T.RandomResizedCrop(size=(32, 32))
		self.to_pil = T.ToPILImage()



		# get a list of files in the directory
		file_batches = os.listdir(directory_path)

		# only keep the data files
		file_batches = [f for f in file_batches if f.startswith('data')]

		# loop over each file batch
		for file_path in file_batches:

			# open data batch
			data_batch = self.open_batch(directory_path + '/' + file_path)
			
			# process data batch
			images, labels = self.process_batch(data_batch)		
			image_tensor_list.append(images)
			labels_tensor_list.append(labels)	

		# stack the list of tensors into one large tensor
		self.labels = torch.stack(labels_tensor_list).view(-1)
		self.images = torch.stack(image_tensor_list).view(-1, 3, 32, 32)

	def open_batch(self, filepath):
		with open(filepath, 'rb') as fo:
			dict = pickle.load(fo, encoding='bytes')
		return dict

	def process_batch(self, batch):
		labels = torch.Tensor(batch[b'labels'])

		images = torch.Tensor(batch[b'data'])
		images = images.view(-1, 3, 32, 32)

		return images, labels

	def __len__(self):
		return self.labels.shape[0]

	def apply_image_transforms(self, image):
		image = self.horizontal_flip(image)
		image = self.vertical_flip(image)
		#image = self.jitter(image)     Makes the transformed image black
		#image = self.blurrer(image)    Resolution too low
		#image = self.resize_cropper(image)
		return image

	def show_image(self, image):
		pil_image = self.to_pil(image)
		pil_image = PIL.ImageOps.invert(pil_image)
		plt.imshow(pil_image)
		plt.show()

	def __getitem__(self, idx):
		image = self.images[idx]
		label = self.labels[idx]

		image = self.apply_image_transforms(image)

		return image, label
