# File for training models
import torch
from tqdm import tqdm

def train(model, dataloader, num_epochs=10, iteration_steps=100):

	criterion = torch.nn.CrossEntropyLoss()
	optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

	epoch_bar = tqdm(range(num_epochs))

	for epoch in epoch_bar:
		train_loss = 0

		for iteration, batch in enumerate(dataloader):
			# read batch data
			images, labels = batch

			# reset optimizer
			optimizer.zero_grad()

			# calculate model predictions
			outputs = model(images)

			# calculate loss and use it to update model parameters
			loss = criterion(outputs, labels.long())
			loss.backward()
			optimizer.step()

			train_loss += loss.item()

			if iteration % iteration_steps == 0:
				epoch_bar.set_description('Epoch {0} | Iteration {1} | Loss {2}'.format(epoch, iteration, train_loss / iteration_steps))
				train_loss = 0

	print('finished training')	
