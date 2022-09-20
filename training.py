# File for training models
import torch
from tqdm import tqdm

def test(model, dataloader, criterion):
	
	test_loss = 0
	test_correct = 0
	test_total = 0

	with torch.no_grad():
		for _, batch in enumerate(dataloader):
			
			images, labels = batch

			predictions = model(images)

			test_loss += criterion(predictions, labels.long()).item()
			test_correct += torch.sum(torch.argmax(predictions, dim=-1) == labels).item()
			test_total += len(batch)

	average_test_loss = round(test_loss / len(dataloader), 3)
	average_test_accuracy = round(test_correct / test_total, 3)

	return average_test_loss, average_test_accuracy

def train(model, train_dataloader, test_dataloader, num_epochs=10, iteration_steps=100):

	criterion = torch.nn.CrossEntropyLoss()
	optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

	epoch_bar = tqdm(range(num_epochs))

	for epoch in epoch_bar:

		train_loss = 0
		train_correct = 0
		train_total = 0

		for iteration, batch in enumerate(train_dataloader):
			# read batch data
			images, labels = batch

			# reset optimizer
			optimizer.zero_grad()

			# calculate model predictions
			predictions = model(images)

			# calculate loss and use it to update model parameters
			loss = criterion(predictions, labels.long())
			loss.backward()
			optimizer.step()

			train_loss += loss.item()
			train_correct += torch.sum(torch.argmax(predictions, dim=-1) == labels).item()
			train_total += len(batch)

			if iteration % iteration_steps == 0 and iteration > 0:
				test_loss, test_accuracy = test(model, test_dataloader, criterion)
				epoch_bar.set_description('Epoch {0} | Iteration {1} | Train(Loss {2}, Acc {3}) | Test(Loss {4}, Acc {5})'.format(
					epoch,
					iteration,
					round(train_loss / iteration_steps,3),
					round(train_correct / train_total, 3),
					test_loss,
					test_accuracy,
					)
				)

				train_loss = 0
				train_correct = 0
				train_total = 0
				train_correct = 0
				train_total = 0

	print('finished training')	
