import torch
import torch.nn as nn
import torch.nn.functional as F


class FeedForward(nn.Module):
	def __init__(self, input_size, hidden_size, output_size, num_hidden):
		super(FeedForward, self).__init__()
		
		# save parameters
		self.input_size  = input_size
		self.hidden_size = hidden_size
		self.output_size = output_size
		self.num_hidden  = num_hidden

		layers  = []

		# input layers
		layers += [
			nn.Linear(input_size, hidden_size),
			nn.ReLU(inplace=True)
		]

		# hidden layers
		for i in range(num_hidden):
			layers += [
				nn.Linear(hidden_size, hidden_size),
				nn.ReLU(inplace=True)
			]

		# output layers
		layers += [
			nn.Linear(hidden_size, output_size)
		]

		self.net = nn.Sequential(*layers)


	def forward(self, x):
		return self.net(x)


class Policy(FeedForward):
	def __init__(self, input_size, hidden_size, output_size, num_hidden):
		super(Policy, self).__init__(input_size, hidden_size, output_size, num_hidden)

	def forward(self, x):
		x = self.net(x)
		x = torch.tanh(x)
		x = 2 * x
		return x