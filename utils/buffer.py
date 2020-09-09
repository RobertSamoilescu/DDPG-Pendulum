import numpy as np
import torch

class Buffer(object):
	def __init__(self, buff_capacity=1000000):
		self.buff = []
		self.idx = 0
		self.buff_capacity = buff_capacity


	def append(self, x):
		# add to the buffer
		if len(self.buff) < self.buff_capacity:
			self.buff.append(x)
		else:
			self.buff[self.idx] = x
		
		# update idx
		self.idx = (self.idx + 1) % self.buff_capacity


	def sample(self, batch_size, device):
		# sample from the buffer
		indices = range(len(self.buff))
		indices_batch = np.random.choice(indices, size=batch_size, replace=False)
		batch = [self.buff[i] for i in indices_batch]

		# pack the data
		s, a, r, s_prime, d = [], [], [], [], []
		for x in batch:
			s.append(x[0])
			a.append(x[1])
			r.append(x[2])
			s_prime.append(x[3])
			d.append(x[4])

		# transform to tensors
		s = torch.tensor(s).float().to(device)
		a = torch.tensor(a).float().to(device)
		r = torch.tensor(r).float().to(device)
		s_prime = torch.tensor(s_prime).float().to(device)
		d = torch.tensor(d).float().to(device)

		return s, a, r, s_prime, d