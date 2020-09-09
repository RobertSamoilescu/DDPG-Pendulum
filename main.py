import argparse
import models
import utils

import torch
import torch.optim as optim

import numpy as np
import random
import gym
from itertools import count
import os
import pandas as pd
import pickle as pkl

parser = argparse.ArgumentParser()
parser.add_argument("--env_name", type=str, default="Pendulum-v0")
parser.add_argument("--seeds", type=int, nargs="+", default=[10, 20, 30])

parser.add_argument("--num_hidden", type=int, default=2)
parser.add_argument("--hidden_size", type=int, default=256)

parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--batch_size", type=int, default=100)

parser.add_argument("--num_steps", type=int, default=15000)
parser.add_argument("--horizon", type=int, default=1024)

parser.add_argument("--update_after", type=int, default=1000)
parser.add_argument("--update_steps", type=int, default=50)
parser.add_argument("--test_steps", type=int, default=500)
parser.add_argument("--save_steps", type=int, default=5000)

parser.add_argument("--rho", type=float, default=0.995)
parser.add_argument("--gamma", type=float, default=0.99)
parser.add_argument("--act_noise", type=float, default=0.1)
args = parser.parse_args()

# define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# define nets
q, q_targ = None, None
pi, pi_targ = None, None
optim_q, optim_pi = None, None


def init_nets(env):
	global q, q_targ
	global pi, pi_targ
	global optim_q, optim_pi

	action_size = env.action_space.shape[0]
	obs_size = env.observation_space.shape[0]

	# initialize policy
	pi_args = {
		"input_size": obs_size,
		"hidden_size": args.hidden_size,
		"output_size": action_size,
		"num_hidden": args.num_hidden
	}
	pi = models.Policy(**pi_args).to(device)
	pi_targ = models.Policy(**pi_args).to(device)
	pi_targ.load_state_dict(pi.state_dict())

	# initalize q function
	q_args = {
		"input_size": obs_size + action_size,
		"hidden_size": args.hidden_size,
		"output_size": 1,
		"num_hidden": args.num_hidden
	}
	q = models.FeedForward(**q_args).to(device)
	q_targ = models.FeedForward(**q_args).to(device)
	q_targ.load_state_dict(q.state_dict())


	# initalize optimzers
	optim_pi = optim.Adam(pi.parameters(), lr=args.lr)
	optim_q = optim.Adam(q.parameters(), lr=args.lr)


def test(step):
	test_env = gym.make(args.env_name)
	returns = []

	with torch.no_grad():
		for i in range(10):
			s = test_env.reset()
			ret = 0

			for j in range(args.horizon):
				tensor_s = torch.tensor([s]).float().to(device)
				a = pi(tensor_s).squeeze(0)

				# act in the environment
				s_prime, r, d, _ = test_env.step(a.cpu().numpy())
				ret += r

				# if the episode ended
				if d:
					break

				# update the state
				s = s_prime

			# append the total return
			returns.append(ret)

	# log the mean return over 100 episodes 
	mreturn = np.mean(returns)
	return mreturn


def update(buff):
	global pi, pi_targ
	global q, q_targ
	global optim_q, optim_pi


	for i in range(args.update_steps):
		s, a, r, s_prime, d = buff.sample(args.batch_size, device)

		# compute targets
		with torch.no_grad():
			a_prime = pi_targ(s_prime)
			s_a_prime = torch.cat([s_prime, a_prime], 1)
			y = r.unsqueeze(1) + args.gamma * (1 - d) * q_targ(s_a_prime)

		# compute q_function loss
		a = a.unsqueeze(1)
		s_a = torch.cat([s, a], 1)
		q_loss = torch.mean((q(s_a) - y)**2)

		# update q
		optim_q.zero_grad()
		q_loss.backward()
		optim_q.step()

		# compute pi loss
		s_a = torch.cat([s, pi(s)], 1)
		pi_loss = -torch.mean(q(s_a))

		# update pi
		optim_pi.zero_grad()
		pi_loss.backward()
		optim_pi.step()

		# update targets using polyak
		for (p, p_targ) in zip(q.parameters(), q_targ.parameters()):
			p_targ.data = args.rho * p_targ.data + (1 - args.rho) * p.data

		for (p, p_targ) in zip(pi.parameters(), pi_targ.parameters()):
			p_targ.data = args.rho * p_targ.data + (1 - args.rho) * p.data


def save_models(path):
	global q, q_targ
	global pi, pi_targ

	data = {
		"q": q.state_dict(),
		"q_targ": q_targ.state_dict(),
		"pi": pi.state_dict(),
		"pi_targ": pi_targ.state_dict()
	}

	mpath = os.path.join(path, "models.pkl")
	with open(mpath, "wb") as fout:
		pkl.dump(data, fout)


def train(seed, path):
	global pi, pi_targ
	global q, q_targ

	# initialize env
	env = gym.make(args.env_name)

	# initialize buffer
	buff = utils.Buffer()

	# set seed
	torch.manual_seed(seed)
	np.random.seed(seed)
	random.seed(seed)
	env.seed(0)

	# initialize networks
	init_nets(env)
	a_low  = env.action_space.low.item()
	a_high = env.action_space.high.item()

	# initalize number of steps
	step = 0
	log = {
		"step": [],
		"mreturn": [],
	}

	for i in count():
		s = env.reset()

		for j in range(args.horizon):
			step += 1

			# pick action according to the state
			with torch.no_grad():
				tensor_s = torch.tensor([s]).float().to(device)
				a = pi(tensor_s).squeeze(0)

			# add noise to the action
			# and clip the values
			a = (a + args.act_noise * torch.randn(1).item())
			a = a.clamp(min=a_low, max=a_high)

			# act in the environment
			s_prime, r, d, _ = env.step(a.cpu().numpy())

			# add tuple to the buffer
			buff.append((s, a, r, s_prime, d))

			# if time to update
			if step % args.update_steps == 0 and step > args.update_after:
				update(buff)

			# if time to test the performance
			if step % args.test_steps == 0:
				mreturn = test(step)
				log["step"].append(step)
				log["mreturn"].append(mreturn)
				print("step: %d, return: %.2f" % (step, mreturn))

			if step % args.save_steps == 0:
				save_models(path)
				print("Model saved!")

			# if the episode ended
			if d:
				break

			# update the state
			s = s_prime

		if step > args.num_steps:
			break

	# save as csv
	lpath = os.path.join(path, "log.csv")
	df = pd.DataFrame.from_dict(log)
	df.to_csv(lpath, index=False)


if __name__ == "__main__":
	if not os.path.exists("ckpts"):
		os.makedirs("ckpts")

	for seed in args.seeds:
		path = os.path.join("ckpts", str(seed))
		
		if not os.path.exists(path):
			os.makedirs(path)

		print(f"========= SEED {seed} =========")
		train(seed, path)