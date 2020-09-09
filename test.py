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
from PIL import Image

parser = argparse.ArgumentParser()
parser.add_argument("--env_name", type=str, default="Pendulum-v0")
parser.add_argument("--seed", type=int, default=10)

parser.add_argument("--num_hidden", type=int, default=2)
parser.add_argument("--hidden_size", type=int, default=256)

parser.add_argument("--horizon", type=int, default=1024)
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


def load_models():
	global q, q_targ
	global pi, pi_targ	

	path = os.path.join("ckpts", str(args.seed), "models.pkl")
	with open(path, "rb") as fin:
		data = pkl.load(fin)

	pi.load_state_dict(data["pi"])
	pi_targ.load_state_dict(data["pi_targ"])
	q.load_state_dict(data["q"])
	q_targ.load_state_dict(data["q_targ"])


def test(env):
	global pi, pi_targ
	global q, q_targ

	returns = []
	imgs = []

	with torch.no_grad():
		for i in range(10):
			s = env.reset()
			imgs.append([])
			ret = 0

			for j in range(args.horizon):
				imgs[i].append(env.render("rgb_array"))
				tensor_s = torch.tensor([s]).float().to(device)
				a = pi(tensor_s).squeeze(0)

				# act in the environment
				s_prime, r, d, _ = env.step(a.cpu().numpy())
				ret += r

				# if the episode ended
				if d:
					break

				# update the state
				s = s_prime

			# append the total return
			returns.append(ret)
			print("episode %d, return %.2f" % (i, ret))

	# log the mean return over 100 episodes 
	return returns, imgs


if __name__ == "__main__":
	env = gym.make(args.env_name)
	env.seed(10)

	init_nets(env)
	load_models()

	returns, imgs = test(env)
	print("========================")
	print("mean return: %.2f" % (np.mean(returns)))
	env.close()

	# save as gif the best episode
	bst_idx = np.argmin(returns)
	episode = imgs[bst_idx]
	episode = [Image.fromarray(img) for img in episode]

	if not os.path.exists("samples"):
		os.makedirs("samples")

	path = "samples/%d.gif" % (args.seed)
	episode[0].save(path, save_all=True, append_images=episode[1:], duration=75, loop=0)
