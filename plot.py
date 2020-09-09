import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument("--seeds", type=int, nargs="+", default=[10, 20, 30])
args = parser.parse_args()


if __name__ == "__main__":
	dfs = []

	for seed in args.seeds:
		path = os.path.join("ckpts", str(seed), "log.csv")
		df = pd.read_csv(path)
		dfs.append(df)


	df = pd.concat(dfs)
	sns.lineplot(df["step"], df["mreturn"])
	plt.show()