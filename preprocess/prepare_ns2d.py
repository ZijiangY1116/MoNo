# Copyright 2026 Zijiang Yang.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import os

import numpy as np
import scipy.io as scio


def parse_args():
	parser = argparse.ArgumentParser(description="Prepare NS2d dataset.")
	parser.add_argument(
		"--input_path",
		type=str,
		required=True,
		help="Raw NS2d .mat file path.",
	)
	parser.add_argument(
		"--output_dir",
		type=str,
		default="./dataset/",
		help="Directory for NS2d train/test data.",
	)
	parser.add_argument("--src_res", type=int, default=64)
	parser.add_argument("--train_num", type=int, default=1000)
	parser.add_argument("--test_num", type=int, default=200)
	return parser.parse_args()


def build_grid(src_res, n_samples):
	grid = np.stack(
		np.meshgrid(np.linspace(0, 1, src_res), np.linspace(0, 1, src_res)),
		axis=-1,
	)
	return np.repeat(grid[None, ...], n_samples, axis=0)


def load_ns2d_dataset(input_file, src_res):
	matdata = scio.loadmat(input_file)
	if "u" not in matdata:
		raise KeyError(f"Cannot find key 'u' in {input_file}")

	u = matdata["u"]
	if u.ndim != 4:
		raise ValueError(f"Expected u to be 4D, got shape {u.shape}")
	if u.shape[1] != src_res or u.shape[2] != src_res:
		raise ValueError(
			f"Expected spatial resolution ({src_res}, {src_res}), got {u.shape[1:3]}"
		)

	coords = build_grid(src_res, u.shape[0])
	sol = np.expand_dims(u, axis=-1)
	return coords, sol


def main():
	args = parse_args()
	input_file = args.input_path
	output_dir = os.path.join(args.output_dir, "ns2d")
	os.makedirs(output_dir, exist_ok=True)

	print("Preparing NS2d data...")
	coords, sol = load_ns2d_dataset(input_file, args.src_res)

	if coords.shape[0] < args.train_num + args.test_num:
		raise ValueError("Not enough samples for the requested train/test split.")

	train_coords = coords[:args.train_num]
	train_sol = sol[:args.train_num]

	test_coords = coords[-args.test_num:]
	test_sol = sol[-args.test_num:]

	print(f"Train set: coords {train_coords.shape}, sol {train_sol.shape}")
	print(f"Test set: coords {test_coords.shape}, sol {test_sol.shape}")

	np.save(os.path.join(output_dir, "train.npy"), {"coords": train_coords, "sol": train_sol})
	np.save(os.path.join(output_dir, "test.npy"), {"coords": test_coords, "sol": test_sol})

	print(f"Saved train/test data to {output_dir}")


if __name__ == "__main__":
	main()
