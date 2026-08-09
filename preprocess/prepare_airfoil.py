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

import os
import argparse

import numpy as np


def parse_args():
	parser = argparse.ArgumentParser(description="Prepare Airfoil dataset.")
	parser.add_argument(
		"--input_dir",
		type=str,
		required=True,
		help="Airfoil data directory.",
	)
	parser.add_argument(
		"--output_dir",
		type=str,
		default="./dataset/",
		help="Directory for Airfoil train/test data.",
	)
	parser.add_argument("--train_num", type=int, default=1000)
	parser.add_argument("--test_num", type=int, default=200)
	return parser.parse_args()


def build_computational_grid(n_samples):
	grid = []
	for x1 in np.linspace(0, 1, 221):
		for x2 in np.linspace(0, 1, 51):
			grid.append([x1, x2])
	grid = np.reshape(np.array(grid), (221, 51, 2))
	grid = np.expand_dims(grid, axis=0)
	return np.repeat(grid, n_samples, axis=0)


def load_airfoil_dataset(raw_files):
	q = np.load(raw_files["q"])
	x_raw = np.expand_dims(np.load(raw_files["x"]), axis=-1)
	y_raw = np.expand_dims(np.load(raw_files["y"]), axis=-1)
	
	coords = build_computational_grid(x_raw.shape[0])
	condition = np.concatenate((x_raw, y_raw), axis=-1)
	sol = np.expand_dims(q[:, 4, :, :], axis=-1)
	return coords, condition, sol


def main():
	args = parse_args()
	raw_files = {
		'q': os.path.join(args.input_dir, 'naca', "NACA_Cylinder_Q.npy"),
		'x': os.path.join(args.input_dir, 'naca', "NACA_Cylinder_X.npy"),
		'y': os.path.join(args.input_dir, 'naca', "NACA_Cylinder_Y.npy"),
    }

	output_dir = os.path.join(args.output_dir, "airfoil")
	os.makedirs(output_dir, exist_ok=True)

	print("Preparing Airfoil data...")
	coords, condition, sol = load_airfoil_dataset(raw_files)

	train_coords = coords[:args.train_num]
	train_condition = condition[:args.train_num]
	train_sol = sol[:args.train_num]

    # keep the same as LinearNO / transolver
	test_coords = coords[args.train_num:args.train_num + args.test_num]
	test_condition = condition[args.train_num:args.train_num + args.test_num]
	test_sol = sol[args.train_num:args.train_num + args.test_num]

	print(f"Train set: coords {train_coords.shape}, condition {train_condition.shape}, sol {train_sol.shape}")
	print(f"Test set: coords {test_coords.shape}, condition {test_condition.shape}, sol {test_sol.shape}")

	np.save(os.path.join(output_dir, "train.npy"), {"coords": train_coords, "condition": train_condition, "sol": train_sol})
	np.save(os.path.join(output_dir, "test.npy"), {"coords": test_coords, "condition": test_condition, "sol": test_sol})

	print(f"Saved train/test data to {output_dir}")


if __name__ == "__main__":
	main()
