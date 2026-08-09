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
	parser = argparse.ArgumentParser(description="Prepare Elasticity dataset.")
	parser.add_argument(
		"--input_dir",
		type=str,
		required=True,
		help="Elasticity data directory. Supports either the dataset root containing Meshes/ or the Meshes directory itself.",
	)
	parser.add_argument(
		"--output_dir",
		type=str,
		default="./dataset/",
		help="Directory for Elasticity train/test data.",
	)
	parser.add_argument("--train_num", type=int, default=1000)
	parser.add_argument("--test_num", type=int, default=200)
	return parser.parse_args()


def load_elasticity_dataset(raw_files):
	xy = np.load(raw_files["xy"])
	sigma = np.load(raw_files["sigma"])

	coords = np.transpose(xy, (2, 0, 1))
	condition = coords.copy()
	sol = np.expand_dims(np.transpose(sigma, (1, 0)), axis=-1)
	return coords, condition, sol


def main():
	args = parse_args()
	raw_files = {
		"xy": os.path.join(args.input_dir, "Meshes", "Random_UnitCell_XY_10.npy"),
		"sigma": os.path.join(args.input_dir, "Meshes", "Random_UnitCell_sigma_10.npy"),
	}
	output_dir = os.path.join(args.output_dir, "elasticity")
	os.makedirs(output_dir, exist_ok=True)

	print("Preparing Elasticity data...")
	coords, condition, sol = load_elasticity_dataset(raw_files)

	train_coords = coords[:args.train_num]
	train_condition = condition[:args.train_num]
	train_sol = sol[:args.train_num]

	test_coords = coords[-args.test_num:]
	test_condition = condition[-args.test_num:]
	test_sol = sol[-args.test_num:]

	print(f"Train set: coords {train_coords.shape}, condition {train_condition.shape}, sol {train_sol.shape}")
	print(f"Test set: coords {test_coords.shape}, condition {test_condition.shape}, sol {test_sol.shape}")

	np.save(os.path.join(output_dir, "train.npy"), {"coords": train_coords, "condition": train_condition, "sol": train_sol})
	np.save(os.path.join(output_dir, "test.npy"), {"coords": test_coords, "condition": test_condition, "sol": test_sol})

	print(f"Saved train/test data to {output_dir}")


if __name__ == "__main__":
	main()
