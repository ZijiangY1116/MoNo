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
	parser = argparse.ArgumentParser(description="Prepare Pipe dataset.")
	parser.add_argument(
		"--input_dir",
		type=str,
		required=True,
		help="Directory containing Pipe_Q.npy, Pipe_X.npy, and Pipe_Y.npy.",
	)
	parser.add_argument(
		"--output_dir",
		type=str,
		default="./dataset/",
		help="Directory for Pipe train/test data.",
	)
	parser.add_argument("--grid_res", type=int, default=129)
	parser.add_argument("--train_num", type=int, default=1000)
	parser.add_argument("--test_num", type=int, default=200)
	return parser.parse_args()


def build_computational_grid(grid_res, n_samples):
	grid = np.stack(
		np.meshgrid(np.linspace(0, 1, grid_res), np.linspace(0, 1, grid_res)),
		axis=-1,
	)
	return np.repeat(grid[None, ...], n_samples, axis=0)


def load_pipe_dataset(raw_files, grid_res):
	q = np.load(raw_files["q"])
	x_raw = np.expand_dims(np.load(raw_files["x"]), axis=-1)
	y_raw = np.expand_dims(np.load(raw_files["y"]), axis=-1)

	if q.ndim != 4:
		raise ValueError(f"Expected Pipe_Q.npy to be 4D, got shape {q.shape}")
	if x_raw.shape[:3] != y_raw.shape[:3]:
		raise ValueError(f"Pipe_X/Pipe_Y shape mismatch: {x_raw.shape} vs {y_raw.shape}")
	if q.shape[0] != x_raw.shape[0]:
		raise ValueError(f"Sample count mismatch between Pipe_Q and Pipe_X/Y: {q.shape[0]} vs {x_raw.shape[0]}")
	if x_raw.shape[1] != grid_res or x_raw.shape[2] != grid_res:
		raise ValueError(
			f"Expected Pipe_X/Pipe_Y spatial resolution ({grid_res}, {grid_res}), got {x_raw.shape[1:3]}"
		)

	coords = build_computational_grid(grid_res, x_raw.shape[0])
	condition = np.concatenate((x_raw, y_raw), axis=-1)
	sol = np.expand_dims(q[:, 0], axis=-1)
	return coords, condition, sol


def main():
	args = parse_args()
	raw_files = {
		"q": os.path.join(args.input_dir, "Pipe_Q.npy"),
        "x": os.path.join(args.input_dir, "Pipe_X.npy"),
        "y": os.path.join(args.input_dir, "Pipe_Y.npy"),
    }
	output_dir = os.path.join(args.output_dir, "pipe")
	os.makedirs(output_dir, exist_ok=True)

	print("Preparing Pipe data...")
	coords, condition, sol = load_pipe_dataset(raw_files, args.grid_res)

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
