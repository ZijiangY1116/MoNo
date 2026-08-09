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
import scipy.io as scio


def parse_args():
	parser = argparse.ArgumentParser(description="Prepare Darcy dataset.")
	parser.add_argument(
		"--input_paths",
		nargs="+",
		required=True,
		help="Raw Darcy .mat files or directories containing them.",
	)
	parser.add_argument(
		"--output_dir",
		type=str,
		default="./dataset/",
		help="Directory for prepared Darcy train/test data.",
	)
	parser.add_argument("--src_res", type=int, default=421)
	parser.add_argument("--obj_res", type=int, default=85)
	parser.add_argument("--train_num", type=int, default=1000)
	parser.add_argument("--test_num", type=int, default=200)
	return parser.parse_args()


def downsample_field(field, src_res, obj_res):
	if src_res <= 1 or obj_res <= 1 or (src_res - 1) % (obj_res - 1) != 0:
		raise ValueError("src_res and obj_res must define an integer downsampling step.")

	step = (src_res - 1) // (obj_res - 1)
	return field[:, ::step, ::step][:, :obj_res, :obj_res]


def build_grid(obj_res, n_samples):
	grid = np.stack(
		np.meshgrid(np.linspace(0, 1, obj_res), np.linspace(0, 1, obj_res)),
		axis=-1,
	)
	return np.repeat(grid[None, ...], n_samples, axis=0)


def load_darcy_file(path, src_res, obj_res):
	matdata = scio.loadmat(path)
	coeff = downsample_field(matdata["coeff"], src_res, obj_res)
	sol = downsample_field(matdata["sol"], src_res, obj_res)
	coords = build_grid(obj_res, coeff.shape[0])
	return coords, coeff[..., None], sol[..., None]


def load_darcy_dataset(input_file, src_res, obj_res):

	coords, coeff, sol = load_darcy_file(input_file, src_res, obj_res)
	print("Load from {}: coeff shape {}, sol shape {}".format(input_file, coeff.shape, sol.shape))

	return coords, coeff, sol


def main():
	args = parse_args()
	print(f"Found {len(args.input_paths)} Darcy .mat files for processing.")
	print(args.input_paths)
	output_dir = os.path.join(args.output_dir, f"darcy_{args.obj_res}")
	os.makedirs(output_dir, exist_ok=True)

	print("Preparing Darcy data...")
	train_coords, train_coeff, train_sol = load_darcy_dataset(args.input_paths[0], args.src_res, args.obj_res)
	test_coords, test_coeff, test_sol = load_darcy_dataset(args.input_paths[1], args.src_res, args.obj_res)

	train_coords = train_coords[:args.train_num]
	train_coeff = train_coeff[:args.train_num]
	train_sol = train_sol[:args.train_num]

	test_coords = test_coords[:args.test_num]
	test_coeff = test_coeff[:args.test_num]
	test_sol = test_sol[:args.test_num]

	print(f"Train set: coords {train_coords.shape}, coeff {train_coeff.shape}, sol {train_sol.shape}")
	print(f"Test set: coords {test_coords.shape}, coeff {test_coeff.shape}, sol {test_sol.shape}")

	np.save(os.path.join(output_dir, "train.npy"), {"coords": train_coords, "condition": train_coeff, "sol": train_sol})
	np.save(os.path.join(output_dir, "test.npy"), {"coords": test_coords, "condition": test_coeff, "sol": test_sol})

	print(f"Saved train/test data to {output_dir}")


if __name__ == "__main__":
	main()
