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
from pathlib import Path

import numpy as np
import scipy.io as scio


def parse_args():
	parser = argparse.ArgumentParser(description="Prepare Plasticity dataset.")
	parser.add_argument(
		"--input_path",
		type=str,
		required=True,
		help="Path to plas_N987_T20.mat or its parent directory.",
	)
	parser.add_argument(
		"--output_dir",
		type=str,
		default="./dataset/",
		help="Directory for Plasticity train/test data.",
	)
	parser.add_argument("--src_res1", type=int, default=101)
	parser.add_argument("--src_res2", type=int, default=31)
	parser.add_argument("--train_num", type=int, default=900)
	parser.add_argument("--test_num", type=int, default=80)
	return parser.parse_args()


def resolve_input_file(raw_path):
	path = Path(raw_path).expanduser()
	if path.is_file() and path.exists():
		return path

	if path.is_dir():
		candidate = path / "plas_N987_T20.mat"
		if candidate.exists():
			return candidate

	if path.suffix != ".mat":
		mat_path = path.with_suffix(".mat")
		if mat_path.exists():
			return mat_path

	raise FileNotFoundError(f"Input file does not exist: {path}")


def build_grid(src_res1, src_res2, n_samples):
	grid = np.stack(
		np.meshgrid(
			np.linspace(0, 1, src_res1),
			np.linspace(0, 1, src_res2),
			indexing="ij",
		),
		axis=-1,
	)
	return np.repeat(grid[None, ...], n_samples, axis=0)


def load_plasticity_dataset(input_file, src_res1, src_res2):
	matdata = scio.loadmat(input_file)
	if "input" not in matdata or "output" not in matdata:
		raise KeyError(f"Expected keys 'input' and 'output' in {input_file}")

	input_field = matdata["input"]
	output_field = matdata["output"]
	if input_field.ndim != 2:
		raise ValueError(f"Expected input shape (n_sample, {src_res1}), got {input_field.shape}")
	if output_field.ndim != 5:
		raise ValueError(
			f"Expected output shape (n_sample, {src_res1}, {src_res2}, t, c), got {output_field.shape}"
		)
	if output_field.shape[1] != src_res1 or output_field.shape[2] != src_res2:
		raise ValueError(
			f"Expected output spatial shape ({src_res1}, {src_res2}), got {output_field.shape[1:3]}"
		)

	n_samples = input_field.shape[0]
	coords = build_grid(src_res1, src_res2, n_samples)
	condition = input_field[:, :, None, None]
	condition = np.repeat(condition, src_res2, axis=2)
	condition = np.squeeze(condition, axis=-1)[..., None]
	t = np.repeat(np.linspace(0, 1, output_field.shape[3], dtype=np.float32)[None, :], n_samples, axis=0)
	sol = output_field.astype(np.float32)
	return coords.astype(np.float32), condition.astype(np.float32), sol, t


def main():
	args = parse_args()
	input_file = resolve_input_file(args.input_path)
	output_dir = os.path.join(args.output_dir, "plasticity")
	os.makedirs(output_dir, exist_ok=True)

	print("Preparing Plasticity data...")
	coords, condition, sol, t = load_plasticity_dataset(input_file, args.src_res1, args.src_res2)

	if coords.shape[0] < args.train_num + args.test_num:
		raise ValueError("Not enough samples for the requested train/test split.")

	train_coords = coords[:args.train_num]
	train_condition = condition[:args.train_num]
	train_sol = sol[:args.train_num]
	train_t = t[:args.train_num]

	test_coords = coords[-args.test_num:]
	test_condition = condition[-args.test_num:]
	test_sol = sol[-args.test_num:]
	test_t = t[-args.test_num:]

	print(
		f"Train set: coords {train_coords.shape}, condition {train_condition.shape}, sol {train_sol.shape}, t {train_t.shape}"
	)
	print(
		f"Test set: coords {test_coords.shape}, condition {test_condition.shape}, sol {test_sol.shape}, t {test_t.shape}"
	)

	np.save(
		os.path.join(output_dir, "train.npy"),
		{"coords": train_coords, "condition": train_condition, "sol": train_sol, "t": train_t},
	)
	np.save(
		os.path.join(output_dir, "test.npy"),
		{"coords": test_coords, "condition": test_condition, "sol": test_sol, "t": test_t},
	)

	print(f"Saved train/test data to {output_dir}")


if __name__ == "__main__":
	main()
