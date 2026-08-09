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
import json
import os

import numpy as np
import pyvista as pv
import tqdm


TASKS = ("full", "reynolds", "aoa")


def parse_args():
	parser = argparse.ArgumentParser(description="Prepare AirfRANS dataset for PONO.")
	parser.add_argument(
		"--input_dir",
		type=str,
		required=True,
		help="AirfRANS Dataset directory containing manifest.json and simulation folders.",
	)
	parser.add_argument(
		"--output_dir",
		type=str,
		default="./dataset/",
		help="Root directory where airfrans_<task> folders will be created.",
	)
	parser.add_argument(
		"--val_ratio",
		type=float,
		default=0.1,
		help="Tail fraction of each *_train split exposed as val in the prepared manifest.",
	)
	parser.add_argument(
		"--tasks",
		type=str,
		nargs="+",
		default=list(TASKS),
		choices=TASKS,
		help="AirfRANS tasks to prepare. Each task is saved as an independent dataset folder.",
	)
	parser.add_argument(
		"--overwrite",
		action="store_true",
		help="Overwrite existing prepared case files.",
	)
	return parser.parse_args()


def reorganize(in_order_points, out_order_points, quantity_to_reordered):
	index = np.zeros(out_order_points.shape[0], dtype=np.int64)
	for point_idx in range(out_order_points.shape[0]):
		matches = np.all(out_order_points[point_idx] == in_order_points, axis=1)
		match_index = np.argwhere(matches)
		if match_index.size == 0:
			raise ValueError("Could not match an airfoil surface point to the internal mesh.")
		index[point_idx] = match_index[0, 0]
	if not np.all(in_order_points[index] == out_order_points):
		raise ValueError("Surface point reordering failed.")
	return quantity_to_reordered[index]


def load_case(input_dir, case_name):
	internal_path = os.path.join(input_dir, case_name, f"{case_name}_internal.vtu")
	aerofoil_path = os.path.join(input_dir, case_name, f"{case_name}_aerofoil.vtp")
	if not os.path.exists(internal_path):
		raise FileNotFoundError(internal_path)
	if not os.path.exists(aerofoil_path):
		raise FileNotFoundError(aerofoil_path)

	internal = pv.read(internal_path)
	aerofoil = pv.read(aerofoil_path)

	surf = internal.point_data["U"][:, 0] == 0
	coords = internal.points[:, :2].astype(np.float32)
	sdf = (-internal.point_data["implicit_distance"][:, None]).astype(np.float32)

	name_parts = case_name.split("_")
	u_inf = float(name_parts[2])
	alpha = float(name_parts[3]) * np.pi / 180.0
	free_stream = (np.array([np.cos(alpha), np.sin(alpha)], dtype=np.float32) * u_inf)
	free_stream = np.broadcast_to(free_stream[None, :], (coords.shape[0], 2)).astype(np.float32)

	normal = np.zeros((coords.shape[0], 2), dtype=np.float32)
	normal[surf] = reorganize(
		aerofoil.points[:, :2],
		internal.points[surf, :2],
		-aerofoil.point_data["Normals"][:, :2],
	).astype(np.float32)

	condition = np.concatenate((coords, free_stream, sdf, normal), axis=-1).astype(np.float32)
	sol = np.concatenate(
		(
			internal.point_data["U"][:, :2],
			internal.point_data["p"][:, None],
			internal.point_data["nut"][:, None],
		),
		axis=-1,
	).astype(np.float32)

	return {
		"coords": coords,
		"condition": condition,
		"sol": sol,
		"surf": surf.astype(np.bool_),
		"case_name": np.array(case_name),
		"u_inf": np.array(u_inf, dtype=np.float32),
		"alpha_deg": np.array(float(name_parts[3]), dtype=np.float32),
	}


def split_train_val(case_names, val_ratio):
	if val_ratio <= 0.0:
		return list(case_names), []
	if val_ratio >= 1.0:
		raise ValueError("--val_ratio must be smaller than 1.0.")
	n_val = int(len(case_names) * val_ratio)
	n_val = min(n_val, max(0, len(case_names) - 1))
	return list(case_names[:-n_val]), list(case_names[-n_val:])


def build_task_manifest(raw_manifest, task, val_ratio):
	train_key = f"{task}_train"
	test_key = "full_test" if task == "scarce" else f"{task}_test"
	if train_key not in raw_manifest:
		raise KeyError(f"Missing {train_key} in raw manifest.")
	if test_key not in raw_manifest:
		raise KeyError(f"Missing {test_key} in raw manifest.")

	train_full = list(raw_manifest[train_key])
	train_cases, val_cases = split_train_val(train_full, val_ratio)
	test_cases = list(raw_manifest[test_key])
	all_cases = sorted(set(train_full) | set(test_cases))

	return {
		"task": task,
		"source_train_key": train_key,
		"source_test_key": test_key,
		"train_full": train_full,
		"train": train_cases,
		"val": val_cases,
		"test": test_cases,
		"all_cases": all_cases,
		"feature_names": {
		"coords": ["x", "y"],
		"condition": ["x", "y", "u_inf_x", "u_inf_y", "sdf", "normal_x", "normal_y"],
		"sol": ["u_x", "u_y", "p", "nut"],
		},
	}


def compute_stats(prepared_dir, case_names):
	count = 0
	condition_sum = None
	sol_sum = None
	for case_name in tqdm.tqdm(case_names, desc="Stats mean"):
		data = np.load(os.path.join(prepared_dir, "cases", f"{case_name}.npz"))
		condition = data["condition"].astype(np.float64)
		sol = data["sol"].astype(np.float64)
		if condition_sum is None:
			condition_sum = np.zeros(condition.shape[-1], dtype=np.float64)
			sol_sum = np.zeros(sol.shape[-1], dtype=np.float64)
		condition_sum += condition.sum(axis=0)
		sol_sum += sol.sum(axis=0)
		count += condition.shape[0]

	condition_mean = condition_sum / count
	sol_mean = sol_sum / count

	condition_sq_sum = np.zeros_like(condition_mean)
	sol_sq_sum = np.zeros_like(sol_mean)
	for case_name in tqdm.tqdm(case_names, desc="Stats std"):
		data = np.load(os.path.join(prepared_dir, "cases", f"{case_name}.npz"))
		condition = data["condition"].astype(np.float64)
		sol = data["sol"].astype(np.float64)
		condition_sq_sum += ((condition - condition_mean) ** 2).sum(axis=0)
		sol_sq_sum += ((sol - sol_mean) ** 2).sum(axis=0)

	condition_std = np.sqrt(condition_sq_sum / count) + 1e-8
	sol_std = np.sqrt(sol_sq_sum / count) + 1e-8
	return condition_mean.astype(np.float32), condition_std.astype(np.float32), sol_mean.astype(np.float32), sol_std.astype(np.float32)


def main():
	args = parse_args()

	with open(os.path.join(args.input_dir, "manifest.json"), "r") as file:
		raw_manifest = json.load(file)

	for task in args.tasks:
		prepared_dir = os.path.join(args.output_dir, f"airfrans_{task}")
		cases_dir = os.path.join(prepared_dir, "cases")
		os.makedirs(cases_dir, exist_ok=True)

		prepared_manifest = build_task_manifest(raw_manifest, task, args.val_ratio)

		for case_name in tqdm.tqdm(prepared_manifest["all_cases"], desc=f"Preparing {task} cases"):
			case_path = os.path.join(cases_dir, f"{case_name}.npz")
			if os.path.exists(case_path) and not args.overwrite:
				continue
			case_data = load_case(args.input_dir, case_name)
			np.savez_compressed(case_path, **case_data)

		with open(os.path.join(prepared_dir, "manifest.json"), "w") as file:
			json.dump(prepared_manifest, file, indent=2)

		train_cases = prepared_manifest["train"]
		condition_mean, condition_std, sol_mean, sol_std = compute_stats(prepared_dir, train_cases)
		np.savez(
			os.path.join(prepared_dir, "stats.npz"),
			condition_mean=condition_mean,
			condition_std=condition_std,
			sol_mean=sol_mean,
			sol_std=sol_std,
		)

	print("Saved prepared AirfRANS datasets:")
	for task in args.tasks:
		print(f"  {os.path.join(args.output_dir, f'airfrans_{task}')}")
	print("Each case contains coords [P, 2], condition [P, 7], sol [P, 4], and surf [P].")


if __name__ == "__main__":
	main()
