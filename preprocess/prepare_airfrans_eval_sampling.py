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
import hashlib
import json
import os

import numpy as np
import tqdm


TASKS = ("full", "reynolds", "aoa")
MODES = ("all_surface")


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare deterministic AirfRANS evaluation sampling indices.")
    parser.add_argument("--data_root", type=str, default="./dataset", help="Root directory containing airfrans_<task> folders.")
    parser.add_argument("--tasks", type=str, nargs="+", default=list(TASKS), choices=TASKS)
    parser.add_argument("--splits", type=str, nargs="+", default=["test"], help="Prepared manifest splits to generate.")
    parser.add_argument("--subsamplings", type=int, nargs="+", default=[32000], help="Point counts per inference pass.")
    parser.add_argument("--modes", type=str, nargs="+", default=list(MODES), choices=MODES)
    parser.add_argument("--seed", type=int, default=0, help="Base seed used to generate per-case sampling indices.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing sampling files.")
    parser.add_argument("--skip_missing", action="store_true", help="Skip missing airfrans_<task> folders.")
    parser.add_argument("--max_passes", type=int, default=10000, help="Safety limit for random full-coverage sampling.")
    return parser.parse_args()


def stable_seed(*parts):
    text = "::".join(str(part) for part in parts)
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="little", signed=False) % (2**63 - 1)


def build_random_passes(n_point, subsampling, rng, max_passes):
    if subsampling <= 0:
        raise ValueError(f"subsampling must be positive, got {subsampling}.")
    if subsampling >= n_point:
        return [np.arange(n_point, dtype=np.int64)]

    covered = np.zeros(n_point, dtype=np.bool_)
    passes = []
    while not covered.all():
        if len(passes) >= max_passes:
            raise RuntimeError(f"Exceeded max_passes={max_passes} before covering all {n_point} points.")
        index = rng.choice(n_point, size=subsampling, replace=False).astype(np.int64)
        covered[index] = True
        passes.append(index)
    return passes


def build_all_surface_passes(surf, subsampling, rng, max_passes):
    if subsampling <= 0:
        raise ValueError(f"subsampling must be positive, got {subsampling}.")

    n_point = surf.shape[0]
    if subsampling >= n_point:
        return [np.arange(n_point, dtype=np.int64)]

    surf_indices = np.flatnonzero(surf.astype(np.bool_)).astype(np.int64)
    volume_indices = np.flatnonzero(~surf.astype(np.bool_)).astype(np.int64)
    n_volume = subsampling - surf_indices.shape[0]
    if n_volume <= 0:
        raise ValueError(
            f"Expected fewer surface points than subsampling={subsampling}, got {surf_indices.shape[0]} surface points."
        )

    covered_volume = np.zeros(volume_indices.shape[0], dtype=np.bool_)
    passes = []
    while not covered_volume.all():
        if len(passes) >= max_passes:
            raise RuntimeError(
                f"Exceeded max_passes={max_passes} before covering all {volume_indices.shape[0]} volume points."
            )
        volume_local = rng.choice(volume_indices.shape[0], size=n_volume, replace=False)
        volume_sample = volume_indices[volume_local]
        covered_volume[volume_local] = True
        index = np.concatenate((surf_indices, volume_sample)).astype(np.int64)
        rng.shuffle(index)
        passes.append(index)
    return passes


def save_passes(output_path, passes, n_point, subsampling, mode, case_name, split, seed):
    lengths = np.asarray([index.shape[0] for index in passes], dtype=np.int32)
    flat_indices = np.concatenate(passes).astype(np.int32)
    np.savez_compressed(
        output_path,
        indices=flat_indices,
        lengths=lengths,
        n_point=np.asarray(n_point, dtype=np.int64),
        subsampling=np.asarray(subsampling, dtype=np.int64),
        mode=np.asarray(mode),
        case_name=np.asarray(case_name),
        split=np.asarray(split),
        seed=np.asarray(seed, dtype=np.int64),
        pass_count=np.asarray(len(passes), dtype=np.int64),
    )


def generate_case_sampling(case_path, output_path, case_name, split, task, subsampling, mode, seed, overwrite, max_passes):
    if os.path.exists(output_path) and not overwrite:
        return False

    data = np.load(case_path)
    n_point = int(data["coords"].shape[0])
    surf = data["surf"].astype(np.bool_)
    case_seed = stable_seed(seed, task, split, case_name, subsampling, mode)
    rng = np.random.default_rng(case_seed)
    if mode == "random":
        passes = build_random_passes(n_point, subsampling, rng, max_passes)
    elif mode == "all_surface":
        passes = build_all_surface_passes(surf, subsampling, rng, max_passes)
    else:
        raise ValueError(f"Unknown sampling mode {mode}.")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    save_passes(output_path, passes, n_point, subsampling, mode, case_name, split, case_seed)
    return True


def main():
    args = parse_args()
    for task in args.tasks:
        dataset_name = f"airfrans_{task}"
        dataset_root = os.path.join(args.data_root, dataset_name)
        manifest_path = os.path.join(dataset_root, "manifest.json")
        if not os.path.exists(manifest_path):
            if args.skip_missing:
                print(f"Skip missing dataset: {dataset_root}")
                continue
            raise FileNotFoundError(manifest_path)

        with open(manifest_path, "r") as file:
            manifest = json.load(file)

        for subsampling in args.subsamplings:
            for mode in args.modes:
                output_dir = os.path.join(dataset_root, "eval_sampling", str(subsampling), mode)
                os.makedirs(output_dir, exist_ok=True)
                generated = 0
                skipped = 0
                split_counts = {}
                for split in args.splits:
                    if split not in manifest:
                        raise KeyError(f"Split {split} is not found in {manifest_path}.")
                    case_names = list(manifest[split])
                    split_counts[split] = len(case_names)
                    iterator = tqdm.tqdm(case_names, desc=f"{dataset_name} {split} {subsampling} {mode}")
                    for case_name in iterator:
                        case_path = os.path.join(dataset_root, "cases", f"{case_name}.npz")
                        if not os.path.exists(case_path):
                            raise FileNotFoundError(case_path)
                        output_path = os.path.join(output_dir, f"{case_name}.npz")
                        did_generate = generate_case_sampling(
                            case_path=case_path,
                            output_path=output_path,
                            case_name=case_name,
                            split=split,
                            task=task,
                            subsampling=subsampling,
                            mode=mode,
                            seed=args.seed,
                            overwrite=args.overwrite,
                            max_passes=args.max_passes,
                        )
                        generated += int(did_generate)
                        skipped += int(not did_generate)

                sampling_manifest = {
                    "dataset": dataset_name,
                    "task": task,
                    "splits": args.splits,
                    "split_counts": split_counts,
                    "subsampling": subsampling,
                    "mode": mode,
                    "seed": args.seed,
                    "generated": generated,
                    "skipped": skipped,
                    "version": 1,
                    "strategy": "linearno_random_until_full_coverage",
                }
                with open(os.path.join(output_dir, "manifest.json"), "w") as file:
                    json.dump(sampling_manifest, file, indent=2)
                print(f"Saved {dataset_name} {subsampling}/{mode}: generated={generated}, skipped={skipped}")


if __name__ == "__main__":
    main()
