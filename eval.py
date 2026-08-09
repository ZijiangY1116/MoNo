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
import tqdm

import numpy as np
import torch

from utils import basic_utils
from utils.dataset import *


NU_AIR = np.array(1.56e-5)


def build_parser():
    parser = argparse.ArgumentParser(description="Evaluate a saved Local-Enhanced Latent Neural Operator experiment.")
    parser.add_argument(
        "--exp_folder",
        type=str,
        required=True,
        help="Experiment directory name under ./outputs or a full path to an experiment directory.",
    )
    parser.add_argument(
        "--outputs_root",
        type=str,
        default="./outputs",
        help="Root directory used when --exp_folder is not already a valid path.",
    )
    parser.add_argument(
        "--checkpoint_epoch",
        type=int,
        default=None,
        help="Optional checkpoint epoch to load. Defaults to the latest checkpoint found.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Torch device. Defaults to cuda if available, else cpu.",
    )
    parser.add_argument(
        "--eval_split",
        type=str,
        default="test",
        help="Dataset split to evaluate. Defaults to test.",
    )
    parser.add_argument(
        "--save_sample",
        action="store_true",
        default=False,
        help="Save every sample prediction under the experiment test directory.",
    )
    parser.add_argument("--airfrans_raw_root", type=str, default="./raw_datasets/AirRANS", help="Raw AirfRANS Dataset directory for CL/rho_l metrics.")
    return parser


def run_model(model, coords, condition, t=None):
    return model(coords, condition, t)


def evaluate_dataset_static(dataset, normalizer, model, device):
    metric_sums = {
        "l1": 0.0,
        "l2": 0.0,
        "rl1": 0.0,
        "rl2": 0.0,
        "peak_error": 0.0,
    }
    pred_records = []

    model.eval()

    with torch.no_grad():
        for index in tqdm.tqdm(range(len(dataset)), desc="Evaluating"):
            coords, condition, sol = dataset[index]
            sol_shape = tuple(sol.shape)
            coords = coords.reshape(1, -1, coords.shape[-1]).to(device)
            condition = condition.reshape(1, -1, condition.shape[-1]).to(device)
            sol = sol.reshape(1, -1, sol.shape[-1]).to(device)
            pred = run_model(model, coords, condition)

            if normalizer.is_apply_y2():
                pred = normalizer.apply_y2(pred, inverse=True)
                sol = normalizer.apply_y2(sol, inverse=True)

            pred_records.append(pred[0].detach().cpu().numpy().reshape(sol_shape))

            batch_metric_sums = basic_utils.compute_validation_metric_sums(pred, sol)
            for name, value in batch_metric_sums.items():
                metric_sums[name] += value.item()

    sample_count = len(dataset)
    return {name: value / sample_count for name, value in metric_sums.items()}, sample_count, pred_records


def evaluate_dataset_rollout(dataset, normalizer, model, device, rollout_history, coords_condition):
    metric_sums = {
        "l1": 0.0,
        "l2": 0.0,
        "rl1": 0.0,
        "rl2": 0.0,
        "peak_error": 0.0,
    }
    pred_records = []

    model.eval()

    with torch.no_grad():
        for index in tqdm.tqdm(range(len(dataset)), desc="Evaluating"):
            coords, sol = dataset[index]
            spatial_shape = tuple(sol.shape[:-2])
            solution_dim = sol.shape[-1]
            coords = coords.reshape(1, -1, coords.shape[-1]).to(device)
            sol = sol.reshape(1, -1, sol.shape[-2], sol.shape[-1]).to(device)

            total_steps = sol.shape[-2]
            if total_steps <= rollout_history:
                raise ValueError(f"rollout_history={rollout_history} must be smaller than total steps {total_steps}.")

            history = sol[:, :, :rollout_history, :]
            target = sol[:, :, rollout_history:, :]
            pred_steps = []

            for step_index in range(target.shape[-2]):
                condition = history.reshape(history.shape[0], history.shape[1], -1)
                if coords_condition:
                    condition = torch.cat((coords, condition), dim=-1)

                pred_step = run_model(model, coords, condition)
                pred_steps.append(pred_step.unsqueeze(-2))
                history = torch.cat((history[:, :, 1:, :], pred_step.unsqueeze(-2)), dim=-2)

            pred = torch.cat(pred_steps, dim=-2)
            if normalizer.is_apply_y2():
                pred = normalizer.apply_y2(pred, inverse=True)
                target = normalizer.apply_y2(target, inverse=True)

            pred_shape = spatial_shape + (target.shape[-2], solution_dim)
            pred_records.append(pred[0].detach().cpu().numpy().reshape(pred_shape))

            batch_metric_sums = basic_utils.compute_validation_metric_sums(pred, target)
            for name, value in batch_metric_sums.items():
                metric_sums[name] += value.item()

    sample_count = len(dataset)
    return {name: value / sample_count for name, value in metric_sums.items()}, sample_count, pred_records


def evaluate_dataset_time_embed(dataset, normalizer, model, device):
    metric_sums = {
        "l1": 0.0,
        "l2": 0.0,
        "rl1": 0.0,
        "rl2": 0.0,
        "peak_error": 0.0,
    }
    pred_records = []

    model.eval()

    with torch.no_grad():
        for index in tqdm.tqdm(range(len(dataset)), desc="Evaluating"):
            coords, condition, sol, t = dataset[index]
            sol_shape = tuple(sol.shape)
            coords = coords.reshape(1, -1, coords.shape[-1]).to(device)
            condition = condition.reshape(1, -1, condition.shape[-1]).to(device)
            sol = sol.reshape(1, -1, sol.shape[-2], sol.shape[-1]).to(device)
            t = t.reshape(1, -1).to(device)

            pred_steps = []
            for step_index in range(sol.shape[-2]):
                pred_step = run_model(model, coords, condition, t[:, step_index])
                pred_steps.append(pred_step.unsqueeze(-2))

            pred = torch.cat(pred_steps, dim=-2)
            if normalizer.is_apply_y2():
                pred = normalizer.apply_y2(pred, inverse=True)
                sol = normalizer.apply_y2(sol, inverse=True)

            pred_records.append(pred[0].detach().cpu().numpy().reshape(sol_shape))

            batch_metric_sums = basic_utils.compute_validation_metric_sums(pred, sol)
            for name, value in batch_metric_sums.items():
                metric_sums[name] += value.item()

    sample_count = len(dataset)
    return {name: value / sample_count for name, value in metric_sums.items()}, sample_count, pred_records


def evaluate_dataset_airfrans(dataset, normalizer, model, device, saved_config, raw_root=None):
    import pyvista as pv
    import scipy as sc

    raw_root = "./raw_datasets/AirRANS" if raw_root is None else raw_root
    volume_mse_vars = []
    surface_mse_vars = []
    true_cls = []
    pred_cls = []

    model.eval()
    for case_name in tqdm.tqdm(dataset.get_case_names(), desc="Evaluating AirfRANS"):
        case_path = os.path.join(dataset.data_root, "cases", f"{case_name}.npz")
        eval_subsampling = getattr(saved_config.data, "airfrans_eval_subsampling", None)
        eval_sampling_mode = getattr(saved_config.data, "airfrans_eval_sampling_mode", "random")
        sampling_path = os.path.join(dataset.data_root, "eval_sampling", str(int(eval_subsampling)), eval_sampling_mode, f"{case_name}.npz")
        pred_physical, _, surf, pred_norm, sol_norm = infer_airfrans_full_case(
            model=model,
            normalizer=normalizer,
            case_path=case_path,
            device=device,
            sampling_path=sampling_path,
            sampling_mode=eval_sampling_mode,
            apply_norm=dataset.apply_norm,
        )
        pred_norm_tensor = torch.from_numpy(pred_norm).float().unsqueeze(0)
        sol_norm_tensor = torch.from_numpy(sol_norm).float().unsqueeze(0)
        surf_tensor = torch.from_numpy(surf.astype(np.bool_))
        vol_tensor = ~surf_tensor

        if vol_tensor.any():
            volume_mse_vars.append(
                basic_utils.compute_mse_per_variable(
                    pred_norm_tensor[:, vol_tensor, :],
                    sol_norm_tensor[:, vol_tensor, :],
                )
            )
        if surf_tensor.any():
            surface_mse_vars.append(
                basic_utils.compute_mse_per_variable(
                    pred_norm_tensor[:, surf_tensor, :],
                    sol_norm_tensor[:, surf_tensor, :],
                )
            )

        internal = pv.read(os.path.join(raw_root, case_name, f"{case_name}_internal.vtu"))
        airfoil = pv.read(os.path.join(raw_root, case_name, f"{case_name}_aerofoil.vtp"))
        u_inf = float(case_name.split("_")[2])
        angle = float(case_name.split("_")[3])
        true_coef = compute_airfrans_coefficients(internal, airfoil, surf, u_inf, angle)
        pred_internal = set_airfrans_prediction_fields(internal, pred_physical, surf)
        pred_coef = compute_airfrans_coefficients(pred_internal, airfoil, surf, u_inf, angle)
        true_cls.append(true_coef[1])
        pred_cls.append(pred_coef[1])

    sample_count = len(dataset)
    true_cls = np.asarray(true_cls, dtype=np.float64)
    pred_cls = np.asarray(pred_cls, dtype=np.float64)
    true_cl_tensor = torch.from_numpy(true_cls).float().reshape(1, -1, 1)
    pred_cl_tensor = torch.from_numpy(pred_cls).float().reshape(1, -1, 1)
    metrics = {
        "transolver-C_L": basic_utils.compute_abs_relative_error(
            pred_cl_tensor,
            true_cl_tensor,
        ).item(),
        "transolver-rho_L": float(sc.stats.spearmanr(true_cls, pred_cls)[0]),
        "transolver-volume": basic_utils.tensor_to_float_list(
            torch.stack(volume_mse_vars).mean(dim=0)
        ),
        "transolver-surf": basic_utils.tensor_to_float_list(
            torch.stack(surface_mse_vars).mean(dim=0)
        ),
    }
    return metrics, sample_count


def reorganize_airfrans(in_order_points, out_order_points, quantity_to_reordered):
    index = np.zeros(out_order_points.shape[0], dtype=np.int64)
    for point_idx in range(out_order_points.shape[0]):
        matches = np.all(out_order_points[point_idx] == in_order_points, axis=1)
        match_index = np.argwhere(matches)
        if match_index.size == 0:
            raise ValueError("Could not match an airfoil surface point to the internal mesh.")
        index[point_idx] = match_index[0, 0]
    return quantity_to_reordered[index]


def wall_shear_stress(jacob_u, normals):
    strain = 0.5 * (jacob_u + jacob_u.transpose(0, 2, 1))
    strain = strain - strain.trace(axis1=1, axis2=2).reshape(-1, 1, 1) * np.eye(2)[None] / 3
    shear_stress = 2 * NU_AIR.reshape(-1, 1, 1) * strain
    return (shear_stress * normals[:, :2].reshape(-1, 1, 2)).sum(axis=2)


def compute_airfrans_coefficients(internal, airfoil, bool_surf, u_inf, angle):
    intern = internal.copy()
    aerofoil = airfoil.copy()
    point_mesh = intern.points[bool_surf, :2]
    point_surf = aerofoil.points[:, :2]

    intern = intern.compute_derivative(scalars="U", gradient="pred_grad")
    surf_grad = intern.point_data["pred_grad"].reshape(-1, 3, 3)[bool_surf, :2, :2]
    surf_p = intern.point_data["p"][bool_surf]
    surf_grad = reorganize_airfrans(point_mesh, point_surf, surf_grad)
    surf_p = reorganize_airfrans(point_mesh, point_surf, surf_p)

    aerofoil.point_data["wallShearStress"] = wall_shear_stress(surf_grad, -aerofoil.point_data["Normals"])
    aerofoil.point_data["p"] = surf_p
    aerofoil = aerofoil.ptc(pass_point_data=True)

    pressure_force = -aerofoil.cell_data["p"][:, None] * aerofoil.cell_data["Normals"][:, :2]
    pressure_force = (pressure_force * aerofoil.cell_data["Length"].reshape(-1, 1)).sum(axis=0)
    shear_force = (aerofoil.cell_data["wallShearStress"] * aerofoil.cell_data["Length"].reshape(-1, 1)).sum(axis=0)
    force = shear_force - pressure_force

    alpha = angle * np.pi / 180
    basis = np.array([[np.cos(alpha), np.sin(alpha)], [-np.sin(alpha), np.cos(alpha)]])
    force_rot = basis @ force
    return 2 * force_rot / u_inf ** 2


def set_airfrans_prediction_fields(internal, pred_physical, bool_surf):
    intern = internal.copy()
    pred_physical = pred_physical.copy()
    pred_physical[bool_surf, :2] = 0.0
    pred_physical[bool_surf, 3] = 0.0
    intern.point_data["U"][:, :2] = pred_physical[:, :2]
    intern.point_data["p"] = pred_physical[:, 2]
    intern.point_data["nut"] = pred_physical[:, 3]
    return intern


def load_eval_sampling(sampling_path, n_point):
    if not os.path.exists(sampling_path) or sampling_path is None:
        raise FileNotFoundError(
            f"{sampling_path}\n"
            "AirfRANS eval now expects precomputed sampling indices. "
            "Generate them with preprocess/prepare_airfrans_eval_sampling.py."
        )

    data = np.load(sampling_path)

    flat_indices = data["indices"].astype(np.int64)
    lengths = data["lengths"].astype(np.int64)
    passes = []
    offset = 0
    covered = np.zeros(n_point, dtype=np.bool_)
    # double-check
    for length in lengths:
        index = flat_indices[offset : offset + int(length)]
        offset += int(length)
        if index.size == 0:
            raise ValueError(f"{sampling_path} contains an empty sampling pass.")
        if index.min() < 0 or index.max() >= n_point:
            raise ValueError(f"{sampling_path} contains out-of-range indices.")
        covered[index] = True
        passes.append(torch.from_numpy(index).long())
    if offset != flat_indices.shape[0]:
        raise ValueError(f"{sampling_path} has inconsistent indices/lengths.")
    if not covered.all():
        raise ValueError(f"{sampling_path} does not cover all {n_point} points.")
    return passes


def infer_airfrans_full_case(model, normalizer, case_path, device, sampling_path=None, sampling_mode="random", apply_norm=True):
    data = np.load(case_path)
    coords = torch.from_numpy(data["coords"]).float()
    condition = torch.from_numpy(data["condition"]).float()
    sol = torch.from_numpy(data["sol"]).float()
    surf = torch.from_numpy(data["surf"].astype(np.bool_))
    if apply_norm:
        coords = normalizer.apply_x(coords)
        condition = normalizer.apply_y1(condition)

    n_point = coords.shape[0]
    if sampling_mode not in ("random", "all", "all_surface"):
        raise ValueError(f"Unknown AirfRANS eval sampling_mode {sampling_mode}.")

    pred_norm_sum = torch.zeros((n_point, data["sol"].shape[-1]), dtype=torch.float32)
    pred_count = torch.zeros((n_point, 1), dtype=torch.float32)
    sampling_passes = load_eval_sampling(sampling_path, n_point)

    model.eval()
    with torch.no_grad():
        for index in sampling_passes:
            coords_batch = coords.index_select(0, index).unsqueeze(0).to(device)
            condition_batch = condition.index_select(0, index).unsqueeze(0).to(device)
            pred = run_model(model, coords_batch, condition_batch).squeeze(0).cpu()
            pred_norm_sum.index_add_(0, index, pred)
            pred_count.index_add_(0, index, torch.ones((index.numel(), 1), dtype=torch.float32))

    if (pred_count == 0).any():
        raise ValueError(f"{sampling_path} left some AirfRANS points without predictions.")

    pred_norm = pred_norm_sum / pred_count
    if apply_norm:
        pred_physical_tensor = normalizer.apply_y2(pred_norm, inverse=True)
    else:
        pred_physical_tensor = pred_norm
    pred_physical = pred_physical_tensor.numpy()
    surf_numpy = surf.numpy()
    pred_physical[surf_numpy, :2] = 0.0
    pred_physical[surf_numpy, 3] = 0.0
    if apply_norm:
        pred_norm = normalizer.apply_y2(torch.from_numpy(pred_physical).float())
        sol_norm = normalizer.apply_y2(sol)
    else:
        pred_norm = torch.from_numpy(pred_physical).float()
        sol_norm = sol
    return pred_physical, sol.numpy(), surf_numpy, pred_norm.numpy(), sol_norm.numpy()


def _result_record_key(result):
    return "{split}/epoch_{epoch}".format(
        split=result["data_split"],
        epoch=result["checkpoint_epoch"],
    )


def save_incremental_result(result_path, result):
    result_key = _result_record_key(result)
    payload = {
        "evaluations": {
            result_key: result,
        },
    }

    if os.path.exists(result_path):
        with open(result_path, "r") as file:
            existing = json.load(file)

        if isinstance(existing, dict) and "evaluations" in existing:
            evaluations = existing["evaluations"]
        elif isinstance(existing, dict):
            evaluations = {
                _result_record_key(existing): existing,
            }
        else:
            raise ValueError(f"Unsupported existing result format in {result_path}.")

        evaluations[result_key] = result
        payload["evaluations"] = evaluations

    with open(result_path, "w") as file:
        json.dump(payload, file, indent=2)


def eval_main(
    exp_folder,
    outputs_root,
    checkpoint_epoch,
    device,
    eval_split="test",
    airfrans_raw_root=None,
    save_sample=False,
):

    # check device
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    
    # rebuild the args
    exp_dir, saved_args, saved_config, checkpoint_dir = basic_utils.load_experiment_meta(
        exp_folder,
        outputs_root,
    )
    eval_seed = getattr(saved_config.data, "seed", saved_args.get("seed", 0))
    saved_config.data["seed"] = eval_seed
    basic_utils.set_seed(eval_seed)
    checkpoint_path, checkpoint_epoch = basic_utils.resolve_checkpoint_path(checkpoint_dir, checkpoint_epoch)

    # build dataset
    if saved_config.data.type == 'static':
        # for eval
        # we transport the data normalization parameters from the training dataset to the test dataset to ensure the evaluation metrics are computed in the physical scale
        train_dataset = StaticDataset(
            'train',
            data_root=os.path.join(saved_config.data.data_root, saved_config.data.name),
            apply_norm=saved_config.data.apply_norm,
            coords_condition=saved_config.data.coords_condition,
        )
        train_norm_params = train_dataset.get_normalizer().export()
        dataset = StaticDataset(
            eval_split,
            data_root=os.path.join(saved_config.data.data_root, saved_config.data.name),
            apply_norm=saved_config.data.apply_norm,
            coords_condition=saved_config.data.coords_condition,
            norm_params=train_norm_params,
        )
    elif saved_config.data.type == 'airfrans':
        train_dataset = AirfRANSDataset(
            'train',
            data_root=os.path.join(saved_config.data.data_root, saved_config.data.name),
            apply_norm=saved_config.data.apply_norm,
            subsampling=saved_config.data.airfrans_subsampling,
            sampling_mode=saved_config.data.airfrans_sampling_mode,
        )
        train_norm_params = train_dataset.get_normalizer().export()
        dataset = AirfRANSDataset(
            eval_split,
            data_root=os.path.join(saved_config.data.data_root, saved_config.data.name),
            apply_norm=saved_config.data.apply_norm,
            norm_params=train_norm_params,
            subsampling=None,
            sampling_mode="all",
        )
    elif saved_config.data.type == 'rollout':
        train_dataset = RolloutDataset(
            'train',
            data_root=os.path.join(saved_config.data.data_root, saved_config.data.name),
            apply_norm=saved_config.data.apply_norm,
        )
        train_norm_params = train_dataset.get_normalizer().export()
        dataset = RolloutDataset(
            eval_split,
            data_root=os.path.join(saved_config.data.data_root, saved_config.data.name),
            apply_norm=saved_config.data.apply_norm,
            norm_params=train_norm_params,
        )
    elif saved_config.data.type == 'time_embedding':
        train_dataset = TimeEmbeddingDataset(
            'train',
            data_root=os.path.join(saved_config.data.data_root, saved_config.data.name),
            apply_norm=saved_config.data.apply_norm,
            coords_condition=saved_config.data.coords_condition,
        )
        train_norm_params = train_dataset.get_normalizer().export()
        dataset = TimeEmbeddingDataset(
            eval_split,
            data_root=os.path.join(saved_config.data.data_root, saved_config.data.name),
            apply_norm=saved_config.data.apply_norm,
            coords_condition=saved_config.data.coords_condition,
            norm_params=train_norm_params,
        )
    else:
        raise ValueError(f"Unsupported dataset type: {saved_config.data.type}")

    # build model
    normalizer = dataset.get_normalizer()
    if isinstance(dataset, (StaticDataset, AirfRANSDataset)):
        coords_dim, condition_dim, sol_dim = dataset.dim()
    elif isinstance(dataset, RolloutDataset):
        coords_dim, sol_dim = dataset.dim()
        rollout_history = getattr(saved_config.data, "rollout_history", 10)
        condition_dim = rollout_history * sol_dim
        if saved_config.data.coords_condition:
            condition_dim += coords_dim
    elif isinstance(dataset, TimeEmbeddingDataset):
        coords_dim, condition_dim, sol_dim = dataset.dim()
    else:
        raise ValueError(f"Unsupported type: {type(dataset)}")

    model = basic_utils.build_model_from_args(
        {
            "model_name": saved_args["model_name"],
            "normed_first_stage": saved_args["normed_first_stage"],
            "out_droprate": saved_args["out_droprate"],
            "coords_dim": coords_dim,
            "condition_dim": condition_dim,
            "sol_dim": sol_dim,
            "time_proj": saved_config.model.time_proj,
        }
    ).to(device)
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(basic_utils.clean_state_dict(state_dict))

    # evaluate
    pred_records = None
    if isinstance(dataset, StaticDataset):
        metrics, sample_count, pred_records = evaluate_dataset_static(
            dataset=dataset,
            normalizer=normalizer,
            model=model,
            device=device,
        )
    elif isinstance(dataset, AirfRANSDataset):
        metrics, sample_count = evaluate_dataset_airfrans(
            dataset=dataset,
            normalizer=normalizer,
            model=model,
            device=device,
            saved_config=saved_config,
            raw_root=airfrans_raw_root,
        )
    elif isinstance(dataset, RolloutDataset):
        metrics, sample_count, pred_records = evaluate_dataset_rollout(
            dataset=dataset,
            normalizer=normalizer,
            model=model,
            device=device,
            rollout_history=getattr(saved_config.data, "rollout_history", 10),
            coords_condition=saved_config.data.coords_condition,
        )
    elif isinstance(dataset, TimeEmbeddingDataset):
        metrics, sample_count, pred_records = evaluate_dataset_time_embed(
            dataset=dataset,
            normalizer=normalizer,
            model=model,
            device=device,
        )
    else:
        raise ValueError(f"Unsupported type: {type(dataset)}")

    # save results
    test_dir = exp_dir / "test"
    os.makedirs(test_dir, exist_ok=True)
    
    prediction_dir = None
    if save_sample:

        prediction_dir = test_dir / "samples" / eval_split / f"epoch_{checkpoint_epoch}"
        os.makedirs(prediction_dir, exist_ok=True)

        if pred_records is None:
            raise ValueError(
                "--save_sample currently supports only static, rollout, and "
                "time_embedding datasets."
            )
        for index, pred in enumerate(pred_records):
            np.save(prediction_dir / f"sample_{index:06d}.npy", np.asarray(pred))

    # save results
    result = {
        "exp_dir": str(exp_dir),
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_epoch": checkpoint_epoch,
        "data_split": eval_split,
        "sample_count": sample_count,
        **metrics,
    }
    if prediction_dir is not None:
        result["sample_prediction_dir"] = str(prediction_dir)

    save_incremental_result(test_dir / "res.json", result)

    if isinstance(dataset, AirfRANSDataset):
        print(
            "checkpoint_epoch={} sample_count={} {}".format(
                checkpoint_epoch,
                sample_count,
                " ".join(f"{name}={value}" for name, value in metrics.items()),
            )
        )
    else:
        print(
            "checkpoint_epoch={epoch} sample_count={count} l1={l1:.2e} l2={l2:.2e} rl1={rl1:.2e} rl2={rl2:.2e} peak_error={peak:.2e}".format(
                epoch=checkpoint_epoch,
                count=sample_count,
                l1=metrics["l1"],
                l2=metrics["l2"],
                rl1=metrics["rl1"],
                rl2=metrics["rl2"],
                peak=metrics["peak_error"],
            )
        )

    print("Finished evaluation. Experiment directory: {}".format(exp_dir))


if __name__ == "__main__":
    args = build_parser().parse_args()
    eval_main(
        args.exp_folder,
        args.outputs_root,
        args.checkpoint_epoch,
        args.device,
        eval_split=args.eval_split,
        airfrans_raw_root=args.airfrans_raw_root,
        save_sample=args.save_sample,
    )
