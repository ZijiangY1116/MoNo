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

import builtins as __builtin__
import json
import math
import os
import random
import sys
import time
import datetime
from pathlib import Path
import torch.distributed as dist

import numpy as np
import torch
import yaml
from matplotlib import pyplot as plt

from .dataset import *
from collections import defaultdict, deque


def unwrap_model(model):
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        return model.module
    return model


def build_train_metric_dict(loss_dict, total_loss, world_size):
    metric_dict = {"loss": total_loss.detach()}
    for loss_name, loss_value in loss_dict.items():
        metric_dict[f"loss_{loss_name}"] = loss_value.detach()
    return reduce_metric_dict(metric_dict, world_size)


def get_rollout_autoreg_ratio(current_epoch, total_epoch, target_ratio, warmup_portion=0.3, ramp_end_portion=0.7):
    if not 0.0 <= target_ratio <= 1.0:
        raise ValueError(f"target_ratio must be in [0, 1], got {target_ratio}.")
    if not 0.0 <= warmup_portion <= ramp_end_portion <= 1.0:
        raise ValueError(
            f"Expected 0 <= warmup_portion <= ramp_end_portion <= 1, got "
            f"{warmup_portion}, {ramp_end_portion}."
        )

    if total_epoch <= 1:
        return float(target_ratio)

    epoch_progress = current_epoch / max(total_epoch - 1, 1)
    if epoch_progress <= warmup_portion:
        return 0.0
    if epoch_progress >= ramp_end_portion:
        return float(target_ratio)

    ramp_progress = (epoch_progress - warmup_portion) / max(ramp_end_portion - warmup_portion, 1e-12)
    cosine_growth = 0.5 * (1.0 - math.cos(math.pi * ramp_progress))
    return float(target_ratio * cosine_growth)


def update_rollout_history_with_scheduled_sampling(history, gt_step, pred_step, autoreg_ratio):

    teacher_forcing_prob = 1.0 - autoreg_ratio
    teacher_forcing_mask = (torch.rand(history.shape[0], device=history.device) < teacher_forcing_prob).view(history.shape[0], 1, 1)
    mixed_step = torch.where(teacher_forcing_mask, gt_step, pred_step.detach())
    updated_history = torch.cat((history[:, :, 1:, :], mixed_step.unsqueeze(-2)), dim=-2)
    return updated_history, teacher_forcing_mask


def compute_lp_loss(pred, target, p):
    reduce_dims = tuple(range(1, len(pred.shape)))
    error = torch.mean(torch.abs(pred - target) ** p, dim=reduce_dims) ** (1 / p)
    return torch.mean(error)


def compute_relative_l2(pred, target):
    reduce_dims = tuple(range(1, len(pred.shape)))
    error = torch.sum(torch.abs(pred - target) ** 2, dim=reduce_dims) ** 0.5
    target_norm = torch.sum(torch.abs(target) ** 2, dim=reduce_dims) ** 0.5
    target_norm = torch.clamp(target_norm, min=1e-12)
    return torch.mean(error / target_norm)


def compute_relative_l1(pred, target):
    reduce_dims = tuple(range(1, len(pred.shape)))
    error = torch.sum(torch.abs(pred - target), dim=reduce_dims)
    target_norm = torch.sum(torch.abs(target), dim=reduce_dims)
    target_norm = torch.clamp(target_norm, min=1e-12)
    return torch.mean(error / target_norm)


def compute_abs_relative_error(pred, target):
    target_abs = torch.clamp(torch.abs(target), min=1e-12)
    return torch.mean(torch.abs(pred - target) / target_abs)


def compute_mse_per_variable(pred, target):
    if pred.shape != target.shape:
        raise ValueError(f"Expected matching shapes, got pred={pred.shape}, target={target.shape}.")
    if pred.ndim < 2:
        raise ValueError(f"Expected a tensor with a trailing variable dimension, got shape {pred.shape}.")
    reduce_dims = tuple(range(pred.ndim - 1))
    return torch.mean((pred - target) ** 2, dim=reduce_dims)


def compute_mse_mean(pred, target):
    return torch.mean(compute_mse_per_variable(pred, target))


def compute_rmse_per_variable(pred, target):
    return torch.sqrt(compute_mse_per_variable(pred, target))


def tensor_to_float_list(tensor):
    return [float(value) for value in tensor.detach().cpu().reshape(-1)]


def compute_peak_error(pred, target):
    reduce_dims = tuple(range(1, len(pred.shape)))
    error = torch.amax(torch.abs(pred - target), dim=reduce_dims)
    return torch.mean(error)


def compute_validation_metrics(pred, target):
    return {
        "l1": compute_lp_loss(pred, target, p=1),
        "l2": compute_lp_loss(pred, target, p=2),
        "rl1": compute_relative_l1(pred, target),
        "rl2": compute_relative_l2(pred, target),
        "peak_error": compute_peak_error(pred, target),
    }


def compute_validation_metrics_per_sample(pred, target):
    reduce_dims = tuple(range(1, len(pred.shape)))
    abs_error = torch.abs(pred - target)
    abs_target = torch.abs(target)
    squared_error = abs_error ** 2
    squared_target = abs_target ** 2

    return {
        "l1": torch.mean(abs_error, dim=reduce_dims),
        "l2": torch.mean(squared_error, dim=reduce_dims) ** 0.5,
        "rl1": torch.sum(abs_error, dim=reduce_dims) / torch.clamp(torch.sum(abs_target, dim=reduce_dims), min=1e-12),
        "rl2": torch.sum(squared_error, dim=reduce_dims) ** 0.5 / torch.clamp(torch.sum(squared_target, dim=reduce_dims) ** 0.5, min=1e-12),
        "peak_error": torch.amax(abs_error, dim=reduce_dims),
    }


def compute_validation_metric_sums(pred, target):
    per_sample_metrics = compute_validation_metrics_per_sample(pred, target)
    return {name: values.sum() for name, values in per_sample_metrics.items()}


def reduce_metric_dict(metrics, world_size):
    reduced = {}
    for name, value in metrics.items():
        metric_tensor = value.detach().clone()
        torch.distributed.all_reduce(metric_tensor)
        reduced[name] = (metric_tensor / world_size).item()
    return reduced


class Dict(dict):
    def __init__(self, dictionary=None):
        super().__init__()
        if dictionary is not None:
            self.load(dictionary)

    def __getattr__(self, item):
        try:
            return self[item] if item in self else getattr(super(), item)
        except AttributeError as exc:
            raise AttributeError(f'This dictionary has no attribute "{item}"') from exc

    def load(self, dictionary, name_list=None):
        for name, data in dictionary.items():
            if name_list is not None and name not in name_list:
                continue

            if isinstance(data, dict):
                if name in self:
                    self[name].load(data)
                else:
                    self[name] = Dict(data)
            elif isinstance(data, list):
                self[name] = []
                for item in data:
                    self[name].append(Dict(item) if isinstance(item, dict) else item)
            else:
                self[name] = data


class Checkpoint:
    def __init__(self, directory, model, device=None):
        self.directory = directory
        os.makedirs(self.directory, exist_ok=True)
        self.model = model
        self.device = device

    def load(self, epoch):
        state_dict = torch.load(
            os.path.join(self.directory, f"{epoch}.pt"),
            map_location="cuda",
        )
        self.model.load_state_dict(state_dict)

    def save(self, epoch):
        torch.save(self.model.state_dict(), os.path.join(self.directory, f"{epoch}.pt"))


class MetricLogger(object):
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                "{}: {}".format(name, str(meter))
            )
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None, log_file=None):
        i = 0
        if not header:
            header = ''
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt='{avg:.6f}')
        data_time = SmoothedValue(fmt='{avg:.6f}')
        space_fmt = ':' + str(len(str(len(iterable)))) + 'd'
        if torch.cuda.is_available():
            log_msg = self.delimiter.join([
                header,
                '[{0' + space_fmt + '}/{1}]',
                'eta: {eta}',
                '{meters}',
                'time: {time}',
                'data: {data}',
                'max mem: {memory:.0f}'
            ])
        else:
            log_msg = self.delimiter.join([
                header,
                '[{0' + space_fmt + '}/{1}]',
                'eta: {eta}',
                '{meters}',
                'time: {time}',
                'data: {data}'
            ])
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    print_str = log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time),
                        memory=torch.cuda.max_memory_allocated() / MB)
                else:
                    print_str = log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time))
                if is_main_process():
                    print(print_str)
                    if log_file is not None:
                        with open(log_file, "a") as f:
                            f.write(print_str + "\n")
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        total_time_msg = '{} Total time: {} ({:.6f} s / it)'.format(
            header, total_time_str, total_time / len(iterable))
        if is_main_process():
            print(total_time_msg)
            if log_file is not None:
                with open(log_file, "a") as f:
                    f.write(total_time_msg + "\n")


class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.6f} ({global_avg:.6f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        if not is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device='cuda')
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value)


class Scheduler_NULL:
    def __init__(self, optimizer):
        self.optimizer = optimizer

    def step(self):
        return


def get_num_params(model):
    total_num = 0
    for parameter in list(model.parameters()):
        dims = parameter.size() + (2,) if parameter.is_complex() else parameter.size()
        total_num += math.prod(dims)
    return total_num


def save_para(arg, config, output_dir):
    para_dir = os.path.join(output_dir, "para")
    os.makedirs(para_dir, exist_ok=True)
    with open(os.path.join(para_dir, "arg.json"), "w") as arg_file:
        json.dump(arg.__dict__, arg_file, indent=2)
    with open(os.path.join(para_dir, "config.json"), "w") as config_file:
        json.dump(config, config_file, indent=2)


def load_json(path):
    with open(path, "r") as file:
        return json.load(file)


def override_args_from_yaml(args, yaml_path):
    with open(yaml_path, "r") as file:
        parameters = yaml.safe_load(file)

    if not isinstance(parameters, dict):
        raise ValueError(f"Expected a parameter mapping in {yaml_path}.")

    unknown_parameters = sorted(set(parameters) - set(vars(args)))
    if unknown_parameters:
        names = ", ".join(unknown_parameters)
        raise ValueError(f"Unknown parameters in {yaml_path}: {names}")

    for name, value in parameters.items():
        setattr(args, name, value)
    return args


def resolve_experiment_dir(exp_name, outputs_root="./outputs"):
    candidate = Path(exp_name).expanduser()
    if candidate.exists():
        return candidate.resolve()

    outputs_root = Path(outputs_root).expanduser().resolve()
    direct = outputs_root / exp_name
    if direct.exists():
        return direct.resolve()

    matches = sorted(path for path in outputs_root.glob(f"*{exp_name}*") if path.is_dir())
    if len(matches) == 1:
        return matches[0].resolve()
    if not matches:
        raise FileNotFoundError(f"Cannot resolve experiment directory for {exp_name}.")
    raise ValueError(
        f"Multiple experiments match {exp_name}: {[path.name for path in matches]}. Please pass a full directory name."
    )


def resolve_checkpoint_path(checkpoint_dir, checkpoint_epoch=None):
    checkpoint_dir = Path(checkpoint_dir)
    if checkpoint_epoch is not None:
        checkpoint_path = checkpoint_dir / f"{checkpoint_epoch}.pt"
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint_path}")
        return checkpoint_path, checkpoint_epoch

    checkpoint_paths = []
    for path in checkpoint_dir.glob("*.pt"):
        try:
            checkpoint_paths.append((int(path.stem), path))
        except ValueError:
            continue

    if not checkpoint_paths:
        # check if there are last.pt
        last_path = checkpoint_dir / "last.pt"
        if last_path.exists():
            return last_path, None
        # raise FileNotFoundError(f"No numeric checkpoints found under {checkpoint_dir}")

    resolved_epoch, resolved_path = max(checkpoint_paths, key=lambda item: item[0])
    return resolved_path, resolved_epoch


def clean_state_dict(state_dict):
    if all(key.startswith("module.") for key in state_dict):
        return {key[len("module."):]: value for key, value in state_dict.items()}
    return state_dict


def get_model_attr(model_name, time_flag=False):
    model_name = model_name.lower()
    return {
        "time": time_flag,
        "single": "_single" in model_name,
    }


def build_model_from_args(arg_dict):
    from . import models

    model_name = arg_dict["model_name"].lower()
    if model_name not in models.__all__:
        raise KeyError(f"Model {model_name} is not registered in utils.models.__all__.")

    return models.__all__[model_name](
        coords_dim=arg_dict["coords_dim"],
        condition_dim=arg_dict["condition_dim"],
        sol_dim=arg_dict["sol_dim"],
        time_proj=arg_dict["time_proj"],
        normed_first_stage=arg_dict["normed_first_stage"],
        out_droprate=arg_dict["out_droprate"],
    )


def load_experiment_meta(exp_name, outputs_root="./outputs"):
    exp_dir = resolve_experiment_dir(exp_name, outputs_root)
    para_dir = exp_dir / "para"
    checkpoint_dir = exp_dir / "checkpoint"
    saved_args = load_json(para_dir / "arg.json")
    saved_config = Dict(load_json(para_dir / "config.json"))
    return exp_dir, saved_args, saved_config, checkpoint_dir


def get_airfoil_physical_window(x_phys, y_phys, field):
    row_slice = slice(40, 180)
    col_slice = slice(None, 35)
    return x_phys[row_slice, col_slice], y_phys[row_slice, col_slice], field[row_slice, col_slice]


def centers_to_corners(center_grid):
    corner_grid = np.empty((center_grid.shape[0] + 1, center_grid.shape[1] + 1), dtype=center_grid.dtype)
    corner_grid[1:-1, 1:-1] = 0.25 * (
        center_grid[:-1, :-1]
        + center_grid[:-1, 1:]
        + center_grid[1:, :-1]
        + center_grid[1:, 1:]
    )

    top_mid = 0.5 * (center_grid[0, :-1] + center_grid[0, 1:])
    bottom_mid = 0.5 * (center_grid[-1, :-1] + center_grid[-1, 1:])
    left_mid = 0.5 * (center_grid[:-1, 0] + center_grid[1:, 0])
    right_mid = 0.5 * (center_grid[:-1, -1] + center_grid[1:, -1])

    corner_grid[0, 1:-1] = 2.0 * top_mid - corner_grid[1, 1:-1]
    corner_grid[-1, 1:-1] = 2.0 * bottom_mid - corner_grid[-2, 1:-1]
    corner_grid[1:-1, 0] = 2.0 * left_mid - corner_grid[1:-1, 1]
    corner_grid[1:-1, -1] = 2.0 * right_mid - corner_grid[1:-1, -2]

    corner_grid[0, 0] = 2.0 * center_grid[0, 0] - corner_grid[1, 1]
    corner_grid[0, -1] = 2.0 * center_grid[0, -1] - corner_grid[1, -2]
    corner_grid[-1, 0] = 2.0 * center_grid[-1, 0] - corner_grid[-2, 1]
    corner_grid[-1, -1] = 2.0 * center_grid[-1, -1] - corner_grid[-2, -2]
    return corner_grid


def get_airfoil_physical_edges(x_phys, y_phys, field):
    x_vis, y_vis, field_vis = get_airfoil_physical_window(x_phys, y_phys, field)
    return centers_to_corners(x_vis), centers_to_corners(y_vis), field_vis


def plot_airfoil_physical_mesh(ax, x_phys, y_phys, field, title, cmap="coolwarm", vmin=None, vmax=None):
    x_edge, y_edge, field_vis = get_airfoil_physical_edges(x_phys, y_phys, field)
    mesh = ax.pcolormesh(x_edge, y_edge, field_vis, shading="flat", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal")
    return mesh


def plot_airfoil_physical_grid(ax, x_phys, y_phys, title):
    x_edge, y_edge, zeros_vis = get_airfoil_physical_edges(x_phys, y_phys, np.zeros_like(x_phys))
    mesh = ax.pcolormesh(
        x_edge,
        y_edge,
        zeros_vis,
        shading="flat",
        cmap="Greys",
        edgecolors="black",
        linewidth=0.1,
    )
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal")
    return mesh


def get_pipe_physical_edges(x_phys, y_phys, field):
    return centers_to_corners(x_phys), centers_to_corners(y_phys), field


def plot_pipe_physical_mesh(ax, x_phys, y_phys, field, title, cmap="coolwarm", vmin=None, vmax=None):
    x_edge, y_edge, field_vis = get_pipe_physical_edges(x_phys, y_phys, field)
    mesh = ax.pcolormesh(x_edge, y_edge, field_vis, shading="flat", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal")
    return mesh


def plot_pipe_physical_grid(ax, x_phys, y_phys, title):
    x_edge, y_edge, zeros_vis = get_pipe_physical_edges(x_phys, y_phys, np.zeros_like(x_phys))
    mesh = ax.pcolormesh(
        x_edge,
        y_edge,
        zeros_vis,
        shading="flat",
        cmap="Greys",
        edgecolors="black",
        linewidth=0.1,
    )
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal")
    return mesh


def plot_plasticity_original_space(ax, state, title, cmap="coolwarm", vmin=None, vmax=None):
    point_x = state[..., 0].reshape(-1)
    point_y = state[..., 1].reshape(-1)
    disp_norm = np.linalg.norm(state[..., 2:], axis=-1).reshape(-1)
    scatter = ax.scatter(point_x, point_y, c=disp_norm, s=10, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal")
    return scatter


def plot_samples(samples, problem_name, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    problem_name = problem_name.lower()

    if problem_name in ['darcy', 'darcy_85']:
        for sample_idx, data in enumerate(samples):
            coords, coeff, sol = data
            coords = coords.detach().cpu().numpy() if torch.is_tensor(coords) else np.asarray(coords)
            coeff = coeff.detach().cpu().numpy() if torch.is_tensor(coeff) else np.asarray(coeff)
            sol = sol.detach().cpu().numpy() if torch.is_tensor(sol) else np.asarray(sol)

            fig, axes = plt.subplots(2, 2, figsize=(10, 8), constrained_layout=True)

            x_map = axes[0, 0].imshow(coords[..., 0], origin="lower", cmap="viridis")
            axes[0, 0].set_title(f"Grid-X #{sample_idx}")
            axes[0, 0].set_xlabel("x")
            axes[0, 0].set_ylabel("y")
            fig.colorbar(x_map, ax=axes[0, 0], fraction=0.046, pad=0.04)

            y_map = axes[0, 1].imshow(coords[..., 1], origin="lower", cmap="viridis")
            axes[0, 1].set_title(f"Grid-Y #{sample_idx}")
            axes[0, 1].set_xlabel("x")
            axes[0, 1].set_ylabel("y")
            fig.colorbar(y_map, ax=axes[0, 1], fraction=0.046, pad=0.04)

            coeff_map = axes[1, 0].imshow(coeff[..., -1], origin="lower", cmap="viridis")
            axes[1, 0].set_title(f"Coeff #{sample_idx}")
            axes[1, 0].set_xlabel("x")
            axes[1, 0].set_ylabel("y")
            fig.colorbar(coeff_map, ax=axes[1, 0], fraction=0.046, pad=0.04)

            sol_map = axes[1, 1].imshow(sol[..., 0], origin="lower", cmap="coolwarm")
            axes[1, 1].set_title(f"Solution #{sample_idx}")
            axes[1, 1].set_xlabel("x")
            axes[1, 1].set_ylabel("y")
            fig.colorbar(sol_map, ax=axes[1, 1], fraction=0.046, pad=0.04)

            fig.savefig(os.path.join(save_dir, f"sample_{sample_idx:03d}.png"), dpi=150)
            plt.close(fig)
    elif problem_name == 'ns2d':
        for sample_idx, data in enumerate(samples):
            coords, sol = data
            coords = coords.detach().cpu().numpy() if torch.is_tensor(coords) else np.asarray(coords)
            sol = sol.detach().cpu().numpy() if torch.is_tensor(sol) else np.asarray(sol)

            total_steps = sol.shape[-2]
            vis_steps = [0, min(9, total_steps - 1), min(10, total_steps - 1), total_steps - 1]
            vis_titles = [
                f"Frame t{vis_steps[0]}",
                f"Frame t{vis_steps[1]}",
                f"Frame t{vis_steps[2]}",
                f"Frame t{vis_steps[3]}",
            ]

            fig, axes = plt.subplots(2, 3, figsize=(12, 7), constrained_layout=True)

            grid_x = axes[0, 0].imshow(coords[..., 0], origin="lower", cmap="viridis")
            axes[0, 0].set_title(f"Grid-X #{sample_idx}")
            fig.colorbar(grid_x, ax=axes[0, 0], fraction=0.046, pad=0.04)

            grid_y = axes[0, 1].imshow(coords[..., 1], origin="lower", cmap="viridis")
            axes[0, 1].set_title(f"Grid-Y #{sample_idx}")
            fig.colorbar(grid_y, ax=axes[0, 1], fraction=0.046, pad=0.04)

            avg_map = axes[0, 2].imshow(np.mean(sol[..., 0], axis=-1), origin="lower", cmap="coolwarm")
            axes[0, 2].set_title(f"Temporal Mean #{sample_idx}")
            fig.colorbar(avg_map, ax=axes[0, 2], fraction=0.046, pad=0.04)

            for axis, step_idx, title in zip(axes[1], vis_steps[:3], vis_titles[:3]):
                field_map = axis.imshow(sol[..., step_idx, 0], origin="lower", cmap="coolwarm")
                axis.set_title(f"{title} #{sample_idx}")
                fig.colorbar(field_map, ax=axis, fraction=0.046, pad=0.04)

            for axis in axes.flat:
                axis.set_xlabel("x")
                axis.set_ylabel("y")

            fig.savefig(os.path.join(save_dir, f"sample_{sample_idx:03d}.png"), dpi=150)
            plt.close(fig)

            fig_tail, axes_tail = plt.subplots(1, 2, figsize=(8, 3.8), constrained_layout=True)
            for axis, step_idx, title in zip(axes_tail, vis_steps[2:], vis_titles[2:]):
                field_map = axis.imshow(sol[..., step_idx, 0], origin="lower", cmap="coolwarm")
                axis.set_title(f"{title} #{sample_idx}")
                axis.set_xlabel("x")
                axis.set_ylabel("y")
                fig_tail.colorbar(field_map, ax=axis, fraction=0.046, pad=0.04)

            fig_tail.savefig(os.path.join(save_dir, f"sample_{sample_idx:03d}_late.png"), dpi=150)
            plt.close(fig_tail)
    elif problem_name == 'pipe':
        for sample_idx, data in enumerate(samples):
            coords, condition, sol = data
            coords = coords.detach().cpu().numpy() if torch.is_tensor(coords) else np.asarray(coords)
            condition = condition.detach().cpu().numpy() if torch.is_tensor(condition) else np.asarray(condition)
            sol = sol.detach().cpu().numpy() if torch.is_tensor(sol) else np.asarray(sol)

            pipe_geom = condition[..., coords.shape[-1]:]

            fig, axes = plt.subplots(2, 3, figsize=(14, 8), constrained_layout=True)

            grid_x = axes[0, 0].imshow(coords[..., 0], origin="lower", cmap="coolwarm")
            axes[0, 0].set_title(f"Comp Grid X #{sample_idx}")
            fig.colorbar(grid_x, ax=axes[0, 0], fraction=0.046, pad=0.04)

            grid_y = axes[0, 1].imshow(coords[..., 1], origin="lower", cmap="coolwarm")
            axes[0, 1].set_title(f"Comp Grid Y #{sample_idx}")
            fig.colorbar(grid_y, ax=axes[0, 1], fraction=0.046, pad=0.04)

            raw_x = axes[0, 2].imshow(pipe_geom[..., 0], origin="lower", cmap="viridis")
            axes[0, 2].set_title(f"Geometry X #{sample_idx}")
            fig.colorbar(raw_x, ax=axes[0, 2], fraction=0.046, pad=0.04)

            raw_y = axes[1, 0].imshow(pipe_geom[..., 1], origin="lower", cmap="viridis")
            axes[1, 0].set_title(f"Geometry Y #{sample_idx}")
            fig.colorbar(raw_y, ax=axes[1, 0], fraction=0.046, pad=0.04)

            target = axes[1, 1].imshow(sol[..., 0], origin="lower", cmap="RdBu_r")
            axes[1, 1].set_title(f"Target u_x #{sample_idx}")
            fig.colorbar(target, ax=axes[1, 1], fraction=0.046, pad=0.04)

            delta = axes[1, 2].imshow(pipe_geom[..., 1] - coords[..., 1], origin="lower", cmap="RdBu_r")
            axes[1, 2].set_title(f"Geometry Y - Comp Y #{sample_idx}")
            fig.colorbar(delta, ax=axes[1, 2], fraction=0.046, pad=0.04)

            for axis in axes.flat:
                axis.set_xlabel("x-index")
                axis.set_ylabel("y-index")

            fig.savefig(os.path.join(save_dir, f"sample_{sample_idx:03d}.png"), dpi=150)
            plt.close(fig)

            x_phys = pipe_geom[..., 0]
            y_phys = pipe_geom[..., 1]
            u_field = sol[..., 0]

            fig_phys, axes_phys = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
            grid_mesh = plot_pipe_physical_grid(axes_phys[0], x_phys, y_phys, f"Physical Mesh #{sample_idx}")
            fig_phys.colorbar(grid_mesh, ax=axes_phys[0], fraction=0.046, pad=0.04)

            u_mesh = plot_pipe_physical_mesh(
                axes_phys[1],
                x_phys,
                y_phys,
                u_field,
                f"u_x on Physical Mesh #{sample_idx}",
                cmap="coolwarm",
            )
            fig_phys.colorbar(u_mesh, ax=axes_phys[1], fraction=0.046, pad=0.04)

            fig_phys.savefig(os.path.join(save_dir, f"sample_{sample_idx:03d}_physical.png"), dpi=150)
            plt.close(fig_phys)
    elif problem_name == 'airfoil':
        for sample_idx, data in enumerate(samples):
            coords, condition, sol = data
            coords = coords.detach().cpu().numpy() if torch.is_tensor(coords) else np.asarray(coords)
            condition = condition.detach().cpu().numpy() if torch.is_tensor(condition) else np.asarray(condition)
            sol = sol.detach().cpu().numpy() if torch.is_tensor(sol) else np.asarray(sol)

            airfoil_geom = condition[..., coords.shape[-1]:]

            fig, axes = plt.subplots(2, 3, figsize=(14, 8), constrained_layout=True)

            grid_x = axes[0, 0].imshow(coords[..., 0], origin="lower", cmap="coolwarm")
            axes[0, 0].set_title(f"Comp Grid X #{sample_idx}")
            fig.colorbar(grid_x, ax=axes[0, 0], fraction=0.046, pad=0.04)

            grid_y = axes[0, 1].imshow(coords[..., 1], origin="lower", cmap="coolwarm")
            axes[0, 1].set_title(f"Comp Grid Y #{sample_idx}")
            fig.colorbar(grid_y, ax=axes[0, 1], fraction=0.046, pad=0.04)

            raw_x = axes[0, 2].imshow(airfoil_geom[..., 0], origin="lower", cmap="viridis")
            axes[0, 2].set_title(f"Geometry X #{sample_idx}")
            fig.colorbar(raw_x, ax=axes[0, 2], fraction=0.046, pad=0.04)

            raw_y = axes[1, 0].imshow(airfoil_geom[..., 1], origin="lower", cmap="viridis")
            axes[1, 0].set_title(f"Geometry Y #{sample_idx}")
            fig.colorbar(raw_y, ax=axes[1, 0], fraction=0.046, pad=0.04)

            target = axes[1, 1].imshow(sol[..., 0], origin="lower", cmap="RdBu_r")
            axes[1, 1].set_title(f"Target c_p #{sample_idx}")
            fig.colorbar(target, ax=axes[1, 1], fraction=0.046, pad=0.04)

            delta = axes[1, 2].imshow(airfoil_geom[..., 1] - coords[..., 1], origin="lower", cmap="RdBu_r")
            axes[1, 2].set_title(f"Geometry Y - Comp Y #{sample_idx}")
            fig.colorbar(delta, ax=axes[1, 2], fraction=0.046, pad=0.04)

            for axis in axes.flat:
                axis.set_xlabel("x-index")
                axis.set_ylabel("y-index")

            fig.savefig(os.path.join(save_dir, f"sample_{sample_idx:03d}.png"), dpi=150)
            plt.close(fig)

            x_phys = airfoil_geom[..., 0]
            y_phys = airfoil_geom[..., 1]
            cp_field = sol[..., 0]

            fig_phys, axes_phys = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
            grid_mesh = plot_airfoil_physical_grid(axes_phys[0], x_phys, y_phys, f"Physical Mesh #{sample_idx}")
            fig_phys.colorbar(grid_mesh, ax=axes_phys[0], fraction=0.046, pad=0.04)

            cp_mesh = plot_airfoil_physical_mesh(
                axes_phys[1],
                x_phys,
                y_phys,
                cp_field,
                f"c_p on Physical Mesh #{sample_idx}",
                cmap="coolwarm",
            )
            fig_phys.colorbar(cp_mesh, ax=axes_phys[1], fraction=0.046, pad=0.04)

            fig_phys.savefig(os.path.join(save_dir, f"sample_{sample_idx:03d}_physical.png"), dpi=150)
            plt.close(fig_phys)
    elif problem_name == 'elasticity':
        for sample_idx, data in enumerate(samples):
            coords, condition, sol = data
            coords = coords.detach().cpu().numpy() if torch.is_tensor(coords) else np.asarray(coords)
            sol = sol.detach().cpu().numpy() if torch.is_tensor(sol) else np.asarray(sol)

            fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)

            coord_x = axes[0].scatter(coords[:, 0], coords[:, 1], c=coords[:, 0], s=10, cmap="viridis")
            axes[0].set_title(f"Point X #{sample_idx}")
            fig.colorbar(coord_x, ax=axes[0], fraction=0.046, pad=0.04)

            coord_y = axes[1].scatter(coords[:, 0], coords[:, 1], c=coords[:, 1], s=10, cmap="coolwarm")
            axes[1].set_title(f"Point Y #{sample_idx}")
            fig.colorbar(coord_y, ax=axes[1], fraction=0.046, pad=0.04)

            sigma_map = axes[2].scatter(coords[:, 0], coords[:, 1], c=sol[:, 0], s=10, cmap="magma")
            axes[2].set_title(f"Sigma #{sample_idx}")
            fig.colorbar(sigma_map, ax=axes[2], fraction=0.046, pad=0.04)

            for axis in axes:
                axis.set_xlabel("x")
                axis.set_ylabel("y")
                axis.set_aspect("equal")

            fig.savefig(os.path.join(save_dir, f"sample_{sample_idx:03d}.png"), dpi=150)
            plt.close(fig)
    elif problem_name == 'plasticity':
        for sample_idx, data in enumerate(samples):
            coords, condition, sol, t = data
            coords = coords.detach().cpu().numpy() if torch.is_tensor(coords) else np.asarray(coords)
            condition = condition.detach().cpu().numpy() if torch.is_tensor(condition) else np.asarray(condition)
            sol = sol.detach().cpu().numpy() if torch.is_tensor(sol) else np.asarray(sol)
            t = t.detach().cpu().numpy() if torch.is_tensor(t) else np.asarray(t)

            raw_condition = condition[..., coords.shape[-1]:]
            vis_steps = [0, sol.shape[-2] // 2, sol.shape[-2] - 1]

            fig, axes = plt.subplots(2, 3, figsize=(13, 8), constrained_layout=True)

            coord_x = axes[0, 0].imshow(coords[..., 0], origin="lower", cmap="viridis")
            axes[0, 0].set_title(f"Coord X #{sample_idx}")
            fig.colorbar(coord_x, ax=axes[0, 0], fraction=0.046, pad=0.04)

            coord_y = axes[0, 1].imshow(coords[..., 1], origin="lower", cmap="viridis")
            axes[0, 1].set_title(f"Coord Y #{sample_idx}")
            fig.colorbar(coord_y, ax=axes[0, 1], fraction=0.046, pad=0.04)

            cond_map = axes[0, 2].imshow(raw_condition[..., 0], origin="lower", cmap="coolwarm")
            axes[0, 2].set_title(f"Condition #{sample_idx}")
            fig.colorbar(cond_map, ax=axes[0, 2], fraction=0.046, pad=0.04)

            for axis, step_idx in zip(axes[1], vis_steps):
                disp_norm = np.linalg.norm(sol[..., step_idx, 2:], axis=-1)
                field_map = axis.imshow(disp_norm, origin="lower", cmap="RdBu_r")
                axis.set_title(f"||disp|| t={t[step_idx]:.2f} #{sample_idx}")
                fig.colorbar(field_map, ax=axis, fraction=0.046, pad=0.04)

            for axis in axes.flat:
                axis.set_xlabel("x")
                axis.set_ylabel("y")

            fig.savefig(os.path.join(save_dir, f"sample_{sample_idx:03d}.png"), dpi=150)
            plt.close(fig)

            fig_phys, axes_phys = plt.subplots(1, 3, figsize=(14, 4.5), constrained_layout=True)
            disp_values = [np.linalg.norm(sol[..., step_idx, 2:], axis=-1) for step_idx in vis_steps]
            vmin = min(value.min() for value in disp_values)
            vmax = max(value.max() for value in disp_values)

            for axis, step_idx in zip(axes_phys, vis_steps):
                scatter = plot_plasticity_original_space(
                    axis,
                    sol[..., step_idx, :],
                    f"Original Space ||disp|| t={t[step_idx]:.2f} #{sample_idx}",
                    vmin=vmin,
                    vmax=vmax,
                )
                fig_phys.colorbar(scatter, ax=axis, fraction=0.046, pad=0.04)

            fig_phys.savefig(os.path.join(save_dir, f"sample_{sample_idx:03d}_physical.png"), dpi=150)
            plt.close(fig_phys)
    else:
        print("Visualization for {} is not supported yet.".format(problem_name))


def get_dataloaders(config):
    data_root = getattr(config.data, "data_root", "./dataset")

    def time_embedding_collate_fn(batch):
        # only for time-embedding dataset
        # shuffle the temporal dimension within each sample to improve training stability
        collated = []
        coords_batch = []
        condition_batch = []
        sol_batch = []
        t_batch = []

        for coords, condition, sol, t in batch:
            permuted_indices = torch.randperm(t.size(0))
            coords_batch.append(coords)
            condition_batch.append(condition)
            sol_batch.append(sol.index_select(dim=-2, index=permuted_indices))
            t_batch.append(t.index_select(dim=0, index=permuted_indices))

        collated.append(torch.stack(coords_batch, dim=0))
        collated.append(torch.stack(condition_batch, dim=0))
        collated.append(torch.stack(sol_batch, dim=0))
        collated.append(torch.stack(t_batch, dim=0))
        return collated

    # build training dataloader
    if config.data.type == "static":
        dataset_cls = StaticDataset
    elif config.data.type == "airfrans":
        dataset_cls = AirfRANSDataset
    elif config.data.type == "rollout":
        dataset_cls = RolloutDataset
    elif config.data.type == "time_embedding":
        dataset_cls = TimeEmbeddingDataset
    else:
        raise NotImplementedError(f"Data type {config.data.type} is not supported.")

    train_dataset_kwargs = {
        "data_mode": "train",
        "data_root": os.path.join(data_root, config.data.name),
        "apply_norm": config.data.apply_norm,
    }
    if config.data.type == "airfrans":
        train_dataset_kwargs.update(
            {
                "subsampling": config.data.airfrans_subsampling,
                "sampling_mode": config.data.airfrans_sampling_mode,
                "sampling_seed": getattr(config.data, "seed", 0),
            }
        )
    if config.data.type in ["static", "time_embedding"]:
        train_dataset_kwargs["coords_condition"] = config.data.coords_condition
    train_dataset = dataset_cls(**train_dataset_kwargs)
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, shuffle=True)
    train_collate_fn = time_embedding_collate_fn if config.data.type == "time_embedding" else None
    train_dataloader = torch.utils.data.dataloader.DataLoader(
        dataset=train_dataset,
        sampler=train_sampler,
        batch_size=config.data.train_batch_size,
        drop_last=True,
        pin_memory=True,
        shuffle=False,
        num_workers=8,
        collate_fn=train_collate_fn,
    )

    normalizer = train_dataset.get_normalizer()

    # build validation dataloader
    val_dataset_kwargs = {
        "data_mode": "test",
        "data_root": os.path.join(data_root, config.data.name),
        "apply_norm": config.data.apply_norm,
        "norm_params": normalizer.export(),
    }
    if config.data.type == "airfrans":
        val_dataset_kwargs.update(
            {
                "subsampling": config.data.airfrans_eval_subsampling,
                "sampling_mode": config.data.airfrans_eval_sampling_mode,
                "sampling_seed": getattr(config.data, "seed", 0),
            }
        )
    if config.data.type in ["static", "time_embedding"]:
        val_dataset_kwargs["coords_condition"] = config.data.coords_condition
    val_dataset = dataset_cls(**val_dataset_kwargs)
    val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False)
    val_dataloader = torch.utils.data.dataloader.DataLoader(
        dataset=val_dataset,
        sampler=val_sampler,
        batch_size=config.data.val_batch_size,
        drop_last=False,
        pin_memory=True,
        shuffle=False,
        num_workers=8
    )

    if config.data.type in ["static", "airfrans"]:
        coords_dim, condition_dim, sol_dim = train_dataset.dim()
        dim_dict = {
            "coords_dim": coords_dim,
            "condition_dim": condition_dim,
            "sol_dim": sol_dim,
        }
    elif config.data.type == "rollout":
        coords_dim, sol_dim = train_dataset.dim()
        condition_dim = config.data.rollout_history * sol_dim
        if config.data.coords_condition:
            condition_dim += coords_dim
        dim_dict = {
            "coords_dim": coords_dim,
            "condition_dim": condition_dim,
            "sol_dim": sol_dim,
            "total_steps": train_dataset.total_steps(),
        }
    else:
        coords_dim, condition_dim, sol_dim = train_dataset.dim()
        dim_dict = {
            "coords_dim": coords_dim,
            "condition_dim": condition_dim,
            "sol_dim": sol_dim,
            "total_steps": train_dataset.total_steps(),
        }

    return train_dataloader, val_dataloader, normalizer, dim_dict


def get_optimizer(config, model, sch_max_step=None):
    if config.optimizer.name == "Adam":
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config.optimizer.lr,
            weight_decay=config.optimizer.weight_decay,
            betas=(0.9, 0.99),
        )
    elif config.optimizer.name == "AdamW":
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.optimizer.lr,
            weight_decay=config.optimizer.weight_decay,
            betas=(0.9, 0.99),
        )
    elif config.optimizer.name == "SGD":
        optimizer = torch.optim.SGD(model.parameters(), lr=config.optimizer.lr)
    else:
        raise NotImplementedError("Invalid Optimizer !")

    if config.scheduler.name == "NULL":
        scheduler = Scheduler_NULL(optimizer)
    elif config.scheduler.name == "Cos":
        assert sch_max_step is not None, "sch_max_step must be provided for Cos scheduler"
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=sch_max_step,
            eta_min=1e-6,
        )
    elif config.scheduler.name == "OneCycle":
        assert sch_max_step is not None, "sch_max_step must be provided for OneCycle scheduler"
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=config.optimizer.lr,
            pct_start=config.scheduler.pct_start,
            div_factor=config.scheduler.div_factor,
            final_div_factor=config.scheduler.final_div_factor,
            total_steps=sch_max_step
        )
    else:
        raise NotImplementedError("Invalid Scheduler !")

    return optimizer, scheduler


def is_dist_avail_and_initialized():
    if not torch.distributed.is_available():
        return False
    if not torch.distributed.is_initialized():
        return False
    return True


def is_main_process():
    if not is_dist_avail_and_initialized():
        return True
    return torch.distributed.get_rank() == 0


def setup_for_distributed(is_master):
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop("force", False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def init_distributed_mode(args):
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ["WORLD_SIZE"])
        args.gpu = int(os.environ["LOCAL_RANK"])
    elif "SLURM_PROCID" in os.environ:
        args.rank = int(os.environ["SLURM_PROCID"])
        args.gpu = args.rank % torch.cuda.device_count()
        args.world_size = int(os.environ.get("WORLD_SIZE", torch.cuda.device_count()))
    elif torch.cuda.is_available():
        args.rank = 0
        args.gpu = 0
        args.world_size = 1
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", str(random.randint(29000, 29599)))
        os.environ.setdefault("RANK", "0")
        os.environ.setdefault("WORLD_SIZE", "1")
        os.environ.setdefault("LOCAL_RANK", "0")
    else:
        print("Does not support training without GPU.")
        sys.exit(1)

    torch.distributed.init_process_group(
        backend="nccl",
        init_method=args.dist_url,
        world_size=args.world_size,
        rank=args.rank,
    )
    torch.cuda.set_device(args.gpu)
    torch.distributed.barrier()
    setup_for_distributed(args.rank == 0)


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # torch.use_deterministic_algorithms(True)
