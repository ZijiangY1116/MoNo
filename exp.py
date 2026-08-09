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
import datetime
import json
import os

import torch
from torch.utils.tensorboard import SummaryWriter

from eval import eval_main
from utils import basic_utils
from utils.loss import LossManager


def build_parser():
    parser = argparse.ArgumentParser(description="Local-Enhanced Latent Neural Operator")

    # basic parameters
    basic_group = parser.add_argument_group("basic")
    basic_group.add_argument("--config", type=str, default=None, help="YAML file used to override command-line parameters.")
    basic_group.add_argument("--exp", type=str, default=None, help="Experiment name.")
    basic_group.add_argument("--exp_root", type=str, default="./outputs", help="Root directory for logs, checkpoints, and saved args.")
    basic_group.add_argument("--seed", type=int, default=0, help="Random seed.")
    basic_group.add_argument("--dist_url", type=str, default="env://", help="Distributed init URL.")

    # data parameters
    data_group = parser.add_argument_group("data")
    data_group.add_argument("--data_name", type=str, default=None, help="Dataset name.")
    data_group.add_argument("--data_root", type=str, default="./dataset", help="Dataset root directory.")
    data_group.add_argument("--train_batch_size", type=int, default=4, help="Training batch size per GPU.")
    data_group.add_argument("--val_batch_size", type=int, default=4, help="Validation batch size per GPU.")
    data_group.add_argument("--rollout_history", type=int, default=10, help="Number of history frames used as rollout input.")
    data_group.add_argument("--airfrans_subsampling", type=int, default=32000, help="Number of AirfRANS points sampled per training case.")
    data_group.add_argument("--airfrans_eval_subsampling", type=int, default=32000, help="Number of AirfRANS points sampled per test case during eval.")
    data_group.add_argument("--airfrans_sampling_mode", type=str, default="all_surface", choices=["all_surface"], help="AirfRANS training point sampling mode.")
    data_group.add_argument("--airfrans_eval_sampling_mode", type=str, default="all_surface", choices=["all_surface"], help="AirfRANS eval point sampling mode.")
    data_group.add_argument("--airfrans_raw_root", type=str, default=None, help="Raw AirfRANS Dataset directory used for CL/rho_l evaluation.")
    # model parameters
    model_group = parser.add_argument_group("model")
    model_group.add_argument("--model_name", type=str, default="mono", choices=["mono-light", "mono"], help="MONO profile: mono-light uses 512 modes/96 channels; mono uses 1024 modes/192 channels.")
    model_group.add_argument("--normed_first_stage", action=argparse.BooleanOptionalAction, default=True, help="Use the row-normalized routing projection in the first stage.")
    model_group.add_argument("--out_droprate", type=float, default=0.0, help="Output MLP dropout rate.")
    model_group.add_argument("--load_ckpt", type=str, default=None, help="Optional checkpoint path used to initialize model weights.")
    model_group.add_argument("--strict_ckpt_load", action="store_true", help="Use strict checkpoint loading. Defaults to shape-safe non-strict loading.")
    model_group.add_argument("--freeze_loaded_params", action="store_true", help="Freeze trainable parameters that are initialized from --load_ckpt.")

    # loss parameters
    loss_group = parser.add_argument_group("loss")
    loss_group.add_argument("--loss_name", type=str, default="rL2", help="Loss function expression, e.g. rL2 or DarcyDeriv.")
    loss_group.add_argument("--loss_weight", type=str, default="1", help="Loss weight expression aligned with loss_name.")
    loss_group.add_argument("--airfrans_surface_weight", type=float, default=1.0, help="Surface-loss weight for AirfRANS MSE_weighted training.")

    # optimizer parameters
    optimizer_group = parser.add_argument_group("optimizer")
    optimizer_group.add_argument("--optimizer_name", type=str, default="AdamW", help="Optimizer type.")
    optimizer_group.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    optimizer_group.add_argument("--weight_decay", type=float, default=5e-5, help="Weight decay.")

    # scheduler parameters
    scheduler_group = parser.add_argument_group("scheduler")
    scheduler_group.add_argument("--scheduler_name", type=str, default="Cos", choices=["Cos", "OneCycle"], help="Scheduler type.")
    scheduler_group.add_argument("--min_lr", type=float, default=1e-5, help="Minimum learning rate for Cosine scheduler.")
    scheduler_group.add_argument("--warmup_epochs", type=int, default=10, help="Number of warmup epochs for Cosine scheduler.")
    scheduler_group.add_argument("--pct_start", type=float, default=0.2, help="OneCycle warmup fraction.")

    # training parameters
    train_group = parser.add_argument_group("train")
    train_group.add_argument("--epoch", type=int, default=500, help="Number of epochs.")
    train_group.add_argument("--log_print_interval_epoch", type=int, default=1, help="Logging interval in epochs.")
    train_group.add_argument("--model_save_interval_epoch", type=int, default=-1, help="Checkpoint interval in epochs.")
    train_group.add_argument("--grad_clip", type=float, default=1000.0, help="Gradient clipping threshold.")
    train_group.add_argument("--n_train_vis", type=int, default=4, help="Number of random train samples to visualize.")
    train_group.add_argument(
        "--rollout_autoreg_ratio",
        type=float,
        default=0.0,
        help="Target autoregressive ratio used by rollout scheduled sampling.",
    )

    return parser


def build_config(args):
    model_save_interval_epoch = args.epoch if args.model_save_interval_epoch == -1 else args.model_save_interval_epoch

    data_type_map = {
        "darcy": "static",
        "darcy_43": "static",
        "darcy_85": "static",
        "darcy_141": "static",
        "darcy_211": "static",
        "darcy_241": "static",
        "airfoil": "static",
        "airfrans_full": "airfrans",
        "airfrans_scarce": "airfrans",
        "airfrans_reynolds": "airfrans",
        "airfrans_aoa": "airfrans",
		"pipe": "static",
		"elasticity": "static",
        "ns2d": "rollout",
        "plasticity": "time_embedding",
    }

    data_norm_map = {
        "darcy": True,
        "darcy_43": True,
        "darcy_85": True,
        "darcy_141": True,
        "darcy_211": True,
        "darcy_241": True,
        "airfoil": False,
        "airfrans_full": True,
        "airfrans_scarce": True,
        "airfrans_reynolds": True,
        "airfrans_aoa": True,
        "pipe": False,
        "elasticity": True,
        "ns2d": False,
        "plasticity": True,
    }

    data_coords_condition = {
        "darcy": True,
        "darcy_43": True,
        "darcy_85": True,
        "darcy_141": True,
        "darcy_211": True,
        "darcy_241": True,
        "airfoil": True,
        "airfrans_full": False,
        "airfrans_scarce": False,
        "airfrans_reynolds": False,
        "airfrans_aoa": False,
        "pipe": True,
        "elasticity": False,
        "ns2d": True,
        "plasticity": True,
    }

    data_type = data_type_map[args.data_name.lower()]
    time_proj = (data_type == "time_embedding")

    return basic_utils.Dict(
        {
            "data": {
                "name": args.data_name,
                "type": data_type,
                "seed": args.seed,
                "data_root": args.data_root,
                "train_batch_size": args.train_batch_size,
                "val_batch_size": args.val_batch_size,
                "apply_norm": data_norm_map[args.data_name.lower()],
                "coords_condition": data_coords_condition[args.data_name.lower()],
                "rollout_history": args.rollout_history,  # only for rollout dataset
                "airfrans_subsampling": args.airfrans_subsampling,
                "airfrans_eval_subsampling": None if args.airfrans_eval_sampling_mode == "all" else args.airfrans_eval_subsampling,
                "airfrans_sampling_mode": args.airfrans_sampling_mode,
                "airfrans_eval_sampling_mode": args.airfrans_eval_sampling_mode,
                "airfrans_raw_root": args.airfrans_raw_root,
            },
            "model": {
                "name": args.model_name,
                "normed_first_stage": args.normed_first_stage,
                "out_droprate": args.out_droprate,
                "time_proj": time_proj,
            },
            "loss": {
                "name": args.loss_name,
                "weight": args.loss_weight,
                "airfrans_surface_weight": args.airfrans_surface_weight,
            },
            "optimizer": {
                "name": args.optimizer_name,
                "lr": args.lr,
                "weight_decay": args.weight_decay,
            },
            "scheduler": {
                "name": args.scheduler_name,
                "div_factor": args.lr / args.min_lr,  # for OneCycle scheduler
                "final_div_factor": args.lr / args.min_lr,  # for OneCycle scheduler
                "pct_start": args.warmup_epochs / args.epoch if args.epoch > 0 else 0,  # for OneCycle scheduler
            },
            "train": {
                "epoch": args.epoch,
                "log_print_interval_epoch": args.log_print_interval_epoch,
                "model_save_interval_epoch": model_save_interval_epoch,
                "grad_clip": args.grad_clip,
                "rollout_autoreg_ratio": args.rollout_autoreg_ratio,  # only for rollout dataset
            },
        }
    )


def load_model_checkpoint_for_init(model, checkpoint_path, strict=False):
    raw_state_dict = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(raw_state_dict, dict):
        if "state_dict" in raw_state_dict:
            raw_state_dict = raw_state_dict["state_dict"]
        elif "model" in raw_state_dict:
            raw_state_dict = raw_state_dict["model"]
    checkpoint_state = basic_utils.clean_state_dict(raw_state_dict)

    if strict:
        model.load_state_dict(checkpoint_state, strict=True)
        model_state = model.state_dict()
        loaded_keys = [
            key for key, value in checkpoint_state.items()
            if key in model_state and model_state[key].shape == value.shape
        ]
        if basic_utils.is_main_process():
            print(f"Loaded checkpoint strictly from {checkpoint_path}")
            print(f"  loaded keys: {len(loaded_keys)}")
        return {
            "loaded_keys": loaded_keys,
            "missing_keys": [],
            "unexpected_keys": [],
            "shape_mismatch_keys": [],
        }

    model_state = model.state_dict()
    loadable_state = {}
    unexpected_keys = []
    shape_mismatch_keys = []

    for key, value in checkpoint_state.items():
        if key not in model_state:
            unexpected_keys.append(key)
        elif model_state[key].shape != value.shape:
            shape_mismatch_keys.append((key, tuple(value.shape), tuple(model_state[key].shape)))
        else:
            loadable_state[key] = value

    missing_keys = [key for key in model_state.keys() if key not in loadable_state]
    updated_state = dict(model_state)
    updated_state.update(loadable_state)
    model.load_state_dict(updated_state, strict=True)

    if basic_utils.is_main_process():
        print(f"Loaded checkpoint non-strictly from {checkpoint_path}")
        print(f"  loaded keys: {len(loadable_state)}")
        print(f"  missing keys: {len(missing_keys)}")
        print(f"  unexpected keys: {len(unexpected_keys)}")
        print(f"  shape mismatch keys: {len(shape_mismatch_keys)}")
        if len(missing_keys) > 0:
            print("  missing key examples:")
            for key in missing_keys[:20]:
                print(f"    {key}")
        if len(unexpected_keys) > 0:
            print("  unexpected key examples:")
            for key in unexpected_keys[:20]:
                print(f"    {key}")
        if len(shape_mismatch_keys) > 0:
            print("  shape mismatch examples:")
            for key, ckpt_shape, model_shape in shape_mismatch_keys[:20]:
                print(f"    {key}: checkpoint {ckpt_shape} vs model {model_shape}")

    return {
        "loaded_keys": list(loadable_state.keys()),
        "missing_keys": missing_keys,
        "unexpected_keys": unexpected_keys,
        "shape_mismatch_keys": shape_mismatch_keys,
    }


def freeze_loaded_parameters(model, loaded_keys):
    loaded_key_set = set(loaded_keys)
    frozen_names = []
    for name, parameter in model.named_parameters():
        if name in loaded_key_set:
            if "out_mlp" in name:
                continue
            parameter.requires_grad_(False)
            frozen_names.append(name)

    if basic_utils.is_main_process():
        print(f"Frozen loaded trainable parameters: {len(frozen_names)}")
        if len(frozen_names) > 0:
            print("  frozen parameter examples:")
            for name in frozen_names[:20]:
                print(f"    {name}")

    return frozen_names


def write_epoch_log(log_file, stats):
    if basic_utils.is_main_process():
        with open(log_file, "a") as f:
            f.write(json.dumps(stats) + "\n")


def train(train_dataloader, normalizer, model, loss_fn, optimizer, scheduler, world_size, grad_clip, epoch, log_print_interval_epoch, model_save_interval_epoch, log_dir, checkpoint_dir):
    log_file = os.path.join(log_dir, "log.txt")
    step_log_file = os.path.join(log_dir, "log-step.txt")

    if basic_utils.is_main_process():
        writer = SummaryWriter(log_dir)
        checker = basic_utils.Checkpoint(checkpoint_dir, model)
        print("Number of Model Parameters: {}".format(basic_utils.get_num_params(model)))
        print(model)
        print("Start Training...")

    for current_epoch in range(epoch):
        torch.distributed.barrier()
        train_dataloader.sampler.set_epoch(current_epoch)
        metric_logger = basic_utils.MetricLogger(delimiter="  ")
        header = "Epoch: [{}/{}]".format(current_epoch + 1, epoch)

        for data in metric_logger.log_every(train_dataloader, 10, header, step_log_file):
            coords, condition, sol = data
            coords = coords.cuda()
            coords = torch.reshape(coords, (coords.shape[0], -1, coords.shape[-1]))
            condition = condition.cuda()
            condition = torch.reshape(condition, (condition.shape[0], -1, condition.shape[-1]))
            sol = sol.cuda()
            sol = torch.reshape(sol, (sol.shape[0], -1, sol.shape[-1]))

            model.train()
            res = model(coords, condition)

            if normalizer.is_apply_y2():
                # de-normalize the prediction and solution to physical scale for loss computation
                res = normalizer.apply_y2(res, inverse=True)
                sol = normalizer.apply_y2(sol, inverse=True)

            loss_dict = loss_fn.compute_loss(res, sol)
            loss = loss_fn.get_total_loss(loss_dict)

            optimizer.zero_grad()
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

            if scheduler.__class__.__name__ == "OneCycleLR":
                # for OneCycleLR, step for each iteration in the entire training process
                scheduler.step()

            metric_dict = basic_utils.build_train_metric_dict(loss_dict, loss, world_size)
            update_kwargs = {**metric_dict, "lr": optimizer.param_groups[0]["lr"]}
            metric_logger.update(**update_kwargs)
        
        if scheduler.__class__.__name__ != "OneCycleLR":
            # for other schedulers, step for each epoch
            scheduler.step()

        metric_logger.synchronize_between_processes()
        train_stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
        train_loss = train_stats["loss"]
        learning_rate = optimizer.state_dict()["param_groups"][0]["lr"]

        if basic_utils.is_main_process():
            if (current_epoch + 1) % log_print_interval_epoch == 0:
                for name, value in train_stats.items():
                    writer.add_scalar(f"Train/{name}", value, current_epoch + 1)
                writer.add_scalar("Learning Rate", learning_rate, current_epoch + 1)
                print(
                    "Epoch: {}\tLearning Rate :{}\tTrain Loss: {}".format(
                        current_epoch + 1,
                        learning_rate,
                        train_loss,
                    )
                )

            log_stats = {
                "epoch": current_epoch + 1,
                **{"train_{}".format(k): v for k, v in train_stats.items()},
            }
            write_epoch_log(log_file, log_stats)

            if (current_epoch + 1) % model_save_interval_epoch == 0:
                checker.save(current_epoch + 1)

    if basic_utils.is_main_process():
        writer.close()
        print("Finish Training !")


def compute_airfrans_mse_loss(pred, target, surf, surface_weight):
    squared_error = (pred - target) ** 2
    loss_per_var = squared_error.mean(dim=(0, 1))
    loss_all = loss_per_var.mean()

    surf_mask = surf.bool()
    vol_mask = ~surf_mask
    zero = squared_error.new_tensor(0.0)

    if surf_mask.any():
        loss_surf_var = squared_error[surf_mask].mean(dim=0)
        loss_surf = loss_surf_var.mean()
    else:
        loss_surf_var = squared_error.new_zeros(squared_error.shape[-1])
        loss_surf = zero

    if vol_mask.any():
        loss_vol_var = squared_error[vol_mask].mean(dim=0)
        loss_vol = loss_vol_var.mean()
    else:
        loss_vol_var = squared_error.new_zeros(squared_error.shape[-1])
        loss_vol = zero

    loss = loss_vol + surface_weight * loss_surf
    loss_dict = {
        "AirfRANS": loss_all,
        "AirfRANS_surf": loss_surf,
        "AirfRANS_vol": loss_vol,
        "AirfRANS_u_x": loss_per_var[0],
        "AirfRANS_u_y": loss_per_var[1],
        "AirfRANS_p": loss_per_var[2],
        "AirfRANS_nut": loss_per_var[3],
        "AirfRANS_surf_p": loss_surf_var[2],
        "AirfRANS_vol_p": loss_vol_var[2],
    }
    return loss, loss_dict


def train_airfrans(train_dataloader, model, optimizer, scheduler, world_size, grad_clip, epoch, surface_weight, log_print_interval_epoch, model_save_interval_epoch, log_dir, checkpoint_dir):
    log_file = os.path.join(log_dir, "log.txt")
    step_log_file = os.path.join(log_dir, "log-step.txt")

    if basic_utils.is_main_process():
        writer = SummaryWriter(log_dir)
        checker = basic_utils.Checkpoint(checkpoint_dir, model)
        print("Number of Model Parameters: {}".format(basic_utils.get_num_params(model)))
        print(model)
        print("Start AirfRANS Training...")

    for current_epoch in range(epoch):
        torch.distributed.barrier()
        train_dataloader.sampler.set_epoch(current_epoch)
        metric_logger = basic_utils.MetricLogger(delimiter="  ")
        header = "Epoch: [{}/{}]".format(current_epoch + 1, epoch)

        for data in metric_logger.log_every(train_dataloader, 10, header, step_log_file):
            coords, condition, sol, surf = data
            coords = coords.cuda()
            condition = condition.cuda()
            sol = sol.cuda()
            surf = surf.cuda()

            model.train()
            pred = model(coords, condition)

            loss, loss_dict = compute_airfrans_mse_loss(pred, sol, surf, surface_weight)

            optimizer.zero_grad()
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

            if scheduler.__class__.__name__ == "OneCycleLR":
                scheduler.step()

            metric_dict = basic_utils.build_train_metric_dict(loss_dict, loss, world_size)
            update_kwargs = {**metric_dict, "lr": optimizer.param_groups[0]["lr"]}
            metric_logger.update(**update_kwargs)

        if scheduler.__class__.__name__ != "OneCycleLR":
            scheduler.step()

        metric_logger.synchronize_between_processes()
        train_stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
        learning_rate = optimizer.state_dict()["param_groups"][0]["lr"]

        if basic_utils.is_main_process():
            if (current_epoch + 1) % log_print_interval_epoch == 0:
                for name, value in train_stats.items():
                    writer.add_scalar(f"Train/{name}", value, current_epoch + 1)
                writer.add_scalar("Learning Rate", learning_rate, current_epoch + 1)
                print(
                    "Epoch: {}\tLearning Rate :{}\tTrain Loss: {}\tSurf Loss: {}\tVol Loss: {}".format(
                        current_epoch + 1,
                        learning_rate,
                        train_stats["loss"],
                        train_stats["loss_AirfRANS_surf"],
                        train_stats["loss_AirfRANS_vol"],
                    )
                )

            log_stats = {
                "epoch": current_epoch + 1,
                **{"train_{}".format(k): v for k, v in train_stats.items()},
            }
            write_epoch_log(log_file, log_stats)

            if (current_epoch + 1) % model_save_interval_epoch == 0:
                checker.save(current_epoch + 1)

    if basic_utils.is_main_process():
        writer.close()
        print("Finish AirfRANS Training !")


def train_rollout(train_dataloader, normalizer, model, loss_fn, optimizer, scheduler, world_size, grad_clip, epoch, rollout_history, coords_condition, rollout_autoreg_ratio, log_print_interval_epoch, model_save_interval_epoch, log_dir, checkpoint_dir):
    log_file = os.path.join(log_dir, "log.txt")
    step_log_file = os.path.join(log_dir, "log-step.txt")

    if basic_utils.is_main_process():
        writer = SummaryWriter(log_dir)
        checker = basic_utils.Checkpoint(checkpoint_dir, model)
        print("Number of Model Parameters: {}".format(basic_utils.get_num_params(model)))
        print(model)
        print("Start Rollout Training...")

    for current_epoch in range(epoch):
        torch.distributed.barrier()
        current_autoreg_ratio = basic_utils.get_rollout_autoreg_ratio(
            current_epoch=current_epoch,
            total_epoch=epoch,
            target_ratio=rollout_autoreg_ratio
        )
        train_dataloader.sampler.set_epoch(current_epoch)
        metric_logger = basic_utils.MetricLogger(delimiter="  ")
        header = "Epoch: [{}/{}]".format(current_epoch + 1, epoch)

        for data in metric_logger.log_every(train_dataloader, 10, header, step_log_file):
            coords, sol = data
            coords = coords.cuda()
            coords = torch.reshape(coords, (coords.shape[0], -1, coords.shape[-1]))  # [B, L, C]
            sol = sol.cuda()
            sol = torch.reshape(sol, (sol.shape[0], -1, sol.shape[-2], sol.shape[-1]))  # [B, L, T, C]

            # basic check for rollout training
            total_steps = sol.shape[-2]
            if total_steps <= rollout_history:
                raise ValueError(f"rollout_history={rollout_history} must be smaller than total steps {total_steps}.")

            # get the history frames and target frames for the rollout training
            history = sol[:, :, :rollout_history, :]  # [B, L, T_his, C]
            target = sol[:, :, rollout_history:, :]  # [B, L, T_pred, C]
            pred_steps = []
            loss_items_step = {}

            model.train()
            for step_index in range(target.shape[-2]):
                gt_step = target[:, :, step_index, :]  # [B, L, C]

                # =========
                # main forward
                # =========
                condition = history.reshape(history.shape[0], history.shape[1], -1)  # [B, L, T_his*C]
                if coords_condition:
                    condition = torch.cat((coords, condition), dim=-1)  # [B, L, C+T_his*C]
                pred_step = model(coords, condition)
                pred_steps.append(pred_step.unsqueeze(-2))

                # =========
                # loss computation
                # =========
                if normalizer.is_apply_y2():
                    pred_step_loss = normalizer.apply_y2(pred_step, inverse=True)
                    gt_step_loss = normalizer.apply_y2(gt_step, inverse=True)
                else:
                    pred_step_loss = pred_step
                    gt_step_loss = gt_step
                step_loss_dict = loss_fn.compute_loss(pred_step_loss, gt_step_loss)
                for loss_name, loss_value in step_loss_dict.items():
                    loss_items_step[loss_name] = loss_items_step.get(loss_name, 0.0) + loss_value

                # =========
                # update history
                # ========
                history, _ = basic_utils.update_rollout_history_with_scheduled_sampling(
                    history=history,
                    gt_step=gt_step,
                    pred_step=pred_step,
                    autoreg_ratio=current_autoreg_ratio,
                )

            train_loss_step = loss_fn.get_total_loss(loss_items_step)

            optimizer.zero_grad()
            train_loss_step.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

            with torch.no_grad():
                pred_full = torch.cat(pred_steps, dim=-2)
                if normalizer.is_apply_y2():
                    pred_full_loss = normalizer.apply_y2(pred_full, inverse=True)
                    target_loss = normalizer.apply_y2(target, inverse=True)
                else:
                    pred_full_loss = pred_full
                    target_loss = target
                loss_items_full = loss_fn.compute_loss(pred_full_loss, target_loss)
                train_loss_full = loss_fn.get_total_loss(loss_items_full)

            if scheduler.__class__.__name__ == "OneCycleLR":
                scheduler.step()

            metric_dict = basic_utils.build_train_metric_dict(
                {**loss_items_step, "full": train_loss_full.detach(), "step": train_loss_step.detach()},
                train_loss_step,
                world_size,
            )
            update_kwargs = {**metric_dict, "lr": optimizer.param_groups[0]["lr"]}
            metric_logger.update(**update_kwargs)
            metric_logger.update(autoreg_ratio=current_autoreg_ratio)

        if scheduler.__class__.__name__ != "OneCycleLR":
            scheduler.step()

        metric_logger.synchronize_between_processes()
        train_stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
        learning_rate = optimizer.state_dict()["param_groups"][0]["lr"]

        if basic_utils.is_main_process():
            if (current_epoch + 1) % log_print_interval_epoch == 0:
                for name, value in train_stats.items():
                    writer.add_scalar(f"Train/{name}", value, current_epoch + 1)
                writer.add_scalar("Learning Rate", learning_rate, current_epoch + 1)
                print(
                    "Epoch: {}\tLearning Rate :{}\tAutoreg Ratio: {}\tTrain Loss Step: {}\tTrain Loss Full: {}".format(
                        current_epoch + 1,
                        learning_rate,
                        train_stats["autoreg_ratio"],
                        train_stats["loss_step"],
                        train_stats["loss_full"],
                    )
                )

            log_stats = {
                "epoch": current_epoch + 1,
                **{"train_{}".format(k): v for k, v in train_stats.items()},
            }
            write_epoch_log(log_file, log_stats)

            if (current_epoch + 1) % model_save_interval_epoch == 0:
                checker.save(current_epoch + 1)

    if basic_utils.is_main_process():
        writer.close()
        print("Finish Rollout Training !")


def train_time_embed(train_dataloader, normalizer, model, loss_fn, optimizer, scheduler, world_size, grad_clip, epoch, log_print_interval_epoch, model_save_interval_epoch, log_dir, checkpoint_dir):
    log_file = os.path.join(log_dir, "log.txt")
    step_log_file = os.path.join(log_dir, "log-step.txt")

    if basic_utils.is_main_process():
        writer = SummaryWriter(log_dir)
        checker = basic_utils.Checkpoint(checkpoint_dir, model)
        print("Number of Model Parameters: {}".format(basic_utils.get_num_params(model)))
        print(model)
        print("Start Time-Embedding Training...")

    for current_epoch in range(epoch):
        torch.distributed.barrier()
        train_dataloader.sampler.set_epoch(current_epoch)
        metric_logger = basic_utils.MetricLogger(delimiter="  ")
        header = "Epoch: [{}/{}]".format(current_epoch + 1, epoch)

        for data in metric_logger.log_every(train_dataloader, 10, header, step_log_file):
            coords, condition, sol, t = data
            coords = coords.cuda()
            coords = torch.reshape(coords, (coords.shape[0], -1, coords.shape[-1]))
            condition = condition.cuda()
            condition = torch.reshape(condition, (condition.shape[0], -1, condition.shape[-1]))
            sol = sol.cuda()
            sol = torch.reshape(sol, (sol.shape[0], -1, sol.shape[-2], sol.shape[-1]))
            t = t.cuda()

            model.train()
            loss_items_accum = {}
            last_loss_step = None

            for step_index in range(sol.shape[-2]):
                step_t = t[:, step_index]
                gt_step = sol[:, :, step_index, :]
                pred_step = model(coords, condition, step_t)

                if normalizer.is_apply_y2():
                    pred_step_loss = normalizer.apply_y2(pred_step, inverse=True)
                    gt_step_loss = normalizer.apply_y2(gt_step, inverse=True)
                else:
                    pred_step_loss = pred_step
                    gt_step_loss = gt_step

                step_loss_dict = loss_fn.compute_loss(pred_step_loss, gt_step_loss)
                train_loss_step = loss_fn.get_total_loss(step_loss_dict)

                optimizer.zero_grad()
                train_loss_step.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
                if scheduler.__class__.__name__ == "OneCycleLR":
                    scheduler.step()

                last_loss_step = train_loss_step.detach()
                for loss_name, loss_value in step_loss_dict.items():
                    detached_value = loss_value.detach()
                    loss_items_accum[loss_name] = loss_items_accum.get(loss_name, 0.0) + detached_value

            step_count = sol.shape[-2]
            averaged_step_loss_dict = {
                loss_name: loss_value / step_count for loss_name, loss_value in loss_items_accum.items()
            }
            metric_dict = basic_utils.build_train_metric_dict(
                {**averaged_step_loss_dict, "step": last_loss_step},
                last_loss_step,
                world_size,
            )
            update_kwargs = {**metric_dict, "lr": optimizer.param_groups[0]["lr"]}
            metric_logger.update(**update_kwargs)

        if scheduler.__class__.__name__ != "OneCycleLR":
            scheduler.step()

        metric_logger.synchronize_between_processes()
        train_stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
        learning_rate = optimizer.state_dict()["param_groups"][0]["lr"]

        if basic_utils.is_main_process():
            if (current_epoch + 1) % log_print_interval_epoch == 0:
                for name, value in train_stats.items():
                    writer.add_scalar(f"Train/{name}", value, current_epoch + 1)
                writer.add_scalar("Learning Rate", learning_rate, current_epoch + 1)
                print(
                    "Epoch: {}\tLearning Rate :{}\tTrain Loss Step: {}".format(
                        current_epoch + 1,
                        learning_rate,
                        train_stats["loss_step"],
                    )
                )

            log_stats = {
                "epoch": current_epoch + 1,
                **{"train_{}".format(k): v for k, v in train_stats.items()},
            }
            write_epoch_log(log_file, log_stats)

            if (current_epoch + 1) % model_save_interval_epoch == 0:
                checker.save(current_epoch + 1)

    if basic_utils.is_main_process():
        writer.close()
        print("Finish Time-Embedding Training !")


def main():
    # ==============================
    # initialization
    # ==============================
    # build args
    parser = build_parser()
    args = parser.parse_args()
    if args.config is not None:
        args = basic_utils.override_args_from_yaml(args, args.config)
    del args.config

    missing_parameters = [name for name in ("exp", "data_name") if getattr(args, name) is None]
    if missing_parameters:
        parser.error(f"Missing required parameters: {', '.join(missing_parameters)}")

    # build output_dir
    cur_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    args.output_dir = os.path.join(args.exp_root, f"{cur_time}_{args.exp}")
    
    # basic setup
    basic_utils.init_distributed_mode(args)
    basic_utils.set_seed(args.seed)

    world_size = args.world_size

    # build config
    config = build_config(args)

    # init output directories and save args
    log_dir = os.path.join(args.output_dir, "log")
    checkpoint_dir = os.path.join(args.output_dir, "checkpoint")
    if basic_utils.is_main_process():
        basic_utils.save_para(args, config, args.output_dir)
        os.makedirs(log_dir, exist_ok=True)

    # ============================
    # build model, dataloader, loss function, optimizer, and scheduler
    # ============================
    # build dataloader
    train_dataloader, _, normalizer, dim_dict = basic_utils.get_dataloaders(config)
    # vis random train samples
    if basic_utils.is_main_process() and args.n_train_vis > 0:
        train_dataset = train_dataloader.dataset
        n_vis = min(args.n_train_vis, len(train_dataset))
        vis_indices = torch.randperm(len(train_dataset))[:n_vis].tolist()
        vis_samples = [train_dataset[index] for index in vis_indices]
        train_vis_dir = os.path.join(args.output_dir, "train_vis")
        basic_utils.plot_samples(vis_samples, config.data.name, train_vis_dir)
    # build model
    model = basic_utils.build_model_from_args(
        {
            "model_name": config.model.name,
            "normed_first_stage": config.model.normed_first_stage,
            "out_droprate": config.model.out_droprate,
            "coords_dim": dim_dict["coords_dim"],
            "condition_dim": dim_dict["condition_dim"],
            "sol_dim": dim_dict["sol_dim"],
            "time_proj": config.model.time_proj,
        }
    ).cuda()
    if args.load_ckpt is not None:
        ckpt_load_report = load_model_checkpoint_for_init(
            model,
            args.load_ckpt,
            strict=args.strict_ckpt_load,
        )
        if args.freeze_loaded_params:
            freeze_loaded_parameters(model, ckpt_load_report["loaded_keys"])
    model = torch.nn.parallel.DistributedDataParallel(
        model,
        device_ids=[args.gpu]
    )
    # build optimizer and scheduler
    if config.scheduler.name == "OneCycle":
        sch_max_step = config.train.epoch * len(train_dataloader)
        if config.data.type == "time_embedding":
            sch_max_step *= dim_dict["total_steps"]
    elif config.scheduler.name == "Cos":
        sch_max_step = config.train.epoch  # step for each epoch
    else:
        sch_max_step = None
    optimizer, scheduler = basic_utils.get_optimizer(config, model, sch_max_step)
    # build loss function
    loss = LossManager(config.loss)

    # ============================
    # training and validation
    # ============================
    if config.data.type == "static":
        # for static datasets
        train(
            train_dataloader,
            normalizer,
            model,
            loss,
            optimizer,
            scheduler,
            world_size,
            config.train.grad_clip,
            config.train.epoch,
            config.train.log_print_interval_epoch,
            config.train.model_save_interval_epoch,
            log_dir,
            checkpoint_dir,
        )
    elif config.data.type == "airfrans":
        train_airfrans(
            train_dataloader,
            model,
            optimizer,
            scheduler,
            world_size,
            config.train.grad_clip,
            config.train.epoch,
            config.loss.airfrans_surface_weight,
            config.train.log_print_interval_epoch,
            config.train.model_save_interval_epoch,
            log_dir,
            checkpoint_dir,
        )
    elif config.data.type == "rollout":
        train_rollout(
            train_dataloader,
            normalizer,
            model,
            loss,
            optimizer,
            scheduler,
            world_size,
            config.train.grad_clip,
            config.train.epoch,
            config.data.rollout_history,
            config.data.coords_condition,
            config.train.rollout_autoreg_ratio,
            config.train.log_print_interval_epoch,
            config.train.model_save_interval_epoch,
            log_dir,
            checkpoint_dir,
        )
    elif config.data.type == "time_embedding":
        train_time_embed(
            train_dataloader,
            normalizer,
            model,
            loss,
            optimizer,
            scheduler,
            world_size,
            config.train.grad_clip,
            config.train.epoch,
            config.train.log_print_interval_epoch,
            config.train.model_save_interval_epoch,
            log_dir,
            checkpoint_dir,
        )

    # ============================
    # cleanup
    # ============================
    is_main_process = basic_utils.is_main_process()
    if is_main_process:
        print("Finish !")
        print("Output dir: {}".format(args.output_dir))
    if basic_utils.is_dist_avail_and_initialized():
        torch.distributed.destroy_process_group()

    # ==============================
    # final evaluation after training
    # ==============================
    if is_main_process:
        eval_main(
            exp_folder=args.output_dir,
            outputs_root=args.exp_root,
            checkpoint_epoch=None,
            device=f"cuda:{args.gpu}",
        )


if __name__ == "__main__":
    main()
