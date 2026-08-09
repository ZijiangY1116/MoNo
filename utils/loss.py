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

import torch


def infer_square_resolution(num_points):
    resolution = int(round(num_points ** 0.5))
    if resolution * resolution != num_points:
        raise ValueError(
            f"DarcyDerivLoss requires a square grid, but got {num_points} points."
        )
    return resolution


def central_diff_2d(field, step_size, resolution):
    field = field.reshape(field.shape[0], resolution, resolution, field.shape[-1])
    field = torch.nn.functional.pad(field, (0, 0, 1, 1, 1, 1), mode="constant", value=0.0)
    grad_x = (field[:, 1:-1, 2:, :] - field[:, 1:-1, :-2, :]) / (2 * step_size)
    grad_y = (field[:, 2:, 1:-1, :] - field[:, :-2, 1:-1, :]) / (2 * step_size)
    return grad_x, grad_y


class RelLpLoss(torch.nn.modules.loss._Loss):
    def __init__(self, p):
        super().__init__()
        self.p = p

    def forward(self, pred, target):
        error = torch.sum(abs(pred - target) ** self.p, tuple(range(1, len(pred.shape)))) ** (1 / self.p)
        target = torch.sum(abs(target) ** self.p, tuple(range(1, len(pred.shape)))) ** (1 / self.p)
        return torch.mean(error / target)


class LpLoss(torch.nn.modules.loss._Loss):
    def __init__(self, p):
        super().__init__()
        self.p = p

    def forward(self, pred, target):
        error = torch.mean(abs(pred - target) ** self.p, tuple(range(1, len(pred.shape)))) ** (1 / self.p)
        return torch.mean(error)
    

class DarcyDerivLoss(torch.nn.modules.loss._Loss):
    def __init__(self, p=2):
        super().__init__()
        self.base_loss = RelLpLoss(p=p)

    def regularize_pred_boundary(self, pred, resolution):
        pred_grid = pred.reshape(pred.shape[0], resolution, resolution, pred.shape[-1])
        pred_grid = pred_grid[:, 1:-1, 1:-1, :].contiguous()
        pred_grid = pred_grid.permute(0, 3, 1, 2)
        pred_grid = torch.nn.functional.pad(pred_grid, (1, 1, 1, 1), mode="constant", value=0.0)
        pred_grid = pred_grid.permute(0, 2, 3, 1)
        return pred_grid.reshape(pred.shape[0], resolution * resolution, pred.shape[-1])

    def forward(self, pred, target):
        if pred.ndim == 2:
            pred = pred.unsqueeze(-1)
        if target.ndim == 2:
            target = target.unsqueeze(-1)

        if pred.shape != target.shape:
            raise ValueError(
                f"DarcyDerivLoss requires pred/target to have the same shape, got {pred.shape} and {target.shape}."
            )

        resolution = infer_square_resolution(pred.shape[1])
        step_size = 1.0 / resolution

        pred = self.regularize_pred_boundary(pred, resolution)
        target_grad_x, target_grad_y = central_diff_2d(target, step_size, resolution)
        pred_grad_x, pred_grad_y = central_diff_2d(pred, step_size, resolution)

        return self.base_loss(pred_grad_x, target_grad_x) + self.base_loss(pred_grad_y, target_grad_y)


class LossManager:
    
    def __init__(self, config):
        loss_names = config["name"].split("+")
        weight_config = config.get("weight")
        if weight_config is None:
            loss_weights = [1.0] * len(loss_names)
        else:
            loss_weights = [float(weight) for weight in weight_config.split("+")]
            assert len(loss_names) == len(loss_weights), "The number of loss functions and weights must match."

        self.loss_fns = {
            loss_name: self.build_loss_fn(loss_name)
            for loss_name in loss_names
        }
        self.loss_weights = {
            loss_name: loss_weight
            for loss_name, loss_weight in zip(loss_names, loss_weights)
        }
    
    def build_loss_fn(self, name):
        if name == 'L2':
            return LpLoss(p=2).cuda()
        elif name == 'L1':
            return LpLoss(p=1).cuda()
        elif name == 'rL2':
            return RelLpLoss(p=2).cuda()
        elif name == 'rL1':
            return RelLpLoss(p=1).cuda()
        elif name == 'DarcyDeriv':
            return DarcyDerivLoss(p=2).cuda()
        else:
            raise NotImplementedError(f"Loss function {name} is not implemented.")
    
    def compute_loss(self, pred, target):
        loss_dict = {}
        for loss_name, loss_fn in self.loss_fns.items():
            loss_dict[loss_name] = loss_fn(pred, target)
        return loss_dict

    def get_total_loss(self, loss_dict):
        total_loss = None
        for loss_name, loss_value in loss_dict.items():
            weighted_loss = self.loss_weights[loss_name] * loss_value
            total_loss = weighted_loss if total_loss is None else total_loss + weighted_loss

        if total_loss is None:
            raise ValueError("LossManager.get_total_loss received an empty loss_dict.")

        return total_loss
