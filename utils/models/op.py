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

import math
import torch

try:
    import triton
    import triton.language as tl
except ImportError:
    triton = None
    tl = None


def sinkhorn_reference(logits, n_iter=8, temperature=1.0, inverse_update=False):
    """
    The fallback Sinkhorn implementation that is used when Triton is not available or the input tensor is not on CUDA.
    This implementation is slower than the Triton implementation but is more general and can run on CPU or GPU.
    """
    batch, n_point, n_mode = logits.shape
    scaled_logits = logits / temperature
    log_mu = scaled_logits.new_full((batch, n_point), -math.log(n_point))
    log_nu = scaled_logits.new_full((batch, n_mode), -math.log(n_mode))
    u = torch.zeros_like(log_mu)
    v = torch.zeros_like(log_nu)

    for _ in range(n_iter):
        if inverse_update:
            v = log_nu - torch.logsumexp(scaled_logits + u.unsqueeze(-1), dim=1)
            u = log_mu - torch.logsumexp(scaled_logits + v.unsqueeze(1), dim=-1)
        else:
            u = log_mu - torch.logsumexp(scaled_logits + v.unsqueeze(1), dim=-1)
            v = log_nu - torch.logsumexp(scaled_logits + u.unsqueeze(-1), dim=1)

    transport = torch.exp(scaled_logits + u.unsqueeze(-1) + v.unsqueeze(1))
    return transport * n_mode, transport * n_point


if triton is not None:

    @triton.jit
    def _sinkhorn_row_update_kernel(
        logits_ptr,
        v_ptr,
        u_ptr,
        n_point: tl.constexpr,
        n_mode: tl.constexpr,
        log_mu: tl.constexpr,
        inv_temperature: tl.constexpr,
        BLOCK_MODE: tl.constexpr,
    ):
        row_id = tl.program_id(0)
        batch_id = row_id // n_point
        point_id = row_id - batch_id * n_point
        mode_offsets = tl.arange(0, BLOCK_MODE)
        mask = mode_offsets < n_mode
        logits_offsets = (batch_id * n_point + point_id) * n_mode + mode_offsets
        v_offsets = batch_id * n_mode + mode_offsets
        values = tl.load(logits_ptr + logits_offsets, mask=mask, other=-float("inf")).to(tl.float32)
        values = values * inv_temperature
        values += tl.load(v_ptr + v_offsets, mask=mask, other=0.0).to(tl.float32)
        maximum = tl.max(values, axis=0)
        logsumexp = maximum + tl.log(tl.sum(tl.exp(values - maximum), axis=0))
        tl.store(u_ptr + row_id, log_mu - logsumexp)


    @triton.jit
    def _sinkhorn_column_update_kernel(
        logits_ptr,
        u_ptr,
        v_ptr,
        n_point: tl.constexpr,
        n_mode: tl.constexpr,
        log_nu: tl.constexpr,
        inv_temperature: tl.constexpr,
        BLOCK_POINT: tl.constexpr,
    ):
        column_id = tl.program_id(0)
        batch_id = column_id // n_mode
        mode_id = column_id - batch_id * n_mode
        point_offsets = tl.arange(0, BLOCK_POINT)
        mask = point_offsets < n_point
        logits_offsets = (batch_id * n_point + point_offsets) * n_mode + mode_id
        u_offsets = batch_id * n_point + point_offsets
        values = tl.load(logits_ptr + logits_offsets, mask=mask, other=-float("inf")).to(tl.float32)
        values = values * inv_temperature
        values += tl.load(u_ptr + u_offsets, mask=mask, other=0.0).to(tl.float32)
        maximum = tl.max(values, axis=0)
        logsumexp = maximum + tl.log(tl.sum(tl.exp(values - maximum), axis=0))
        tl.store(v_ptr + column_id, log_nu - logsumexp)


    @triton.jit
    def _sinkhorn_output_kernel(
        logits_ptr,
        u_ptr,
        v_ptr,
        score_encode_ptr,
        score_decode_ptr,
        n_element,
        n_point: tl.constexpr,
        n_mode: tl.constexpr,
        inv_temperature: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_element
        mode_id = offsets % n_mode
        row_id = offsets // n_mode
        batch_id = row_id // n_point
        logits = tl.load(logits_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        u = tl.load(u_ptr + row_id, mask=mask, other=0.0).to(tl.float32)
        v = tl.load(v_ptr + batch_id * n_mode + mode_id, mask=mask, other=0.0).to(tl.float32)
        log_transport = tl.where(mask, logits * inv_temperature + u + v, -float("inf"))
        log_transport = tl.minimum(log_transport, 0.0)
        transport = tl.exp(log_transport)
        tl.store(score_encode_ptr + offsets, transport * n_mode, mask=mask)
        tl.store(score_decode_ptr + offsets, transport * n_point, mask=mask)


    @triton.jit
    def _sinkhorn_output_row_backward_kernel(
        logits_ptr,
        u_ptr,
        v_ptr,
        grad_score_encode_ptr,
        grad_score_decode_ptr,
        grad_logits_ptr,
        grad_u_ptr,
        n_point: tl.constexpr,
        n_mode: tl.constexpr,
        inv_temperature: tl.constexpr,
        BLOCK_MODE: tl.constexpr,
    ):
        row_id = tl.program_id(0)
        batch_id = row_id // n_point
        mode_offsets = tl.arange(0, BLOCK_MODE)
        mask = mode_offsets < n_mode
        offsets = row_id * n_mode + mode_offsets
        logits = tl.load(logits_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        u = tl.load(u_ptr + row_id).to(tl.float32)
        v = tl.load(
            v_ptr + batch_id * n_mode + mode_offsets,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        grad_encode = tl.load(
            grad_score_encode_ptr + offsets,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        grad_decode = tl.load(
            grad_score_decode_ptr + offsets,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        log_transport = tl.where(mask, logits * inv_temperature + u + v, -float("inf"))
        log_transport = tl.minimum(log_transport, 0.0)
        transport = tl.exp(log_transport)
        grad_transport = grad_encode * n_mode + grad_decode * n_point
        grad_z = tl.where(mask, grad_transport * transport, 0.0)
        tl.store(grad_logits_ptr + offsets, grad_z * inv_temperature, mask=mask)
        tl.store(grad_u_ptr + row_id, tl.sum(grad_z, axis=0))


    @triton.jit
    def _sinkhorn_output_column_backward_kernel(
        logits_ptr,
        u_ptr,
        v_ptr,
        grad_score_encode_ptr,
        grad_score_decode_ptr,
        grad_v_ptr,
        n_point: tl.constexpr,
        n_mode: tl.constexpr,
        inv_temperature: tl.constexpr,
        BLOCK_POINT: tl.constexpr,
    ):
        column_id = tl.program_id(0)
        batch_id = column_id // n_mode
        mode_id = column_id - batch_id * n_mode
        point_offsets = tl.arange(0, BLOCK_POINT)
        mask = point_offsets < n_point
        offsets = (batch_id * n_point + point_offsets) * n_mode + mode_id
        logits = tl.load(logits_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        u = tl.load(
            u_ptr + batch_id * n_point + point_offsets,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        v = tl.load(v_ptr + column_id).to(tl.float32)
        grad_encode = tl.load(
            grad_score_encode_ptr + offsets,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        grad_decode = tl.load(
            grad_score_decode_ptr + offsets,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        log_transport = tl.where(mask, logits * inv_temperature + u + v, -float("inf"))
        log_transport = tl.minimum(log_transport, 0.0)
        transport = tl.exp(log_transport)
        grad_transport = grad_encode * n_mode + grad_decode * n_point
        grad_z = tl.where(mask, grad_transport * transport, 0.0)
        tl.store(grad_v_ptr + column_id, tl.sum(grad_z, axis=0))


    @triton.jit
    def _sinkhorn_column_update_backward_kernel(
        logits_ptr,
        u_ptr,
        v_ptr,
        grad_v_ptr,
        grad_u_base_ptr,
        grad_u_ptr,
        grad_logits_ptr,
        n_point: tl.constexpr,
        n_mode: tl.constexpr,
        log_nu: tl.constexpr,
        inv_temperature: tl.constexpr,
        HAS_U_BASE: tl.constexpr,
        BLOCK_MODE: tl.constexpr,
    ):
        row_id = tl.program_id(0)
        batch_id = row_id // n_point
        mode_offsets = tl.arange(0, BLOCK_MODE)
        mask = mode_offsets < n_mode
        offsets = row_id * n_mode + mode_offsets
        logits = tl.load(logits_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        u = tl.load(u_ptr + row_id).to(tl.float32)
        v = tl.load(
            v_ptr + batch_id * n_mode + mode_offsets,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        grad_v = tl.load(
            grad_v_ptr + batch_id * n_mode + mode_offsets,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        log_probability = tl.where(mask, logits * inv_temperature + u + v - log_nu, -float("inf"))
        log_probability = tl.minimum(log_probability, 0.0)
        probability = tl.exp(log_probability)
        contribution = tl.where(mask, -probability * grad_v, 0.0)
        previous = 0.0
        if HAS_U_BASE:
            previous = tl.load(grad_u_base_ptr + row_id).to(tl.float32)
        tl.store(grad_u_ptr + row_id, previous + tl.sum(contribution, axis=0))
        grad_logits = tl.load(grad_logits_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        tl.store(
            grad_logits_ptr + offsets,
            grad_logits + contribution * inv_temperature,
            mask=mask,
        )


    @triton.jit
    def _sinkhorn_row_update_backward_kernel(
        logits_ptr,
        u_ptr,
        v_previous_ptr,
        grad_u_ptr,
        grad_v_previous_ptr,
        grad_logits_ptr,
        n_point: tl.constexpr,
        n_mode: tl.constexpr,
        log_mu: tl.constexpr,
        inv_temperature: tl.constexpr,
        BLOCK_POINT: tl.constexpr,
    ):
        column_id = tl.program_id(0)
        batch_id = column_id // n_mode
        mode_id = column_id - batch_id * n_mode
        point_offsets = tl.arange(0, BLOCK_POINT)
        mask = point_offsets < n_point
        offsets = (batch_id * n_point + point_offsets) * n_mode + mode_id
        logits = tl.load(logits_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        u = tl.load(
            u_ptr + batch_id * n_point + point_offsets,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        v_previous = tl.load(v_previous_ptr + column_id).to(tl.float32)
        grad_u = tl.load(
            grad_u_ptr + batch_id * n_point + point_offsets,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        log_probability = tl.where(mask, logits * inv_temperature + v_previous + u - log_mu, -float("inf"))
        log_probability = tl.minimum(log_probability, 0.0)
        probability = tl.exp(log_probability)
        contribution = tl.where(mask, -probability * grad_u, 0.0)
        tl.store(grad_v_previous_ptr + column_id, tl.sum(contribution, axis=0))
        grad_logits = tl.load(grad_logits_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        tl.store(
            grad_logits_ptr + offsets,
            grad_logits + contribution * inv_temperature,
            mask=mask,
        )


def _num_warps(block_size):
    if block_size >= 4096:
        return 16
    if block_size >= 1024:
        return 8
    return 4


def _sinkhorn_triton_forward(logits, n_iter, temperature, inverse_update):
    if triton is None:
        raise RuntimeError("Triton is not installed.")
    if not logits.is_cuda:
        raise ValueError("The Triton Sinkhorn implementation requires a CUDA tensor.")
    if logits.ndim != 3:
        raise ValueError(f"Expected logits with shape [B, N, M], got {tuple(logits.shape)}.")
    if not logits.is_contiguous():
        logits = logits.contiguous()

    batch, n_point, n_mode = logits.shape
    block_point = triton.next_power_of_2(n_point)
    block_mode = triton.next_power_of_2(n_mode)
    u_history = torch.empty(
        (n_iter, batch, n_point),
        device=logits.device,
        dtype=torch.float32,
    )
    v_history = torch.empty(
        (n_iter + 1, batch, n_mode),
        device=logits.device,
        dtype=torch.float32,
    )
    inv_temperature = 1.0 / float(temperature)

    def update_rows(v, u):
        _sinkhorn_row_update_kernel[(batch * n_point,)](
            logits,
            v,
            u,
            n_point=n_point,
            n_mode=n_mode,
            log_mu=-math.log(n_point),
            inv_temperature=inv_temperature,
            BLOCK_MODE=block_mode,
            num_warps=_num_warps(block_mode),
        )

    def update_columns(u, v):
        _sinkhorn_column_update_kernel[(batch * n_mode,)](
            logits,
            u,
            v,
            n_point=n_point,
            n_mode=n_mode,
            log_nu=-math.log(n_mode),
            inv_temperature=inv_temperature,
            BLOCK_POINT=block_point,
            num_warps=_num_warps(block_point),
        )

    # The first update does not depend on the initial value of the opposite
    # dual because it is mathematically zero.
    if inverse_update:
        # inverse_update is uncommon and retained through the reference path
        # until its reverse recurrence kernels are added.
        return (*sinkhorn_reference(logits, n_iter, temperature, inverse_update), None, None)
    else:
        v_history[0].zero_()
        for iteration in range(n_iter):
            update_rows(v_history[iteration], u_history[iteration])
            update_columns(u_history[iteration], v_history[iteration + 1])

    score_encode = torch.empty_like(logits)
    score_decode = torch.empty_like(logits)
    n_element = logits.numel()
    grid = (triton.cdiv(n_element, 256),)
    _sinkhorn_output_kernel[grid](
        logits,
        u_history[-1],
        v_history[-1],
        score_encode,
        score_decode,
        n_element,
        n_point=n_point,
        n_mode=n_mode,
        inv_temperature=inv_temperature,
        BLOCK_SIZE=256,
        num_warps=4,
    )
    return score_encode, score_decode, u_history, v_history


def _sinkhorn_triton_backward(
    logits,
    u_history,
    v_history,
    grad_score_encode,
    grad_score_decode,
    temperature,
):
    """
    Reverse the finite Sinkhorn recurrence.

    Let L = logits / temperature. For the standard update order:

        u_t = log_mu - logsumexp_m(L + v_{t-1})
        v_t = log_nu - logsumexp_n(L + u_t)

    The required local Jacobians are the corresponding normalized rows and
    columns:

        p_row = exp(L + v_{t-1} + u_t - log_mu)
        p_col = exp(L + u_t + v_t - log_nu)

    Therefore:

        d(L + u_t) = -p_col * d(v_t)
        d(L + v_{t-1}) = -p_row * d(u_t)

    Iterating these two equations backward only requires the saved u/v
    histories, not the full [B, N, M] tensors from every Sinkhorn iteration.
    """
    batch, n_point, n_mode = logits.shape
    n_iter = u_history.shape[0]
    block_point = triton.next_power_of_2(n_point)
    block_mode = triton.next_power_of_2(n_mode)
    inv_temperature = 1.0 / float(temperature)
    grad_logits = torch.empty_like(logits)
    grad_u_output = torch.empty(
        (batch, n_point),
        device=logits.device,
        dtype=torch.float32,
    )
    grad_v = torch.empty(
        (batch, n_mode),
        device=logits.device,
        dtype=torch.float32,
    )

    _sinkhorn_output_row_backward_kernel[(batch * n_point,)](
        logits,
        u_history[-1],
        v_history[-1],
        grad_score_encode,
        grad_score_decode,
        grad_logits,
        grad_u_output,
        n_point=n_point,
        n_mode=n_mode,
        inv_temperature=inv_temperature,
        BLOCK_MODE=block_mode,
        num_warps=_num_warps(block_mode),
    )
    _sinkhorn_output_column_backward_kernel[(batch * n_mode,)](
        logits,
        u_history[-1],
        v_history[-1],
        grad_score_encode,
        grad_score_decode,
        grad_v,
        n_point=n_point,
        n_mode=n_mode,
        inv_temperature=inv_temperature,
        BLOCK_POINT=block_point,
        num_warps=_num_warps(block_point),
    )

    grad_u = torch.empty_like(grad_u_output)
    grad_v_previous = torch.empty_like(grad_v)
    for iteration in reversed(range(n_iter)):
        _sinkhorn_column_update_backward_kernel[(batch * n_point,)](
            logits,
            u_history[iteration],
            v_history[iteration + 1],
            grad_v,
            grad_u_output,
            grad_u,
            grad_logits,
            n_point=n_point,
            n_mode=n_mode,
            log_nu=-math.log(n_mode),
            inv_temperature=inv_temperature,
            HAS_U_BASE=iteration == n_iter - 1,
            BLOCK_MODE=block_mode,
            num_warps=_num_warps(block_mode),
        )
        _sinkhorn_row_update_backward_kernel[(batch * n_mode,)](
            logits,
            u_history[iteration],
            v_history[iteration],
            grad_u,
            grad_v_previous,
            grad_logits,
            n_point=n_point,
            n_mode=n_mode,
            log_mu=-math.log(n_point),
            inv_temperature=inv_temperature,
            BLOCK_POINT=block_point,
            num_warps=_num_warps(block_point),
        )
        grad_v, grad_v_previous = grad_v_previous, grad_v

    return grad_logits


class _FasterSinkhornFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits, n_iter, temperature, inverse_update):
        ctx.n_iter = int(n_iter)
        ctx.temperature = float(temperature)
        ctx.inverse_update = bool(inverse_update)
        score_encode, score_decode, u_history, v_history = _sinkhorn_triton_forward(
            logits,
            n_iter=ctx.n_iter,
            temperature=ctx.temperature,
            inverse_update=ctx.inverse_update,
        )
        if u_history is None:
            ctx.reference_backward = True
            ctx.save_for_backward(logits)
        else:
            ctx.reference_backward = False
            ctx.save_for_backward(logits, u_history, v_history)
        return score_encode, score_decode

    @staticmethod
    def backward(ctx, grad_score_encode, grad_score_decode):
        if not ctx.reference_backward:
            logits, u_history, v_history = ctx.saved_tensors
            if grad_score_encode is None:
                grad_score_encode = torch.zeros_like(logits)
            if grad_score_decode is None:
                grad_score_decode = torch.zeros_like(logits)
            return (
                _sinkhorn_triton_backward(
                    logits,
                    u_history,
                    v_history,
                    grad_score_encode.contiguous(),
                    grad_score_decode.contiguous(),
                    temperature=ctx.temperature,
                ),
                None,
                None,
                None,
            )

        (logits,) = ctx.saved_tensors
        create_graph = torch.is_grad_enabled()
        with torch.enable_grad():
            work_logits = logits.detach().requires_grad_(True)
            score_encode, score_decode = sinkhorn_reference(
                work_logits,
                n_iter=ctx.n_iter,
                temperature=ctx.temperature,
                inverse_update=ctx.inverse_update,
            )
            if grad_score_encode is None:
                grad_score_encode = torch.zeros_like(score_encode)
            if grad_score_decode is None:
                grad_score_decode = torch.zeros_like(score_decode)
            grad_logits = torch.autograd.grad(
                outputs=(score_encode, score_decode),
                inputs=work_logits,
                grad_outputs=(grad_score_encode, grad_score_decode),
                create_graph=create_graph,
            )[0]
        return grad_logits, None, None, None


def faster_sinkhorn(logits, n_iter=8, temperature=1.0, inverse_update=False):
    if n_iter <= 0:
        raise ValueError("n_iter must be positive.")
    if temperature <= 0:
        raise ValueError("temperature must be positive.")
    if (
        logits.is_cuda
        and logits.dtype == torch.float32
        and triton is not None
        and not inverse_update
    ):
        if not logits.is_contiguous():
            logits = logits.contiguous()
        return _FasterSinkhornFunction.apply(logits, n_iter, temperature, inverse_update)
    return sinkhorn_reference(
        logits,
        n_iter=n_iter,
        temperature=temperature,
        inverse_update=inverse_update,
    )


def has_triton():
    return triton is not None
