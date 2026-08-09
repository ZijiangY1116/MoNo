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

import copy
import math

import torch
import torch.nn.functional as F
from timm.layers import trunc_normal_

from .op import faster_sinkhorn


ACTIVATION = {
    "Sigmoid": torch.nn.Sigmoid(),
    "Tanh": torch.nn.Tanh(),
    "ReLU": torch.nn.ReLU(),
    "LeakyReLU": torch.nn.LeakyReLU(0.1),
    "ELU": torch.nn.ELU(),
    "GELU": torch.nn.GELU(),
}


def timestep_embedding(timesteps, dim, max_period=10000):
    """Build sinusoidal timestep embeddings."""
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period)
        * torch.arange(
            start=0,
            end=half,
            dtype=torch.float32,
            device=timesteps.device,
        )
        / half
    )
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat(
            [embedding, torch.zeros_like(embedding[:, :1])],
            dim=-1,
        )
    return embedding


class FasterSinkhornRouter(torch.nn.Module):
    """
    Balanced point-to-mode routing backed by the fused Sinkhorn operator.

    The encode mapping aggregates point features into modes, while the decode
    mapping broadcasts mode features back to the original points.
    """

    def __init__(self, n_iter=8, temperature=1.0, inverse_update=False):
        super().__init__()
        if n_iter <= 0:
            raise ValueError("n_iter must be positive.")
        if temperature <= 0:
            raise ValueError("temperature must be positive.")
        self.n_iter = n_iter
        self.temperature = temperature
        self.inverse_update = inverse_update

    def forward(self, logits, *args, **kwargs):
        score_encode, score_decode = faster_sinkhorn(
            logits,
            n_iter=self.n_iter,
            temperature=self.temperature,
            inverse_update=self.inverse_update,
        )
        return {
            "score_encode": score_encode,
            "score_decode": score_decode,
        }


class MONORowNormalizedLinear(torch.nn.Module):
    """
    Bias-free linear projection with row-normalized weights and one learnable
    global scale shared by all output rows.
    """

    def __init__(self, in_features, out_features, scale_init=3.0, eps=1e-12):
        super().__init__()
        if in_features <= 0:
            raise ValueError("in_features must be positive.")
        if out_features <= 0:
            raise ValueError("out_features must be positive.")
        if eps <= 0:
            raise ValueError("eps must be positive.")

        self.in_features = in_features
        self.out_features = out_features
        self.eps = eps
        self.weight = torch.nn.Parameter(
            torch.empty(out_features, in_features)
        )
        self.global_scale = torch.nn.Parameter(
            torch.tensor(float(scale_init))
        )
        self.reset_parameters()

    def reset_parameters(self):
        # Match nn.Linear's default weight initialization.
        torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, x):
        direction = F.normalize(
            self.weight,
            p=2,
            dim=1,
            eps=self.eps,
        )
        effective_weight = direction * self.global_scale
        return F.linear(x, effective_weight)


class MONOMLP(torch.nn.Module):
    """Residual MLP used by feature, routing, and output projectors."""

    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim,
        n_layer,
        act,
        row_normalized_output=False,
        row_norm_scale_init=3.0,
        drop_rate=0.0,
    ):
        super().__init__()
        self.input = torch.nn.Linear(input_dim, hidden_dim)
        self.hidden = torch.nn.ModuleList(
            [
                torch.nn.Linear(hidden_dim, hidden_dim)
                for _ in range(n_layer)
            ]
        )
        if row_normalized_output:
            self.output = MONORowNormalizedLinear(
                hidden_dim,
                output_dim,
                scale_init=row_norm_scale_init,
            )
        else:
            self.output = torch.nn.Linear(hidden_dim, output_dim)
        self.act = act
        self.drop = torch.nn.Dropout(drop_rate)

    def forward(self, x):
        hidden = self.act(self.input(x))
        for layer in self.hidden:
            hidden = hidden + self.drop(self.act(layer(hidden)))
        return self.output(hidden)


class TimeProjector(torch.nn.Module):
    """Project a scalar timestep and broadcast it to every input point."""

    def __init__(self, n_dim, n_layer, act):
        super().__init__()
        self.projector = MONOMLP(
            n_dim,
            n_dim,
            n_dim,
            n_layer,
            act,
        )
        self.n_dim = n_dim

    def forward(self, t, n_point):
        if t.dim() > 1:
            t = t.reshape(t.shape[0])
        time_feature = timestep_embedding(t, self.n_dim)
        time_feature = self.projector(time_feature)
        return time_feature.unsqueeze(1).expand(-1, n_point, -1)


class MONOSelfAttention(torch.nn.Module):
    """
    Explicit multi-head self-attention used by MONO.

    This is intentionally the non-fused attention path of the reference model.
    """

    def __init__(self, n_dim, n_head, drop_rate=0.0):
        super().__init__()
        self.n_head = n_head
        self.scale = (n_dim // n_head) ** -0.5
        self.fused_attn = False
        self.attn_drop = torch.nn.Dropout(drop_rate)
        self.Wq = torch.nn.Linear(n_dim, n_dim)
        self.Wk = torch.nn.Linear(n_dim, n_dim)
        self.Wv = torch.nn.Linear(n_dim, n_dim)
        self.proj = torch.nn.Linear(n_dim, n_dim)

    def forward(self, x):
        batch, tokens, dim = x.size()

        # [B, T, D] -> [B, H, T, D/H]
        q = self.Wq(x).view(
            batch,
            tokens,
            self.n_head,
            dim // self.n_head,
        ).permute(0, 2, 1, 3)
        k = self.Wk(x).view(
            batch,
            tokens,
            self.n_head,
            dim // self.n_head,
        ).permute(0, 2, 1, 3)
        v = self.Wv(x).view(
            batch,
            tokens,
            self.n_head,
            dim // self.n_head,
        ).permute(0, 2, 1, 3)

        # The explicit attention path is kept to match fused_attn=False.
        q = q * self.scale
        attn = q @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        out = attn @ v

        out = out.permute(0, 2, 1, 3).contiguous().view(
            batch,
            tokens,
            dim,
        )
        return self.proj(out)


class MONOAttentionBlock(torch.nn.Module):
    """Pre-normalized self-attention block followed by a residual MLP."""

    def __init__(
        self,
        n_dim,
        n_head,
        act,
        drop_rate=0.0,
        mlp_expand_rate=4.0,
    ):
        super().__init__()
        self.self_attn = MONOSelfAttention(
            n_dim,
            n_head,
            drop_rate=drop_rate,
        )
        self.ln1 = torch.nn.LayerNorm(n_dim)
        self.ln2 = torch.nn.LayerNorm(n_dim)
        self.drop = torch.nn.Dropout(drop_rate)
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(n_dim, int(n_dim * mlp_expand_rate)),
            act,
            torch.nn.Linear(int(n_dim * mlp_expand_rate), n_dim),
        )

    def forward(self, y):
        y = y + self.drop(self.self_attn(self.ln1(y)))
        y = y + self.mlp(self.ln2(y))
        return y


class MONOLatentStack(torch.nn.Module):
    """Stack multiple attention blocks in the mode/latent space."""

    def __init__(
        self,
        n_block,
        n_dim,
        n_head,
        act,
        drop_rate=0.0,
    ):
        super().__init__()
        self.blocks = torch.nn.Sequential(
            *[
                MONOAttentionBlock(
                    n_dim,
                    n_head,
                    act,
                    drop_rate=drop_rate,
                    mlp_expand_rate=4.0,
                )
                for _ in range(n_block)
            ]
        )

    def forward(self, latent):
        return self.blocks(latent)


class MONOInitialProjectorEMA(torch.nn.Module):
    """
    Initial point-space projector with an EMA trunk teacher.

    During training, a fixed fraction of coordinates is projected by the
    trainable trunk and the remainder by the frozen EMA copy. Evaluation uses
    only the EMA trunk. The condition branch is always trainable.
    """

    def __init__(
        self,
        n_dim,
        n_layer,
        coords_dim,
        condition_dim,
        act,
    ):
        super().__init__()
        self.alpha = 0.25
        self.ema_decay = 0.0

        self.trunk_projector_training = MONOMLP(
            coords_dim,
            n_dim,
            n_dim,
            n_layer,
            act,
        )
        self.trunk_projector_forward = copy.deepcopy(
            self.trunk_projector_training
        )
        self.branch_projector = MONOMLP(
            condition_dim,
            n_dim,
            n_dim,
            n_layer,
            act,
        )
        self._freeze_ema_projector()

    def _freeze_ema_projector(self):
        # The forward/teacher projector is updated only through EMA copies.
        for parameter in self.trunk_projector_forward.parameters():
            parameter.requires_grad_(False)

    @torch.no_grad()
    def sync_ema_projector(self):
        for ema_parameter, student_parameter in zip(
            self.trunk_projector_forward.parameters(),
            self.trunk_projector_training.parameters(),
        ):
            ema_parameter.data.copy_(student_parameter.data)

    @torch.no_grad()
    def _update_ema_projector(self):
        for ema_parameter, student_parameter in zip(
            self.trunk_projector_forward.parameters(),
            self.trunk_projector_training.parameters(),
        ):
            ema_parameter.data.mul_(self.ema_decay).add_(
                student_parameter.data,
                alpha=1.0 - self.ema_decay,
            )

    def _build_mixed_feature(self, x):
        if not self.training:
            with torch.no_grad():
                return self.trunk_projector_forward(x)

        # ema_decay=0 keeps the teacher synchronized with the current student
        # before the sampled student/teacher point split is evaluated.
        self._update_ema_projector()

        batch = x.shape[0]
        n_point = x.shape[1]
        hidden_dim = self.trunk_projector_training.output.out_features
        sample_size = max(1, int(math.ceil(n_point * self.alpha)))
        permuted_idx = torch.randperm(n_point, device=x.device)
        sampled_idx = permuted_idx[:sample_size].sort().values
        teacher_idx = permuted_idx[sample_size:].sort().values

        # Project the sampled subset with gradients.
        sampled_x = x.index_select(dim=1, index=sampled_idx)
        student_feature = self.trunk_projector_training(sampled_x)
        feature = x.new_zeros(batch, n_point, hidden_dim)

        # Project all remaining points through the frozen teacher.
        if teacher_idx.numel() > 0:
            teacher_x = x.index_select(dim=1, index=teacher_idx)
            with torch.no_grad():
                teacher_feature = self.trunk_projector_forward(teacher_x)
            teacher_scatter_index = teacher_idx.view(
                1,
                teacher_idx.numel(),
                1,
            ).expand(batch, teacher_idx.numel(), hidden_dim)
            feature = feature.scatter(
                dim=1,
                index=teacher_scatter_index,
                src=teacher_feature,
            )

        # Scatter both subsets back to the original point order.
        student_scatter_index = sampled_idx.view(
            1,
            sample_size,
            1,
        ).expand(batch, sample_size, hidden_dim)
        return feature.scatter(
            dim=1,
            index=student_scatter_index,
            src=student_feature,
        )

    def forward(self, x, y):
        x = self._build_mixed_feature(x)
        y = self.branch_projector(y)
        return x, y


class CoTAP(torch.nn.Module):
    """
    Build coordinate/anchor-to-mode routing and adjacent-level mappings.

    The first stage uses an MLP with a row-normalized output projection;
    subsequent stages use the same MLP with a standard linear output. All
    stages route with the fixed faster-Sinkhorn path.
    """

    def __init__(
        self,
        n_dim,
        n_mode,
        n_layer,
        act,
        mlp_rownorm=False,
    ):
        super().__init__()
        self.identity_mapping = False
        self.attention_projector = MONOMLP(
            n_dim,
            n_dim,
            n_mode,
            n_layer,
            act,
            row_normalized_output=mlp_rownorm,
            row_norm_scale_init=3.0,
        )
        self.router_type = "fastersinkhorn"
        self.router = FasterSinkhornRouter(
            n_iter=8,
            temperature=1.0,
            inverse_update=False,
        )

    def build_mapping(self, score_source, *args, **kwargs):
        # score: [B, N_source, N_mode]
        score = self.attention_projector(score_source)
        router_outputs = self.router(score, *args, **kwargs)
        return {"score": score, **router_outputs}

    def build_encode_mapping(self, mapping, *args, **kwargs):
        return mapping["score_encode"]

    def build_decode_mapping(self, mapping, *args, **kwargs):
        return mapping["score_decode"]

    def encode(self, score_encode, source_feature):
        # [B, N_source, N_mode] x [B, N_source, D]
        # -> [B, N_mode, D]
        return torch.einsum(
            "bij,bic->bjc",
            score_encode,
            source_feature,
        )

    def decode(self, score_decode, latent_feature):
        # [B, N_source, N_mode] x [B, N_mode, D]
        # -> [B, N_source, D]
        return torch.einsum(
            "bij,bjc->bic",
            score_decode,
            latent_feature,
        )


class MONOMultiLevelStage(torch.nn.Module):
    """
    One level of the MONO hierarchy.

    A stage owns its CoTAP mapping, encoder and decoder latent stacks, and the
    additive skip fusion used during top-down decoding.
    """

    def __init__(
        self,
        name,
        n_mode,
        n_dim,
        n_head,
        n_layer,
        n_block,
        act,
        mlp_rownorm,
        drop_rate,
    ):
        super().__init__()
        self.name = name
        self.n_mode = n_mode
        self.projector = CoTAP(
            n_dim,
            n_mode,
            n_layer,
            act,
            mlp_rownorm=mlp_rownorm,
        )
        self.encoder_stack = MONOLatentStack(
            n_block,
            n_dim,
            n_head,
            act,
            drop_rate=drop_rate,
        )
        self.decoder_stack = MONOLatentStack(
            n_block,
            n_dim,
            n_head,
            act,
            drop_rate=drop_rate,
        )
        self.fusion_fn = lambda x, skip: x + skip

    def build_mapping(self, score_source, *args, **kwargs):
        return self.projector.build_mapping(score_source, *args, **kwargs)

    def build_encode_mapping(self, mapping, *args, **kwargs):
        return self.projector.build_encode_mapping(mapping, *args, **kwargs)

    def build_decode_mapping(self, mapping, *args, **kwargs):
        return self.projector.build_decode_mapping(mapping, *args, **kwargs)

    def encode(self, score_encode, source_feature):
        return self.projector.encode(score_encode, source_feature)

    def decode(self, score_decode, latent_feature):
        return self.projector.decode(score_decode, latent_feature)

    def forward_encoder(self, latent):
        return self.encoder_stack(latent)

    def forward_fusion(self, latent, skip_feature):
        return self.fusion_fn(latent, skip_feature)

    def forward_decoder(self, latent):
        return self.decoder_stack(latent)


class MONO(torch.nn.Module):
    """
    Multiscale Optimal Transport Neural Operator
    """

    def __init__(
        self,
        n_mode,
        n_dim,
        n_layer,
        coords_dim,
        condition_dim,
        sol_dim,
        act,
        time_proj=False,
        attn_drop_rate=0.0,
        out_droprate=0.0,
        normed_first_stage=True,
    ):
        super().__init__()
        if n_mode <= 0:
            raise ValueError(f"n_mode must be positive, got {n_mode}.")

        activation = ACTIVATION[act]

        # ===================
        # build the initial point-space projector
        # ===================
        self.initial_projector = MONOInitialProjectorEMA(
            n_dim,
            n_layer,
            coords_dim,
            condition_dim,
            activation,
        )
        self.time_proj = time_proj
        if self.time_proj:
            self.time_projector = TimeProjector(
                n_dim,
                n_layer,
                activation,
            )

        self.anchor_router = True

        # ===================
        # build the fixed four-level latent hierarchy
        # ===================
        self.stages = torch.nn.ModuleList()
        n_blocks = (3, 1, 1, 1)
        n_heads = (4, 4, 8, 8)
        for stage_idx in range(4):
            self.stages.append(
                MONOMultiLevelStage(
                    name=f"stage_{stage_idx}",
                    n_mode=max(1, n_mode // (2 ** stage_idx)),
                    n_dim=n_dim,
                    n_head=n_heads[stage_idx],
                    n_layer=n_layer,
                    n_block=n_blocks[stage_idx],
                    act=activation,
                    # Optionally apply row normalization to the first stage.
                    mlp_rownorm=normed_first_stage and stage_idx == 0,
                    drop_rate=attn_drop_rate,
                )
            )

        # ===================
        # build the point-space output MLP
        # ===================
        self.out_mlp = MONOMLP(
            n_dim,
            n_dim,
            sol_dim,
            n_layer,
            activation,
            drop_rate=out_droprate,
        )
        self.relative_decode_residual = None

        # Keep this initialization sequence aligned with the reference model.
        self.initialize_weights()
        self._sync_ema_projectors()

    def initialize_weights(self):
        """
        Initialize normalization, output/time MLPs, and attention projections.

        Routing and initial projectors retain their construction-time PyTorch
        initialization, matching the reference model's initialization order.
        """
        self.apply(self._init_layer_norm)
        self._init_non_projector_mlps()
        self._init_attention_weights()

    def _init_layer_norm(self, module):
        if isinstance(module, torch.nn.LayerNorm):
            torch.nn.init.constant_(module.weight, 1.0)
            torch.nn.init.constant_(module.bias, 0)

    def _init_attention_linear(self, module):
        trunc_normal_(module.weight, std=0.02)
        if module.bias is not None:
            torch.nn.init.constant_(module.bias, 0)

    def _init_non_projector_mlps(self):
        # The output and optional time MLP use truncated-normal linear weights.
        for module in self.out_mlp.modules():
            if isinstance(module, torch.nn.Linear):
                self._init_attention_linear(module)
        if self.time_proj:
            for module in self.time_projector.modules():
                if isinstance(module, torch.nn.Linear):
                    self._init_attention_linear(module)

    def _init_attention_weights(self):
        # Initialize encoder stacks before decoder stacks, stage by stage.
        for stage in self.stages:
            for stack in (stage.encoder_stack, stage.decoder_stack):
                for module in stack.modules():
                    if isinstance(module, torch.nn.Linear):
                        self._init_attention_linear(module)

    @torch.no_grad()
    def _sync_ema_projectors(self):
        # Synchronize after all initialization so student and teacher match.
        self.initial_projector.sync_ema_projector()

    def forward(self, x, y, t=None):
        """
        Run point-to-mode encoding followed by mode-to-point decoding.

        Args:
            x: Point coordinates/anchors with shape ``[B, N, C_coord]``.
            y: Point conditions/features with shape ``[B, N, C_condition]``.
            t: Optional batch timesteps.

        Returns:
            The prediction tensor with shape ``[B, N, C_solution]``.
        """
        # Embed coordinates as routing anchors and conditions as features.
        point_anchor, point_feature = self.initial_projector(x, y)
        if t is not None and self.time_proj:
            point_feature = point_feature + self.time_projector(
                t,
                point_feature.shape[1],
            )

        source_feature = point_feature
        source_anchor = point_anchor
        encoder_features = []
        stage_mappings = {}

        # Build the hierarchy from point space to progressively fewer modes.
        for stage in self.stages:
            # Routing is computed from the anchor path, independently of the
            # feature path transformed by the latent attention stack.
            stage_mapping = stage.build_mapping(source_anchor)
            stage_score_encode = stage.build_encode_mapping(stage_mapping)
            latent_feature = stage.encode(
                stage_score_encode,
                source_feature,
            )
            latent_feature = stage.forward_encoder(latent_feature)
            encoder_features.append(latent_feature)
            stage_mappings[stage.name] = stage_mapping

            # Anchor routing: propagate encoded anchors rather than replacing
            # them with the encoded feature representation.
            source_anchor = stage.encode(
                stage_score_encode,
                source_anchor,
            )
            source_feature = latent_feature

        # Decode the hierarchy back to the original point resolution.
        result = encoder_features[-1]
        for idx in reversed(range(len(self.stages))):
            stage = self.stages[idx]
            if idx < len(self.stages) - 1:
                # Fuse the top-down result with the same-level encoder skip.
                result = stage.forward_fusion(
                    result,
                    encoder_features[idx],
                )
            result = stage.forward_decoder(result)
            stage_mapping = stage_mappings[stage.name]
            stage_score_decode = stage.build_decode_mapping(
                stage_mapping,
                mode_features=result,
            )
            # Each decode returns to the source resolution of this stage.
            result = stage.decode(stage_score_decode, result)

        # Map the recovered point features to the requested solution channels.
        return self.out_mlp(result)


def _build_mono(
    n_mode,
    n_dim,
    *,
    fixed_normed_first_stage=None,
    fixed_out_droprate=None,
):
    """Create a deterministic MONO profile builder."""

    def builder(
        *,
        coords_dim,
        condition_dim,
        sol_dim,
        time_proj=False,
        normed_first_stage=True,
        out_droprate=0.0,
    ):
        if fixed_normed_first_stage is not None:
            normed_first_stage = fixed_normed_first_stage
        if fixed_out_droprate is not None:
            out_droprate = fixed_out_droprate
        return MONO(
            n_mode=n_mode,
            n_dim=n_dim,
            n_layer=3,
            coords_dim=coords_dim,
            condition_dim=condition_dim,
            sol_dim=sol_dim,
            time_proj=time_proj,
            act="GELU",
            attn_drop_rate=0.0,
            out_droprate=out_droprate,
            normed_first_stage=normed_first_stage,
        )

    return builder


get_mono_light = _build_mono(n_mode=512, n_dim=96)
get_mono = _build_mono(n_mode=1024, n_dim=192)
