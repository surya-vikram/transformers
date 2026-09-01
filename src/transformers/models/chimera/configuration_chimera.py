# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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
"""Chimera model configuration."""

from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig
from ...modeling_rope_utils import RopeParameters


CHIMERA_CONTEXT_PHASES = {
    "8k": {"max_position_embeddings": 8192, "factor": 1.0},
    "32k": {"max_position_embeddings": 32768, "factor": 4.0},
    "64k": {"max_position_embeddings": 65536, "factor": 8.0},
    "128k": {"max_position_embeddings": 131072, "factor": 16.0},
}
CHIMERA_YARN_ORIGINAL_MAX_POSITION_EMBEDDINGS = 8192


@strict
class ChimeraConfig(PreTrainedConfig):
    r"""
    Configuration for Chimera, a decoder-only sparse MoE language model.
    """

    model_type = "chimera"
    keys_to_ignore_at_inference = ["past_key_values"]

    attribute_map = {
        "num_local_experts": "n_routed_experts",
    }

    vocab_size: int = 50176
    hidden_size: int = 2048
    intermediate_size: int = 8192
    moe_intermediate_size: int = 2048
    shared_expert_intermediate_size: int = 0
    num_hidden_layers: int = 25
    num_attention_heads: int = 16
    num_key_value_heads: int | None = 2
    head_dim: int = 256
    hidden_act: str = "silu"
    context_phase: str | None = None
    position_embedding_type: str = "yarn"
    max_position_embeddings: int = 8192
    original_max_position_embeddings: int = 8192
    initializer_range: float = 0.02
    rms_norm_eps: float = 1e-5
    use_cache: bool = True
    pad_token_id: int | None = 1
    bos_token_id: int | None = 0
    eos_token_id: int | list[int] | None = 1
    tie_word_embeddings: bool = False
    rope_parameters: RopeParameters | dict | None = None
    rope_theta: float = 10000000.0
    rope_scaling: dict | None = None
    attention_bias: bool = False
    attention_dropout: float | int | None = 0.0
    mlp_bias: bool = False
    qk_layernorm: bool = True
    load_with_bias: bool = True

    first_k_dense_replace: int = 2
    last_k_dense_replace: int = 0
    n_routed_experts: int = 32
    num_experts_per_tok: int = 4
    n_shared_experts: int = 0
    n_group: int = 1
    topk_group: int = 1
    norm_topk_prob: bool = True
    scoring_func: str = "sigmoid"
    topk_method: str = "noaux_tc"
    routed_scaling_factor: float = 2.5
    router_aux_loss_coef: float = 0.0
    router_z_loss_coef: float = 0.001
    router_bias_update_rate: float = 0.0
    router_load_balancing_type: str = "quantile_balancing"
    moe_qb_num_bins: int = 1000
    moe_qb_ema_decay: float = 0.0
    output_router_logits: bool = False

    def __post_init__(self, **kwargs):
        if self.num_key_value_heads is None:
            self.num_key_value_heads = self.num_attention_heads

        self._validate_and_normalize_yarn()

        if self.num_attention_heads % self.num_key_value_heads != 0:
            raise ValueError("num_attention_heads must be divisible by num_key_value_heads.")
        if self.n_routed_experts % self.n_group != 0:
            raise ValueError("n_routed_experts must be divisible by n_group.")
        if self.num_experts_per_tok > self.n_routed_experts:
            raise ValueError("num_experts_per_tok cannot exceed n_routed_experts.")
        if self.first_k_dense_replace + self.last_k_dense_replace > self.num_hidden_layers:
            raise ValueError("Dense replacement layers cannot exceed num_hidden_layers.")
        if self.scoring_func != "sigmoid":
            raise ValueError("Chimera currently supports only sigmoid MoE scoring.")
        if self.topk_method != "noaux_tc":
            raise ValueError("Chimera currently supports only noaux_tc top-k routing.")
        if self.n_shared_experts == 0 and self.shared_expert_intermediate_size != 0:
            raise ValueError("shared_expert_intermediate_size must be 0 when n_shared_experts is 0.")
        if self.n_shared_experts > 0 and self.shared_expert_intermediate_size <= 0:
            raise ValueError("shared_expert_intermediate_size must be positive when shared experts are enabled.")
        if self.router_load_balancing_type not in {"quantile_balancing", "none"}:
            raise ValueError("router_load_balancing_type must be 'quantile_balancing' or 'none'.")
        if self.moe_qb_num_bins <= 0:
            raise ValueError("moe_qb_num_bins must be positive.")
        if not 0.0 <= self.moe_qb_ema_decay < 1.0:
            raise ValueError("moe_qb_ema_decay must be greater than or equal to 0 and less than 1.")

        super().__post_init__(**kwargs)

    def _validate_and_normalize_yarn(self) -> None:
        if self.position_embedding_type != "yarn":
            raise ValueError("Chimera supports only position_embedding_type='yarn'.")
        if self.original_max_position_embeddings != CHIMERA_YARN_ORIGINAL_MAX_POSITION_EMBEDDINGS:
            raise ValueError(
                "Chimera requires original_max_position_embeddings=8192 across every context phase."
            )
        if self.rope_theta != 10_000_000.0:
            raise ValueError("Chimera requires rope_theta=10000000.0.")
        if self.rms_norm_eps != 1e-5:
            raise ValueError("Chimera requires rms_norm_eps=1e-5.")

        matching_phase = next(
            (
                phase
                for phase, geometry in CHIMERA_CONTEXT_PHASES.items()
                if self.max_position_embeddings == geometry["max_position_embeddings"]
            ),
            None,
        )
        if matching_phase is None:
            raise ValueError(
                "Chimera max_position_embeddings must be one of "
                f"{[geometry['max_position_embeddings'] for geometry in CHIMERA_CONTEXT_PHASES.values()]}."
            )
        if self.context_phase is not None and self.context_phase != matching_phase:
            raise ValueError(
                f"context_phase={self.context_phase!r} does not match "
                f"max_position_embeddings={self.max_position_embeddings}."
            )
        self.context_phase = matching_phase

        supplied_parameters = []
        for source in (self.rope_scaling, self.rope_parameters):
            if source is None:
                continue
            normalized = dict(source)
            rope_type = normalized.pop("type", normalized.get("rope_type", "yarn"))
            normalized["rope_type"] = rope_type
            supplied_parameters.append(normalized)

        if len(supplied_parameters) == 2:
            first, second = supplied_parameters
            for key in first.keys() & second.keys():
                if first[key] != second[key]:
                    raise ValueError(
                        f"rope_scaling and rope_parameters disagree for {key!r}: "
                        f"{first[key]!r} != {second[key]!r}."
                    )

        yarn_parameters = supplied_parameters[-1] if supplied_parameters else {}
        expected = {
            "rope_type": "yarn",
            "rope_theta": self.rope_theta,
            "factor": CHIMERA_CONTEXT_PHASES[matching_phase]["factor"],
            "beta_fast": 32.0,
            "beta_slow": 1.0,
            "mscale": 1.0,
            "mscale_all_dim": 0.0,
            "original_max_position_embeddings": CHIMERA_YARN_ORIGINAL_MAX_POSITION_EMBEDDINGS,
            # Megatron uses yarn_correction_range_round_to_int=False.
            "truncate": False,
        }
        for key, value in expected.items():
            actual = yarn_parameters.setdefault(key, value)
            if actual != value:
                raise ValueError(f"Chimera YaRN requires {key}={value!r}, found {actual!r}.")

        self.rope_parameters = yarn_parameters
        self.rope_scaling = {
            "type": "yarn",
            **{key: value for key, value in yarn_parameters.items() if key not in {"rope_type", "rope_theta"}},
        }


__all__ = ["ChimeraConfig"]
