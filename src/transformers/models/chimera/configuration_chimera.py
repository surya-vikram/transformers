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
    moe_intermediate_size: int = 1024
    shared_expert_intermediate_size: int = 1024
    num_hidden_layers: int = 25
    num_attention_heads: int = 16
    num_key_value_heads: int | None = 2
    head_dim: int = 256
    hidden_act: str = "silu"
    max_position_embeddings: int = 32768
    original_max_position_embeddings: int = 8192
    initializer_range: float = 0.02
    rms_norm_eps: float = 1e-6
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
    qk_layernorm: bool = False

    first_k_dense_replace: int = 2
    last_k_dense_replace: int = 0
    n_routed_experts: int = 64
    num_experts_per_tok: int = 4
    n_shared_experts: int = 1
    n_group: int = 1
    topk_group: int = 1
    norm_topk_prob: bool = True
    scoring_func: str = "sigmoid"
    topk_method: str = "noaux_tc"
    routed_scaling_factor: float = 2.5
    router_aux_loss_coef: float = 0.0001
    router_bias_update_rate: float = 0.001
    output_router_logits: bool = False

    def __post_init__(self, **kwargs):
        if self.num_key_value_heads is None:
            self.num_key_value_heads = self.num_attention_heads

        if self.rope_parameters is None:
            rope_scaling = self.rope_scaling or {
                "type": "yarn",
                "factor": 4.0,
                "original_max_position_embeddings": self.original_max_position_embeddings,
            }
            rope_type = rope_scaling.get("rope_type", rope_scaling.get("type", "default"))
            self.rope_parameters = {
                "rope_type": rope_type,
                "rope_theta": self.rope_theta,
                **{k: v for k, v in rope_scaling.items() if k != "type"},
            }

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

        super().__post_init__(**kwargs)


__all__ = ["ChimeraConfig"]
