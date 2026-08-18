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

import unittest
from copy import deepcopy

import torch

from transformers import ChimeraConfig
from transformers.testing_utils import require_torch
from transformers.utils import is_torch_available


if is_torch_available():
    from transformers.models.chimera.modeling_chimera import ChimeraAttention, ChimeraExperts, ChimeraSparseMoeBlock


@require_torch
class ChimeraExpertsTest(unittest.TestCase):
    def setUp(self):
        self.config = ChimeraConfig(
            hidden_size=8,
            intermediate_size=16,
            moe_intermediate_size=12,
            shared_expert_intermediate_size=12,
            num_hidden_layers=3,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=4,
            first_k_dense_replace=2,
            n_routed_experts=4,
            num_experts_per_tok=2,
            n_shared_experts=1,
        )

    def test_experts_vllm_forward_contract(self):
        torch.manual_seed(0)
        experts = ChimeraExperts(self.config)
        hidden_states = torch.randn(6, self.config.hidden_size)
        top_k_index = torch.tensor([[0, 1], [2, 3], [1, 2], [3, 0], [0, 2], [1, 3]])
        top_k_weights = torch.rand(6, self.config.num_experts_per_tok)

        expected = torch.zeros_like(hidden_states)
        for token_idx in range(hidden_states.shape[0]):
            for topk_pos in range(self.config.num_experts_per_tok):
                expert_idx = top_k_index[token_idx, topk_pos]
                expected[token_idx] += (
                    experts[expert_idx](hidden_states[token_idx : token_idx + 1]).squeeze(0)
                    * top_k_weights[token_idx, topk_pos]
                )

        actual = experts(hidden_states, top_k_index, top_k_weights)
        torch.testing.assert_close(actual, expected)

    def test_locked_router_defaults(self):
        config = ChimeraConfig()

        self.assertEqual(config.router_aux_loss_coef, 0.0001)
        self.assertEqual(config.router_bias_update_rate, 0.001)
        self.assertEqual(config.routed_scaling_factor, 2.5)
        self.assertTrue(config.load_with_bias)
        self.assertFalse(config.qk_layernorm)

    def test_attention_qk_layernorm_checkpoint_keys(self):
        config = ChimeraConfig(
            hidden_size=8,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=4,
            qk_layernorm=True,
        )
        attention = ChimeraAttention(config, layer_idx=0)
        keys = set(attention.state_dict())

        self.assertIn("q_norm.weight", keys)
        self.assertIn("k_norm.weight", keys)

    def test_attention_without_qk_layernorm_has_no_norm_keys(self):
        config = ChimeraConfig(
            hidden_size=8,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=4,
            qk_layernorm=False,
        )
        attention = ChimeraAttention(config, layer_idx=0)
        keys = set(attention.state_dict())

        self.assertNotIn("q_norm.weight", keys)
        self.assertNotIn("k_norm.weight", keys)

    def test_attention_forward_with_qk_layernorm(self):
        config = ChimeraConfig(
            hidden_size=8,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=4,
            qk_layernorm=True,
        )
        attention = ChimeraAttention(config, layer_idx=0)
        hidden_states = torch.randn(2, 5, config.hidden_size)
        position_embeddings = (
            torch.ones(2, 5, config.head_dim),
            torch.zeros(2, 5, config.head_dim),
        )

        output, _ = attention(hidden_states, position_embeddings)

        self.assertEqual(output.shape, hidden_states.shape)

    def test_sparse_moe_checkpoint_keys_are_unchanged(self):
        block = ChimeraSparseMoeBlock(self.config)
        keys = set(block.state_dict())

        self.assertIn("experts.0.gate_proj.weight", keys)
        self.assertIn("experts.3.down_proj.weight", keys)
        self.assertIn("gate.e_score_correction_bias", keys)
        self.assertFalse(any("experts.experts" in key for key in keys))

    def test_sparse_moe_block_forward(self):
        block = ChimeraSparseMoeBlock(self.config)
        hidden_states = torch.randn(2, 5, self.config.hidden_size)

        output, router_logits = block(hidden_states)

        self.assertEqual(output.shape, hidden_states.shape)
        self.assertEqual(router_logits.shape, (10, self.config.n_routed_experts))

    def test_sparse_moe_route_can_ignore_expert_bias(self):
        router_logits = torch.tensor([[10.0, 9.0, 0.0, 0.0]])

        block_with_bias = ChimeraSparseMoeBlock(self.config)
        block_with_bias.gate.e_score_correction_bias.data = torch.tensor([0.0, 0.0, 20.0, 19.0])
        biased_indices, _ = block_with_bias.route_tokens_to_experts(router_logits)

        config_without_bias = deepcopy(self.config)
        config_without_bias.load_with_bias = False
        block_without_bias = ChimeraSparseMoeBlock(config_without_bias)
        block_without_bias.gate.e_score_correction_bias.data = torch.tensor([0.0, 0.0, 20.0, 19.0])
        unbiased_indices, _ = block_without_bias.route_tokens_to_experts(router_logits)

        self.assertEqual(set(biased_indices[0].tolist()), {2, 3})
        self.assertEqual(set(unbiased_indices[0].tolist()), {0, 1})

    def test_sparse_moe_load_can_skip_expert_bias(self):
        source_block = ChimeraSparseMoeBlock(self.config)
        source_block.gate.e_score_correction_bias.data.fill_(3.0)

        config_without_bias = deepcopy(self.config)
        config_without_bias.load_with_bias = False
        target_block = ChimeraSparseMoeBlock(config_without_bias)
        target_block.load_state_dict(source_block.state_dict(), strict=False)

        self.assertTrue(torch.equal(target_block.gate.e_score_correction_bias, torch.zeros(4)))


if __name__ == "__main__":
    unittest.main()
