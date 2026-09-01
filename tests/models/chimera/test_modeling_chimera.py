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
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import torch

from transformers import ChimeraConfig, ChimeraForCausalLM
from transformers.models.chimera.configuration_chimera import CHIMERA_CONTEXT_PHASES
from transformers.models.chimera.scripts.export_to_hf import build_config
from transformers.models.chimera.scripts.infer import prepare_inputs
from transformers.testing_utils import require_torch
from transformers.utils import is_torch_available


if is_torch_available():
    from transformers.models.chimera.modeling_chimera import (
        ChimeraAttention,
        ChimeraExperts,
        ChimeraRotaryEmbedding,
        ChimeraSparseMoeBlock,
        ChimeraTopkRouter,
    )


@require_torch
class ChimeraExpertsTest(unittest.TestCase):
    def setUp(self):
        self.config = ChimeraConfig(
            hidden_size=8,
            intermediate_size=16,
            moe_intermediate_size=12,
            shared_expert_intermediate_size=0,
            num_hidden_layers=3,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=4,
            first_k_dense_replace=2,
            n_routed_experts=4,
            num_experts_per_tok=2,
            n_shared_experts=0,
            qk_layernorm=True,
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

        self.assertEqual(config.n_routed_experts, 32)
        self.assertEqual(config.moe_intermediate_size, 2048)
        self.assertEqual(config.n_shared_experts, 0)
        self.assertEqual(config.shared_expert_intermediate_size, 0)
        self.assertEqual(config.max_position_embeddings, 8192)
        self.assertEqual(config.original_max_position_embeddings, 8192)
        self.assertEqual(config.context_phase, "8k")
        self.assertEqual(config.position_embedding_type, "yarn")
        self.assertEqual(config.rms_norm_eps, 1e-5)
        self.assertEqual(config.rope_parameters["factor"], 1.0)
        self.assertFalse(config.rope_parameters["truncate"])
        self.assertEqual(config.router_aux_loss_coef, 0.0)
        self.assertEqual(config.router_z_loss_coef, 0.001)
        self.assertEqual(config.router_bias_update_rate, 0.0)
        self.assertEqual(config.router_load_balancing_type, "quantile_balancing")
        self.assertEqual(config.moe_qb_num_bins, 1000)
        self.assertEqual(config.moe_qb_ema_decay, 0.0)
        self.assertEqual(config.routed_scaling_factor, 2.5)
        self.assertTrue(config.load_with_bias)
        self.assertTrue(config.qk_layernorm)

    def test_all_context_phase_configs_round_trip(self):
        for phase, geometry in CHIMERA_CONTEXT_PHASES.items():
            with self.subTest(phase=phase), TemporaryDirectory() as tmpdir:
                config = build_config(profile="tiny", context_phase=phase)
                config.save_pretrained(tmpdir)
                reloaded = ChimeraConfig.from_pretrained(tmpdir)

                self.assertEqual(reloaded.context_phase, phase)
                self.assertEqual(reloaded.position_embedding_type, "yarn")
                self.assertEqual(reloaded.max_position_embeddings, geometry["max_position_embeddings"])
                self.assertEqual(reloaded.original_max_position_embeddings, 8192)
                self.assertEqual(reloaded.rope_parameters["rope_type"], "yarn")
                self.assertEqual(reloaded.rope_parameters["factor"], geometry["factor"])
                self.assertFalse(reloaded.rope_parameters["truncate"])

    def test_context_phase_rejects_non_yarn_and_mismatched_geometry(self):
        with self.assertRaisesRegex(ValueError, "only position_embedding_type='yarn'"):
            ChimeraConfig(position_embedding_type="none")
        with self.assertRaisesRegex(ValueError, "requires factor=4.0"):
            ChimeraConfig(
                context_phase="32k",
                max_position_embeddings=32768,
                rope_scaling={"type": "yarn", "factor": 1.0},
            )
        with self.assertRaisesRegex(ValueError, "does not match"):
            ChimeraConfig(context_phase="8k", max_position_embeddings=32768)

    def test_all_context_phases_forward_and_cache(self):
        input_ids = torch.tensor([[4, 5, 6, 7]])
        for phase, geometry in CHIMERA_CONTEXT_PHASES.items():
            with self.subTest(phase=phase):
                config = ChimeraConfig(
                    vocab_size=32,
                    hidden_size=8,
                    intermediate_size=16,
                    moe_intermediate_size=12,
                    num_hidden_layers=1,
                    num_attention_heads=2,
                    num_key_value_heads=1,
                    head_dim=4,
                    first_k_dense_replace=1,
                    n_routed_experts=4,
                    num_experts_per_tok=2,
                    context_phase=phase,
                    max_position_embeddings=geometry["max_position_embeddings"],
                    eos_token_id=None,
                )
                model = ChimeraForCausalLM(config).eval()
                outputs = model(input_ids, use_cache=True)
                self.assertEqual(outputs.past_key_values.get_seq_length(), 4)
                next_outputs = model(
                    torch.tensor([[8]]),
                    past_key_values=outputs.past_key_values,
                    use_cache=True,
                )
                generated = model.generate(input_ids, max_new_tokens=2, do_sample=False)

                self.assertEqual(outputs.logits.shape, (1, 4, config.vocab_size))
                self.assertEqual(next_outputs.past_key_values.get_seq_length(), 5)
                self.assertEqual(next_outputs.logits.shape, (1, 1, config.vocab_size))
                self.assertTrue(torch.isfinite(next_outputs.logits).all())
                self.assertEqual(generated.shape, (1, 6))

    def test_all_context_phases_rotary_boundary_values_are_finite(self):
        for phase, geometry in CHIMERA_CONTEXT_PHASES.items():
            with self.subTest(phase=phase):
                config = build_config(profile="tiny", context_phase=phase)
                rotary = ChimeraRotaryEmbedding(config)
                positions = torch.tensor(
                    [[0, 8191, geometry["max_position_embeddings"] - 1]]
                )
                cos, sin = rotary(torch.zeros(1, 3, config.hidden_size), positions)

                self.assertEqual(cos.shape, (1, 3, config.head_dim))
                self.assertEqual(sin.shape, cos.shape)
                self.assertTrue(torch.isfinite(cos).all())
                self.assertTrue(torch.isfinite(sin).all())

    def test_raw_inference_does_not_insert_special_tokens(self):
        class RecordingTokenizer:
            def __call__(self, prompt, **kwargs):
                self.prompt = prompt
                self.kwargs = kwargs
                return {"input_ids": torch.tensor([[4, 5]])}

        tokenizer = RecordingTokenizer()
        inputs = prepare_inputs(
            tokenizer,
            "The capital of France is",
            SimpleNamespace(chat=False, system_prompt=None),
        )

        self.assertEqual(tokenizer.prompt, "The capital of France is")
        self.assertFalse(tokenizer.kwargs["add_special_tokens"])
        self.assertEqual(inputs["input_ids"].tolist(), [[4, 5]])

    def test_chat_inference_uses_template_without_duplicate_special_tokens(self):
        class RecordingTokenizer:
            def apply_chat_template(self, messages, **kwargs):
                self.messages = messages
                self.template_kwargs = kwargs
                return "<rendered-chat>"

            def __call__(self, prompt, **kwargs):
                self.prompt = prompt
                self.tokenizer_kwargs = kwargs
                return {"input_ids": torch.tensor([[2, 4, 5, 2]])}

        tokenizer = RecordingTokenizer()
        inputs = prepare_inputs(
            tokenizer,
            "Complete this request",
            SimpleNamespace(chat=True, system_prompt="Be concise"),
        )

        self.assertEqual(
            tokenizer.messages,
            [
                {"role": "system", "content": "Be concise"},
                {"role": "user", "content": "Complete this request"},
            ],
        )
        self.assertFalse(tokenizer.template_kwargs["tokenize"])
        self.assertTrue(tokenizer.template_kwargs["add_generation_prompt"])
        self.assertEqual(tokenizer.prompt, "<rendered-chat>")
        self.assertFalse(tokenizer.tokenizer_kwargs["add_special_tokens"])
        self.assertEqual(inputs["input_ids"].tolist(), [[2, 4, 5, 2]])

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

    def test_sparse_moe_load_preserves_expert_bias_when_use_is_disabled(self):
        source_block = ChimeraSparseMoeBlock(self.config)
        source_block.gate.e_score_correction_bias.data.fill_(3.0)

        config_without_bias = deepcopy(self.config)
        config_without_bias.load_with_bias = False
        target_block = ChimeraSparseMoeBlock(config_without_bias)
        self.assertEqual(set(source_block.state_dict()), set(target_block.state_dict()))
        target_block.load_state_dict(source_block.state_dict(), strict=True)

        self.assertTrue(torch.equal(target_block.gate.e_score_correction_bias, torch.full((4,), 3.0)))
        self.assertFalse(target_block.gate.e_score_correction_bias.requires_grad)

    def test_router_load_weights_can_load_expert_bias(self):
        router = ChimeraTopkRouter(self.config)
        loaded_params = router.load_weights(
            [
                ("weight", torch.ones_like(router.weight)),
                ("e_score_correction_bias", torch.full_like(router.e_score_correction_bias, 3.0)),
            ]
        )

        self.assertEqual(loaded_params, {"weight", "e_score_correction_bias"})
        self.assertTrue(torch.equal(router.weight, torch.ones_like(router.weight)))
        self.assertTrue(
            torch.equal(router.e_score_correction_bias, torch.full_like(router.e_score_correction_bias, 3.0))
        )

    def test_router_load_weights_restores_exact_fp32_frozen_expert_bias(self):
        router = ChimeraTopkRouter(self.config)
        original_device = router.e_score_correction_bias.device
        router.e_score_correction_bias.data = router.e_score_correction_bias.data.to(torch.bfloat16)
        router.e_score_correction_bias.requires_grad_(True)
        checkpoint_bias = torch.tensor([0.12345679, -0.9876543, 0.33333334, -0.14285715])
        self.assertFalse(torch.equal(checkpoint_bias, checkpoint_bias.to(torch.bfloat16).float()))

        loaded_params = router.load_weights([("e_score_correction_bias", checkpoint_bias)])

        self.assertEqual(loaded_params, {"e_score_correction_bias"})
        self.assertEqual(router.e_score_correction_bias.device, original_device)
        self.assertEqual(router.e_score_correction_bias.dtype, torch.float32)
        self.assertFalse(router.e_score_correction_bias.requires_grad)
        self.assertTrue(torch.equal(router.e_score_correction_bias, checkpoint_bias))

    def test_router_bias_stays_float32_when_model_uses_bfloat16(self):
        router = ChimeraTopkRouter(self.config)
        router.e_score_correction_bias.data.copy_(torch.arange(self.config.n_routed_experts))

        router.to(dtype=torch.bfloat16)

        self.assertEqual(router.weight.dtype, torch.bfloat16)
        self.assertEqual(router.e_score_correction_bias.dtype, torch.float32)
        self.assertTrue(
            torch.equal(
                router.e_score_correction_bias,
                torch.arange(self.config.n_routed_experts, dtype=torch.float32),
            )
        )

    def test_router_bias_stays_float32_when_pretrained_model_loads_bfloat16(self):
        config = deepcopy(self.config)
        config.vocab_size = 32
        model = ChimeraForCausalLM(config)

        with TemporaryDirectory() as tmpdir:
            model.save_pretrained(tmpdir)
            reloaded = ChimeraForCausalLM.from_pretrained(tmpdir, dtype=torch.bfloat16)

        router_biases = [
            parameter
            for name, parameter in reloaded.named_parameters()
            if name.endswith(".gate.e_score_correction_bias")
        ]
        self.assertTrue(router_biases)
        self.assertTrue(all(parameter.dtype == torch.float32 for parameter in router_biases))
        self.assertFalse(any(parameter.requires_grad for parameter in router_biases))

    def test_router_load_weights_preserves_expert_bias_when_use_is_disabled(self):
        config_without_bias = deepcopy(self.config)
        config_without_bias.load_with_bias = False
        router = ChimeraTopkRouter(config_without_bias)
        router.e_score_correction_bias.fill_(2.0)

        loaded_params = router.load_weights(
            [
                ("weight", torch.ones_like(router.weight)),
                ("e_score_correction_bias", torch.full_like(router.e_score_correction_bias, 3.0)),
            ]
        )

        self.assertEqual(loaded_params, {"weight", "e_score_correction_bias"})
        self.assertTrue(torch.equal(router.weight, torch.ones_like(router.weight)))
        self.assertTrue(
            torch.equal(router.e_score_correction_bias, torch.full_like(router.e_score_correction_bias, 3.0))
        )

    def test_router_strict_load_rejects_missing_expert_bias(self):
        router = ChimeraTopkRouter(self.config)

        with self.assertRaisesRegex(RuntimeError, "e_score_correction_bias"):
            router.load_state_dict({"weight": torch.ones_like(router.weight)}, strict=True)

    def test_from_pretrained_rejects_missing_expert_bias_in_both_modes(self):
        config = deepcopy(self.config)
        config.vocab_size = 32
        model = ChimeraForCausalLM(config)
        state_dict = model.state_dict()
        missing_key = next(key for key in state_dict if key.endswith(".gate.e_score_correction_bias"))
        state_dict.pop(missing_key)

        for load_with_bias in (True, False):
            load_config = deepcopy(config)
            load_config.load_with_bias = load_with_bias
            with self.subTest(load_with_bias=load_with_bias):
                with self.assertRaisesRegex(RuntimeError, missing_key):
                    ChimeraForCausalLM.from_pretrained(None, config=load_config, state_dict=state_dict)

    def test_model_save_reload_keeps_identical_weights_and_frozen_biases_in_both_modes(self):
        config = deepcopy(self.config)
        config.vocab_size = 32
        model = ChimeraForCausalLM(config)
        for module in model.modules():
            if isinstance(module, ChimeraTopkRouter):
                module.e_score_correction_bias.data.copy_(
                    torch.arange(config.n_routed_experts, dtype=module.e_score_correction_bias.dtype)
                )

        with TemporaryDirectory() as tmpdir:
            model.save_pretrained(tmpdir)
            source_state = model.state_dict()
            for load_with_bias in (True, False):
                load_config = ChimeraConfig.from_pretrained(tmpdir)
                load_config.load_with_bias = load_with_bias
                reloaded = ChimeraForCausalLM.from_pretrained(tmpdir, config=load_config)

                reloaded_state = reloaded.state_dict()
                self.assertEqual(set(source_state), set(reloaded_state))
                for key in source_state:
                    self.assertTrue(torch.equal(source_state[key], reloaded_state[key]), key)
                router_biases = [
                    parameter
                    for name, parameter in reloaded.named_parameters()
                    if name.endswith(".gate.e_score_correction_bias")
                ]
                expected_router_biases = (
                    config.num_hidden_layers - config.first_k_dense_replace - config.last_k_dense_replace
                )
                self.assertEqual(len(router_biases), expected_router_biases)
                self.assertFalse(any(parameter.requires_grad for parameter in router_biases))


if __name__ == "__main__":
    unittest.main()
