import os

def extract_class(filepath, class_name):
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            lines = f.readlines()
    except FileNotFoundError:
        return ''
    
    start_idx = -1
    for i, line in enumerate(lines):
        if line.startswith(f'class {class_name}(') or line.startswith(f'class {class_name}:'):
            start_idx = i
            break
            
    if start_idx == -1: return ''
    
    end_idx = len(lines)
    for i in range(start_idx + 1, len(lines)):
        if lines[i].startswith('class ') or (lines[i].startswith('def ') and not lines[i].startswith('    ')):
            end_idx = i
            break
            
    return ''.join(lines[start_idx:end_idx])

def extract_func(filepath, func_name):
    with open(filepath, 'r', encoding='utf-8') as f: lines = f.readlines()
    start = -1
    for i, l in enumerate(lines):
        if l.startswith(f'def {func_name}('):
            start = i
            break
    if start == -1: return ''
    end = len(lines)
    for i in range(start + 1, len(lines)):
        if lines[i].startswith('def ') or lines[i].startswith('class '):
            end = i
            break
    return ''.join(lines[start:end])

base_dir = 'src/transformers/models'
llama_file = os.path.join(base_dir, 'llama/modeling_llama.py')
ds_file = os.path.join(base_dir, 'deepseek_v2/modeling_deepseek_v2.py')

rotary = extract_class(llama_file, 'LlamaRotaryEmbedding')
rmsnorm = extract_class(llama_file, 'LlamaRMSNorm')
attention = extract_class(llama_file, 'LlamaAttention')
sdpa_attention = extract_class(llama_file, 'LlamaSdpaAttention')
flash_attention = extract_class(llama_file, 'LlamaFlashAttention2')
ds_mlp = extract_class(ds_file, 'DeepseekV2MLP')
ds_moe = extract_class(ds_file, 'DeepseekV2MoE')

repeat_kv = extract_func(llama_file, 'repeat_kv')
apply_rotary = extract_func(llama_file, 'apply_rotary_pos_emb')

header = """# coding=utf-8
# Copyright 2026 Surya Vikram and The HuggingFace Inc. team. All rights reserved.
\"\"\" PyTorch Chimera model.\"\"\"

import math
import warnings
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...activations import ACT2FN
from ...cache_utils import Cache, DynamicCache, StaticCache
from ...modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from ...modeling_utils import PreTrainedModel
from ...utils import logging
from .configuration_chimera import ChimeraConfig

logger = logging.get_logger(__name__)

"""

content = header + repeat_kv + apply_rotary + rotary + rmsnorm + attention + sdpa_attention + flash_attention + ds_mlp + ds_moe

content = content.replace('Llama', 'Chimera').replace('llama', 'chimera')
content = content.replace('DeepseekV2', 'Chimera').replace('deepseek_v2', 'chimera')

custom_arch = """
class ChimeraDecoderLayer(nn.Module):
    def __init__(self, config: ChimeraConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.layer_idx = layer_idx
        
        self.input_layernorm = ChimeraRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.self_attn = ChimeraAttention(config=config, layer_idx=layer_idx)
        self.post_attention_layernorm = ChimeraRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        
        is_dense = False
        if layer_idx < config.first_k_dense_replace or layer_idx >= (config.num_hidden_layers - config.last_k_dense_replace):
            is_dense = True
            
        if is_dense:
            self.mlp = ChimeraMLP(config)
        else:
            self.mlp = ChimeraMoE(config)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        position_ids=None,
        past_key_value=None,
        output_attentions=False,
        use_cache=False,
        cache_position=None,
        **kwargs,
    ):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            **kwargs,
        )
        hidden_states = residual + hidden_states
        
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        
        outputs = (hidden_states,)
        if output_attentions:
            outputs += (self_attn_weights,)
        if use_cache:
            outputs += (present_key_value,)
            
        return outputs

class ChimeraPreTrainedModel(PreTrainedModel):
    config_class = ChimeraConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["ChimeraDecoderLayer"]
    _skip_keys_device_placement = "past_key_values"
    _supports_flash_attn_2 = True
    _supports_sdpa = True
    _supports_cache_class = True
    
    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

class ChimeraModel(ChimeraPreTrainedModel):
    def __init__(self, config: ChimeraConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [ChimeraDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = ChimeraRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_init()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        cache_position=None,
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        hidden_states = inputs_embeds

        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None

        for idx, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_values,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
            )
            hidden_states = layer_outputs[0]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, past_key_values, all_hidden_states, all_self_attns] if v is not None)
            
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )

class ChimeraForCausalLM(ChimeraPreTrainedModel):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.model = ChimeraModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        cache_position=None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
        )
        
        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)
        
        if self.config.final_logit_softcapping is not None:
            logits = logits / self.config.final_logit_softcapping
            logits = torch.tanh(logits)
            logits = logits * self.config.final_logit_softcapping
            
        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, self.config.vocab_size), shift_labels.view(-1))
            
        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
"""

content += custom_arch

out_dir = os.path.join(base_dir, 'chimera')
os.makedirs(out_dir, exist_ok=True)
with open(os.path.join(out_dir, 'modeling_chimera.py'), 'w', encoding='utf-8') as f:
    f.write(content)
