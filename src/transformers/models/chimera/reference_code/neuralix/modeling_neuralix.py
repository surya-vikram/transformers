import torch
import torch.nn as nn
from transformers.modeling_utils import PreTrainedModel
from transformers.models.llama.modeling_llama import LlamaAttention, LlamaRMSNorm
from transformers.models.deepseek_v2.modeling_deepseek_v2 import DeepseekV2MoE, DeepseekV2MLP
from .configuration_neuralix import NeuralixConfig

class NeuralixDecoderLayer(nn.Module):
    def __init__(self, config: NeuralixConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.layer_idx = layer_idx
        
        self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        
        # We use LlamaAttention because it natively supports GQA and YaRN.
        self.self_attn = LlamaAttention(config=config, layer_idx=layer_idx)
        
        self.post_attention_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        
        # Routing logic: First k, last k layers are Dense, otherwise MoE.
        is_dense = False
        if layer_idx < config.first_k_dense_replace or layer_idx >= (config.num_hidden_layers - config.last_k_dense_replace):
            is_dense = True
            
        if is_dense:
            # Reusing Deepseek's standard MLP for the dense layers
            self.mlp = DeepseekV2MLP(config)
        else:
            # Reusing Deepseek's MoE block (Shared Expert + noaux_tc)
            self.mlp = DeepseekV2MoE(config)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        position_ids=None,
        past_key_value=None,
        output_attentions=False,
        use_cache=False,
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

class NeuralixModel(PreTrainedModel):
    config_class = NeuralixConfig
    
    def __init__(self, config: NeuralixConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [NeuralixDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        
        self.gradient_checkpointing = False
        self.post_init()

    def forward(self, input_ids, attention_mask=None, position_ids=None, past_key_values=None, use_cache=None, **kwargs):
        hidden_states = self.embed_tokens(input_ids)
        
        for idx, decoder_layer in enumerate(self.layers):
            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_values[idx] if past_key_values is not None else None,
                use_cache=use_cache,
                **kwargs,
            )
            hidden_states = layer_outputs[0]
            
        hidden_states = self.norm(hidden_states)
        return hidden_states

class NeuralixForCausalLM(PreTrainedModel):
    config_class = NeuralixConfig
    
    def __init__(self, config):
        super().__init__(config)
        self.model = NeuralixModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

    def forward(self, input_ids, attention_mask=None, position_ids=None, past_key_values=None, use_cache=None, labels=None, **kwargs):
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            **kwargs
        )
        
        hidden_states = outputs
        logits = self.lm_head(hidden_states)
        
        # SOTA Polish: Gemma 4 Logit Softcapping
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
            
        return {"loss": loss, "logits": logits}
