from transformers.configuration_utils import PretrainedConfig

class NeuralixConfig(PretrainedConfig):
    model_type = "neuralix"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size=51200,
        hidden_size=2048,
        intermediate_size=8192,
        num_hidden_layers=28,
        num_attention_heads=16,
        num_key_value_heads=2,
        hidden_act="silu",
        max_position_embeddings=16384,
        initializer_range=0.02,
        rms_norm_eps=1e-5,
        use_cache=True,
        rope_theta=10000.0,
        rope_scaling={"type": "yarn", "factor": 2.0, "original_max_position_embeddings": 16384},
        attention_bias=False,
        attention_dropout=0.0,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        tie_word_embeddings=False,
        # MoE configurations
        moe_intermediate_size=1024,
        n_routed_experts=64,
        num_experts_per_tok=4,
        n_shared_experts=1,
        scoring_func="sigmoid",
        topk_method="noaux_tc",
        moe_layer_freq=1,
        first_k_dense_replace=2,
        last_k_dense_replace=1,
        routed_scaling_factor=2.0,
        # Stability polish
        final_logit_softcapping=30.0,
        swiglu_limit=10.0,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        
        if num_key_value_heads is None:
            self.num_key_value_heads = num_attention_heads
        else:
            self.num_key_value_heads = num_key_value_heads
            
        self.hidden_act = hidden_act
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.tie_word_embeddings = tie_word_embeddings
        
        # MoE parameters
        self.moe_intermediate_size = moe_intermediate_size
        self.n_routed_experts = n_routed_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.n_shared_experts = n_shared_experts
        self.scoring_func = scoring_func
        self.topk_method = topk_method
        self.moe_layer_freq = moe_layer_freq
        self.first_k_dense_replace = first_k_dense_replace
        self.last_k_dense_replace = last_k_dense_replace
        self.routed_scaling_factor = routed_scaling_factor
        
        # Stability
        self.final_logit_softcapping = final_logit_softcapping
        self.swiglu_limit = swiglu_limit
        
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )
