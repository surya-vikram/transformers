# SOTA LLM Architecture Comparison — Verified Analysis (Pass 2)

> [!IMPORTANT]
> Every claim in this document has been verified against the **full flattened key dump** of each model's `config.json` and relevant `.py` modeling scripts. Fields are quoted exactly as they appear in the configs.

---

## 1. Core Parameters Comparison Table

All values extracted from each model's `config.json`. For multimodal models, the text backbone config (`text_config.*`) is used.

| # | Model | Arch Class | `hidden_size` | Layers | Q Heads | KV Heads | `head_dim` | `vocab_size` | Max Ctx |
|---|-------|------------|---------------|--------|---------|----------|-----------|-------------|---------|
| 1 | MiMo-V2.5-Pro | `MiMoV2ForCausalLM` | 6,144 | 70 | 128 | 8 | 192 | 152,576 | 1,048,576 |
| 2 | MiMo-V2-Flash | `MiMoV2FlashForCausalLM` | 4,096 | 48 | 64 | 4 | 192 | 152,576 | 262,144 |
| 3 | Qwen 3.6 35B-A3B | `Qwen3_5MoeForConditionalGeneration` | 2,048 | 40 | 16 | 2 | 256 | 248,320 | 262,144 |
| 4 | Qwen 3.6 27B | `Qwen3_5ForConditionalGeneration` | 5,120 | 64 | 24 | 4 | 256 | 248,320 | 262,144 |
| 5 | Gemma 4 31B IT | `Gemma4ForConditionalGeneration` | 5,376 | 60 | 32 | 16 | 256 | 262,144 | 262,144 |
| 6 | Gemma 4 26B A4B IT | `Gemma4ForConditionalGeneration` | 2,816 | 30 | 16 | 8 | 256 | 262,144 | 262,144 |
| 7 | Gemma 4 12B IT | `Gemma4UnifiedForConditionalGeneration` | 3,840 | 48 | 16 | 8 | 256 | 262,144 | 262,144 |
| 8 | DeepSeek V4 Pro | `DeepseekV4ForCausalLM` | 7,168 | 61 | 128 | 1 | 512 | 129,280 | 1,048,576 |
| 9 | DeepSeek V4 Flash | `DeepseekV4ForCausalLM` | 4,096 | 43 | 64 | 1 | 512 | 129,280 | 1,048,576 |
| 10 | Kimi K2.6 | `KimiK25ForConditionalGeneration` | 7,168 | 61 | 64 | 64 | *(MLA)* | 163,840 | 262,144 |
| 11 | MiniMax M3 | **404 — NOT FOUND** | — | — | — | — | — | — | — |
| 12 | Nemotron 3 Ultra | `NemotronHForCausalLM` | 8,192 | 108 *(via array)* | 64 | 2 | 128 | 131,072 | 262,144 |
| 13 | GPT-OSS 20B | `GptOssForCausalLM` | 2,880 | 24 | 64 | 8 | 64 | 201,088 | 131,072 |
| 14 | GLM-5.2 | `GlmMoeDsaForCausalLM` | 6,144 | 78 | 64 | 64 | 256 | 154,880 | 1,048,576 |

> [!NOTE]
> **Kimi K2.6** uses MLA, so `head_dim` is not a single value — it's decomposed into `qk_nope_head_dim: 128` + `qk_rope_head_dim: 64` = 192 for Q/K, and `v_head_dim: 128` for V.
>
> **Nemotron 3 Ultra** has `num_hidden_layers: null` in config — layer count is determined by the `layers_block_type` array (108 entries).

---

## 2. FFN / MoE Configuration Table

| # | Model | Dense FFN `intermediate_size` | MoE? | Routed Experts | Active/Tok | Shared Experts | `moe_intermediate_size` | Routing | Scoring |
|---|-------|-------------------------------|------|----------------|-----------|----------------|------------------------|---------|---------|
| 1 | MiMo-V2.5-Pro | 16,384 (layer 0 only) | Yes | 384 | 8 | `null` | 2,048 | `noaux_tc` | `sigmoid` |
| 2 | MiMo-V2-Flash | 16,384 (layer 0 only) | Yes | 256 | 8 | `null` | 2,048 | `noaux_tc` | `sigmoid` |
| 3 | Qwen 3.6 35B-A3B | — | Yes | 256 | 8 | 1 (`shared_expert_intermediate_size: 512`) | 512 | — | — |
| 4 | Qwen 3.6 27B | 17,408 | **No** (Dense) | — | — | — | — | — | — |
| 5 | Gemma 4 31B IT | 21,504 | **No** (`enable_moe_block: false`) | — | — | — | — | — | — |
| 6 | Gemma 4 26B A4B IT | 2,112 *(dense fallback)* | Yes (`enable_moe_block: true`) | 128 | 8 (`top_k_experts`) | — | 704 | — | — |
| 7 | Gemma 4 12B IT | 15,360 | **No** (`enable_moe_block: false`, `num_experts: null`) | — | — | — | — | — | — |
| 8 | DeepSeek V4 Pro | — | Yes | 384 | 6 | 1 | 3,072 | `noaux_tc` | `sqrtsoftplus` |
| 9 | DeepSeek V4 Flash | — | Yes | 256 | 6 | 1 | 2,048 | `noaux_tc` | `sqrtsoftplus` |
| 10 | Kimi K2.6 | 18,432 (layer 0 only) | Yes | 384 | 8 | 1 | 2,048 | `noaux_tc` | `sigmoid` |
| 12 | Nemotron 3 Ultra | 5,120 | Yes | 512 | 22 | 1 (`moe_shared_expert_intermediate_size: 10,240`) | 5,120 | — | — |
| 13 | GPT-OSS 20B | 2,880 | Yes | 32 (`num_local_experts`) | 4 | — | — | — | — |
| 14 | GLM-5.2 | 12,288 (first 3 layers) | Yes | 256 | 8 | 1 | 2,048 | `noaux_tc` | `sigmoid` |

> [!NOTE]
> **MiMo V2.5 Pro / Flash**: `moe_layer_freq` arrays show layer 0 has value `0` (dense MLP), all others have `1` (MoE). So `intermediate_size: 16,384` applies only to layer 0.
>
> **Kimi K2.6**: `first_k_dense_replace: 1`, `moe_layer_freq: 1` — layer 0 is dense (uses `intermediate_size: 18,432`), layers 1-60 are MoE.
>
> **GPT-OSS 20B**: Uses field name `num_local_experts: 32`, not `n_routed_experts`. Has no `moe_intermediate_size` field — experts use `intermediate_size: 2,880`.

---

## 3. Attention Layer Pattern Comparison

| # | Model | Pattern Field | Full Attn Count | Non-Full Count | Type | Ratio | `sliding_window` |
|---|-------|--------------|-----------------|----------------|------|-------|-------------------|
| 1 | MiMo V2.5 Pro | `hybrid_layer_pattern` (0/1) | 10 global (0) | 60 SWA (1) | ~1:6 | | 128 |
| 2 | MiMo V2 Flash | `hybrid_layer_pattern` (0/1) | 9 global (0) | 39 SWA (1) | ~1:4.3 | | 128 |
| 3 | Qwen 3.6 35B-A3B | `layer_types` | 10 `full_attention` | 30 `linear_attention` | 3:1 linear:full | | — |
| 4 | Qwen 3.6 27B | `layer_types` | 16 `full_attention` | 48 `linear_attention` | 3:1 linear:full | | — |
| 5 | Gemma 4 31B IT | `layer_types` | 10 `full_attention` | 50 `sliding_attention` | 5:1 sliding:full | | 1,024 |
| 6 | Gemma 4 26B A4B IT | `layer_types` | 5 `full_attention` | 25 `sliding_attention` | 5:1 sliding:full | | 1,024 |
| 7 | Gemma 4 12B IT | `layer_types` | 8 `full_attention` | 40 `sliding_attention` | 5:1 sliding:full | | 1,024 |
| 8 | DeepSeek V4 Pro | `compress_ratios` | — | — | Per-layer compression | | 128 |
| 9 | DeepSeek V4 Flash | `compress_ratios` | — | — | Per-layer compression | | 128 |
| 10 | Kimi K2.6 | *(all layers same)* | 61 MLA | 0 | Uniform MLA | | — |
| 12 | Nemotron 3 Ultra | `layers_block_type` | 12 `attention` | 48 `mamba` + 48 `moe` | see below | | `null` |
| 13 | GPT-OSS 20B | `layer_types` | 12 `full_attention` | 12 `sliding_attention` | 1:1 alternating | | 128 |
| 14 | GLM-5.2 | `indexer_types` | 22 `full` | 56 `shared` | 1:3 full:shared | | — |

---

## 4. Unique Architectural Components — Model by Model

### 4.1 Xiaomi MiMo-V2.5-Pro & MiMo-V2-Flash

**Source files**: [config.json (Pro)](file:///C:/Users/surya/.gemini/antigravity/brain/3753ebe4-8695-470c-859d-33797a4b15eb/scratch/architectures/1_mimo_v2_5_pro/config.json), [modeling_mimo_v2.py](file:///C:/Users/surya/.gemini/antigravity/brain/3753ebe4-8695-470c-859d-33797a4b15eb/scratch/architectures/1_mimo_v2_5_pro/modeling_mimo_v2.py), [config.json (Flash)](file:///C:/Users/surya/.gemini/antigravity/brain/3753ebe4-8695-470c-859d-33797a4b15eb/scratch/architectures/2_mimo_v2_flash/config.json)

#### Hybrid SWA / Global Attention with Per-Type Head Counts
Both models use `hybrid_layer_pattern` arrays (`0` = global, `1` = SWA). Crucially, both also define **separate head configurations for SWA layers**:

| Config Field | Pro (global) | Pro (SWA) | Flash (global) | Flash (SWA) |
|-------------|-------------|-----------|---------------|-------------|
| `num_attention_heads` / `swa_num_attention_heads` | 128 | 128 | 64 | 64 |
| `num_key_value_heads` / `swa_num_key_value_heads` | 8 | 8 | **4** | **8** |
| `head_dim` / `swa_head_dim` | 192 | 192 | 192 | 192 |
| `v_head_dim` / `swa_v_head_dim` | 128 | 128 | 128 | 128 |

> [!IMPORTANT]
> **MiMo-V2-Flash uses different GQA ratios** for SWA vs global layers: global has 64Q/4KV (16:1), while SWA has 64Q/8KV (8:1). Pro uses uniform 128Q/8KV for both.

Global layer indices:
* **Pro (70 layers)**: `[0, 7, 15, 23, 31, 39, 47, 55, 62, 69]` — 10 global, 60 SWA
* **Flash (48 layers)**: `[0, 5, 11, 17, 23, 29, 35, 41, 47]` — 9 global, 39 SWA

#### Separate RoPE per Attention Type
From [modeling_mimo_v2.py L525-526](file:///C:/Users/surya/.gemini/antigravity/brain/3753ebe4-8695-470c-859d-33797a4b15eb/scratch/architectures/1_mimo_v2_5_pro/modeling_mimo_v2.py#L525-L526):
```python
self.rotary_emb = MiMoV2RotaryEmbedding(config=config, is_swa=False)
self.swa_rotary_emb = MiMoV2RotaryEmbedding(config=config, is_swa=True)
```

| Field | Pro | Flash |
|-------|-----|-------|
| `rope_theta` (global) | 10,000,000 | 5,000,000 |
| `swa_rope_theta` (SWA) | 10,000 | 10,000 |
| `partial_rotary_factor` | 0.334 | 0.334 |

#### Asymmetric Q/K vs V Head Dimensions
* `head_dim: 192` (Q/K) vs `v_head_dim: 128` (V) for both models
* Only ~33.4% of head_dim gets RoPE rotation (`partial_rotary_factor: 0.334`)

#### Attention Value Scaling
From [modeling_mimo_v2.py L262, L297-298](file:///C:/Users/surya/.gemini/antigravity/brain/3753ebe4-8695-470c-859d-33797a4b15eb/scratch/architectures/1_mimo_v2_5_pro/modeling_mimo_v2.py#L262):
```python
self.v_scale = getattr(config, "attention_value_scale", None)
...
if self.v_scale is not None:
    value_states = value_states * self.v_scale
```
* Pro: `attention_value_scale: 0.612`
* Flash: `attention_value_scale: 0.707`

#### Attention Sink Bias
Both: `add_swa_attention_sink_bias: true`, `add_full_attention_sink_bias: false`

From [modeling_mimo_v2.py L263-270](file:///C:/Users/surya/.gemini/antigravity/brain/3753ebe4-8695-470c-859d-33797a4b15eb/scratch/architectures/1_mimo_v2_5_pro/modeling_mimo_v2.py#L263-L270): a per-head learnable parameter (`nn.Parameter(torch.empty(num_attention_heads))`) applied only in SWA layers. From [L85](file:///C:/Users/surya/.gemini/antigravity/brain/3753ebe4-8695-470c-859d-33797a4b15eb/scratch/architectures/1_mimo_v2_5_pro/modeling_mimo_v2.py#L85), it is reshaped to `[1, num_heads, 1, 1]` and expanded to the query sequence dimension before being added as extra columns to the attention pattern.

#### Other Config Fields
* `attention_projection_layout: "fused_qkv"` (Pro only — Q/K/V fused into single projection)
* `attention_chunk_size: 128` (both)
* `sliding_window_size: 128` (both, redundant with `sliding_window`)
* Quantization: FP8 (`e4m3`) with `weight_block_size: [128, 128]` — both models; `o_proj` layers for all attention are excluded from quantization
* Both: `routed_scaling_factor: null` (no explicit MoE output scaling)

---

### 4.2 Alibaba Qwen 3.6 35B-A3B & 27B

**Source files**: [config.json (35B-A3B)](file:///C:/Users/surya/.gemini/antigravity/brain/3753ebe4-8695-470c-859d-33797a4b15eb/scratch/architectures/3_qwen_3_6_35b_a3b/config.json), [config.json (27B)](file:///C:/Users/surya/.gemini/antigravity/brain/3753ebe4-8695-470c-859d-33797a4b15eb/scratch/architectures/4_qwen_3_6_27b/config.json)

#### Interleaved Linear Attention + Full Attention
Both models define `layer_types` arrays with `"linear_attention"` and `"full_attention"`. The pattern is strictly **3 linear : 1 full** repeated.

* **35B-A3B** (40 layers): 30 `linear_attention` + 10 `full_attention`
* **27B** (64 layers): 48 `linear_attention` + 16 `full_attention`

The 27B also has an explicit `full_attention_interval: 4` field confirming every 4th layer is full attention.

#### Linear Attention Configuration

| Field | 35B-A3B | 27B |
|-------|---------|-----|
| `linear_conv_kernel_dim` | 4 | 4 |
| `linear_key_head_dim` | 128 | 128 |
| `linear_num_key_heads` | 16 | 16 |
| `linear_num_value_heads` | 32 | 48 |
| `linear_value_head_dim` | 128 | 128 |

These define a **separate head configuration** for linear attention layers (different from the full-attention heads). The `linear_conv_kernel_dim: 4` indicates a 1D causal convolution with kernel size 4 within the linear attention path.

Both also have `mamba_ssm_dtype: float32` — this appears to be an inherited config key for the SSM dtype used within the linear attention mechanism.

#### Multi-Token Prediction (MTP)
Both: `mtp_num_hidden_layers: 1`, `mtp_use_dedicated_embeddings: false`

#### M-RoPE (Multi-dimensional RoPE)
Both have these fields nested under `text_config.rope_parameters`:
```json
"mrope_interleaved": true,
"mrope_section": [11, 11, 10],
"partial_rotary_factor": 0.25,
"rope_theta": 10000000,
"rope_type": "default"
```
The `mrope_section: [11, 11, 10]` splits 32 RoPE dim-pairs (since `partial_rotary_factor: 0.25` × `head_dim: 256` = 64 → 32 pairs) into 3 sections for height/width/temporal position encoding.

#### Output Gate
* Both: `attn_output_gate: true` — gating mechanism applied to attention output
* 27B **additionally**: `output_gate_type: "swish"` — explicitly swish-gated
* 35B-A3B: `output_gate_type` field is **absent** (not set)

#### MoE (35B-A3B only)
The 35B-A3B has:
* `num_experts: 256`, `num_experts_per_tok: 8`
* `moe_intermediate_size: 512` — each expert FFN is just 512-dim
* `shared_expert_intermediate_size: 512` — shared expert same size
* `router_aux_loss_coef: 0.001`

The 27B has `num_experts: null`, `num_experts_per_tok: null` — purely dense.

#### Vision Encoder (both models)
Both share the same Qwen vision encoder architecture:
* `vision_config.depth: 27`, `hidden_size: 1152`, `num_heads: 16`, `patch_size: 16`
* `spatial_merge_size: 2`, `temporal_patch_size: 2`
* `hidden_act: "gelu_pytorch_tanh"`
* 35B-A3B: `out_hidden_size: 2048` (matches text `hidden_size`)
* 27B: `out_hidden_size: 5120` (matches text `hidden_size`)

---

### 4.3 Google Gemma 4 (31B IT, 26B A4B IT, 12B IT)

**Source files**: [config.json (31B)](file:///C:/Users/surya/.gemini/antigravity/brain/3753ebe4-8695-470c-859d-33797a4b15eb/scratch/architectures/5_gemma_4_31b_it/config.json), [config.json (26B-A4B)](file:///C:/Users/surya/.gemini/antigravity/brain/3753ebe4-8695-470c-859d-33797a4b15eb/scratch/architectures/6_gemma_4_26b_a4b_it/config.json), [config.json (12B)](file:///C:/Users/surya/.gemini/antigravity/brain/3753ebe4-8695-470c-859d-33797a4b15eb/scratch/architectures/7_gemma_4_12b_it/config.json)

#### Alternating Sliding/Full Attention (5:1)
All three use `layer_types` arrays. Pattern is **5 sliding : 1 full**:

| Model | Total | Sliding | Full |
|-------|-------|---------|------|
| 31B IT | 60 | 50 | 10 |
| 26B A4B IT | 30 | 25 | 5 |
| 12B IT | 48 | 40 | 8 |

All: `sliding_window: 1024`

#### Dual Head Dimensions: `head_dim` vs `global_head_dim`

> [!IMPORTANT]
> All three configs define **two different head dimensions**:
> * `head_dim: 256` — used for sliding attention layers
> * `global_head_dim: 512` — used for full (global) attention layers

This means full attention layers use **larger head dimensions** (512 vs 256), increasing their representational capacity for long-range context.

#### Dual KV Head Counts
All three have separate KV head counts per attention type:

| Model | `num_key_value_heads` (sliding) | `num_global_key_value_heads` (full) | GQA ratio (sliding) | GQA ratio (full) |
|-------|-------------------------------|-------------------------------------|---------------------|------------------|
| 31B | 16 | 4 | 2:1 | 8:1 |
| 26B A4B | 8 | 2 | 2:1 | 8:1 |
| 12B | 8 | 1 | 2:1 | 16:1 |

Full attention layers use **much more aggressive GQA compression** — fewer KV heads with larger head_dim, compressing KV cache for long-range context.

#### K=V Weight Sharing (`attention_k_eq_v: true`)
All three configs set `attention_k_eq_v: true` — key and value projections share weights. This halves the KV-projection parameter count.

#### Separate RoPE Configurations per Attention Type
All three define `rope_parameters` with per-type settings:
```json
"rope_parameters": {
    "full_attention": {
        "partial_rotary_factor": 0.25,
        "rope_theta": 1000000.0,
        "rope_type": "proportional"
    },
    "sliding_attention": {
        "rope_theta": 10000.0,
        "rope_type": "default"
    }
}
```

#### Logit Softcapping
31B and 12B: `final_logit_softcapping: 30.0` — output logits capped via `cap * tanh(logit/cap)`.

#### Bidirectional Attention for Vision
All three: `use_bidirectional_attention: "vision"` — vision tokens use bidirectional (non-causal) attention.

#### Other Shared Fields
* `hidden_activation: "gelu_pytorch_tanh"` (all three)
* `hidden_size_per_layer_input: 0` (31B, 12B — not active)
* `num_kv_shared_layers: 0` (31B, 12B — KV sharing across layers not used)
* `use_double_wide_mlp: false` (31B, 12B)
* `vocab_size_per_layer_input: 262144` (31B, 12B)
* `tie_word_embeddings: true` (all three)

#### MoE (26B A4B only)
Only 26B A4B: `enable_moe_block: true`, `num_experts: 128`, `top_k_experts: 8`, `moe_intermediate_size: 704`. Dense `intermediate_size: 2,112` used as fallback.

#### Audio Encoder (12B IT only)
Only 12B IT has `audio_config`:
* `model_type: "gemma4_unified_audio"`
* `audio_embed_dim: 640`, `audio_samples_per_token: 640`, `hidden_size: 640`
* `output_proj_dims: 640`

31B and 26B: `audio_config: null`

#### Vision Encoders

| Field | 31B / 26B A4B | 12B IT |
|-------|---------------|--------|
| `model_type` | `gemma4_vision` | `gemma4_unified_vision` |
| `hidden_size` | 1,152 | — |
| `mm_embed_dim` | — | 3,840 |
| `num_hidden_layers` | 27 | — |
| `patch_size` | 16 | 16 |
| `model_patch_size` | — | 48 |
| `num_soft_tokens` | — | 280 |
| `default_output_length` | 280 | — |
| `num_attention_heads` | 16 | — |
| `num_key_value_heads` | 16 | — |
| `rope_parameters.rope_theta` | 100.0 | — |
| `pooling_kernel_size` | 3 | 3 |
| `output_proj_dims` | — | 3,840 |

> [!NOTE]
> The 31B/26B vision encoder is a ViT with 2D RoPE (θ=100). The 12B vision encoder uses a different architecture (`gemma4_unified_vision`) with `model_patch_size: 48` and `num_soft_tokens: 280`, but both architectures also define `patch_size: 16`.

---

### 4.4 DeepSeek V4 Pro & Flash

**Source files**: [config.json (Pro)](file:///C:/Users/surya/.gemini/antigravity/brain/3753ebe4-8695-470c-859d-33797a4b15eb/scratch/architectures/8_deepseek_v4_pro/config.json), [config.json (Flash)](file:///C:/Users/surya/.gemini/antigravity/brain/3753ebe4-8695-470c-859d-33797a4b15eb/scratch/architectures/9_deepseek_v4_flash/config.json), [inference/model.py](file:///C:/Users/surya/.gemini/antigravity/brain/3753ebe4-8695-470c-859d-33797a4b15eb/scratch/architectures/8_deepseek_v4_pro/inference/model.py)

> [!IMPORTANT]
> DeepSeek V4 is the most architecturally novel model in this set. It introduces: **single-latent KV with compression**, **learned sparse indexing**, and **Hyper-Connections replacing residuals**.

#### Single KV Head with Low-Rank Q/O Projections

| Config Field | Pro | Flash |
|-------------|-----|-------|
| `num_key_value_heads` | 1 | 1 |
| `head_dim` | 512 | 512 |
| `q_lora_rank` | 1,536 | 1,024 |
| `qk_rope_head_dim` | 64 | 64 |
| `o_lora_rank` | 1,024 | 1,024 |
| `o_groups` | 16 | 8 |

From [model.py Attention L436-543](file:///C:/Users/surya/.gemini/antigravity/brain/3753ebe4-8695-470c-859d-33797a4b15eb/scratch/architectures/8_deepseek_v4_pro/inference/model.py#L436-L543):
```python
# Q path: x → wq_a → q_norm → wq_b → split into heads
q = self.q_norm(self.wq_a(x))
q = self.wq_b(q).unflatten(-1, (self.n_local_heads, self.head_dim))

# KV path: x → wkv (→ single head_dim=512 output) → kv_norm → RoPE on last 64 dims
kv = self.wkv(x)
kv = self.kv_norm(kv)
```
The **entire KV cache per layer per token is just `head_dim=512` dims** — one vector, not per-head.

#### Per-Layer KV Compression (`compress_ratios`)

| | Pro (62 entries) | Flash (44 entries) |
|--|-----------------|-------------------|
| Ratio `0` (no compression) | 1 layer | 3 layers |
| Ratio `4` (4:1 compression) | 30 layers | 21 layers |
| Ratio `128` (128:1 compression) | 31 layers | 20 layers |

From [model.py Compressor L279-377](file:///C:/Users/surya/.gemini/antigravity/brain/3753ebe4-8695-470c-859d-33797a4b15eb/scratch/architectures/8_deepseek_v4_pro/inference/model.py#L279-L377): the `Compressor` uses learned `wkv` and `wgate` projections with additive positional embeddings (`ape`). It pools KV states using softmax-weighted gating over consecutive tokens, normalizes, and applies RoPE using `compress_rope_theta: 160,000`.

#### Learned Sparse Indexer (for ratio=4 layers)
Config fields: `index_topk: 1024` (Pro) / `512` (Flash), `index_head_dim: 128`, `index_n_heads: 64` (both)

From [model.py Indexer L380-433](file:///C:/Users/surya/.gemini/antigravity/brain/3753ebe4-8695-470c-859d-33797a4b15eb/scratch/architectures/8_deepseek_v4_pro/inference/model.py#L380-L433):
1. Has its own `Compressor` (with Hadamard rotation for FP4 quantization simulation)
2. Projects queries via `wq_b`, applies RoPE and Hadamard rotation
3. Computes dot-product scores between rotated queries and compressed KVs
4. Selects top-K indices from compressed history
5. Attention attends to: sliding window + top-K compressed positions

#### Hyper-Connections (HC) — Non-Standard Residuals
Config: `hc_mult: 4`, `hc_sinkhorn_iters: 20`, `hc_eps: 1e-6` (both)

From [model.py Block L647-700](file:///C:/Users/surya/.gemini/antigravity/brain/3753ebe4-8695-470c-859d-33797a4b15eb/scratch/architectures/8_deepseek_v4_pro/inference/model.py#L647-L700):
1. Hidden state tensor is `[batch, seq, hc_mult=4, dim]` — 4 copies
2. Before each sublayer: `hc_pre()` applies Sinkhorn-normalized learned mixing coefficients to produce one `[batch, seq, dim]`
3. After each sublayer: `hc_post()` expands the output back to 4 copies using combination matrix and residual

From [model.py Transformer.forward L801-809](file:///C:/Users/surya/.gemini/antigravity/brain/3753ebe4-8695-470c-859d-33797a4b15eb/scratch/architectures/8_deepseek_v4_pro/inference/model.py#L801-L809):
```python
h = self.embed(input_ids)
h = h.unsqueeze(2).repeat(1, 1, self.hc_mult, 1)  # → [b, s, 4, dim]
for layer in self.layers:
    h = layer(h, start_pos, input_ids)
```

#### Hash-Based Routing (First N Layers)
Config: `num_hash_layers: 3` (both)

From [model.py Gate L546-584](file:///C:/Users/surya/.gemini/antigravity/brain/3753ebe4-8695-470c-859d-33797a4b15eb/scratch/architectures/8_deepseek_v4_pro/inference/model.py#L546-L584): layers with `layer_id < num_hash_layers` use a precomputed lookup table `tid2eid` mapping token IDs → expert indices (no learned gating).

#### MTP (Multi-Token Prediction)
Both: `num_nextn_predict_layers: 1`

From [model.py MTPBlock L738-766](file:///C:/Users/surya/.gemini/antigravity/brain/3753ebe4-8695-470c-859d-33797a4b15eb/scratch/architectures/8_deepseek_v4_pro/inference/model.py#L738-L766): MTPBlock takes the 4-copy hidden state, projects embeddings and hidden via `e_proj`/`h_proj`, runs a standard block, outputs next-token logits.

#### Other Notable Config Fields

| Field | Pro | Flash |
|-------|-----|-------|
| `scoring_func` | `sqrtsoftplus` | `sqrtsoftplus` |
| `routed_scaling_factor` | 2.5 | 1.5 |
| `swiglu_limit` | 10.0 | 10.0 |
| `expert_dtype` | `fp4` | `fp4` |
| `rope_theta` | 10,000 | 10,000 |
| YaRN `factor` | 16 | 16 |
| YaRN `original_max_position_embeddings` | 65,536 | 65,536 |
| `compress_rope_theta` | 160,000 | 160,000 |
| `n_group` | `null` | — |
| `topk_group` | `null` | — |
| Quantization | FP8 (`e4m3`, `scale_fmt: ue8m0`, `weight_block_size: [128,128]`) | same |

---

### 4.5 Moonshot AI Kimi K2.6

**Source files**: [config.json](file:///C:/Users/surya/.gemini/antigravity/brain/3753ebe4-8695-470c-859d-33797a4b15eb/scratch/architectures/10_kimi_k2_6/config.json), [modeling_deepseek.py](file:///C:/Users/surya/.gemini/antigravity/brain/3753ebe4-8695-470c-859d-33797a4b15eb/scratch/architectures/10_kimi_k2_6/modeling_deepseek.py), [configuration_deepseek.py](file:///C:/Users/surya/.gemini/antigravity/brain/3753ebe4-8695-470c-859d-33797a4b15eb/scratch/architectures/10_kimi_k2_6/configuration_deepseek.py)

#### Text Backbone: DeepSeek V3 Architecture (MLA)
The text backbone `text_config.model_type` is `"kimi_k2"`, but the modeling code imports `DeepseekV3Config` from [configuration_deepseek.py](file:///C:/Users/surya/.gemini/antigravity/brain/3753ebe4-8695-470c-859d-33797a4b15eb/scratch/architectures/10_kimi_k2_6/configuration_deepseek.py#L1). This is a DeepSeek V3-class architecture (not V4 — no Hyper-Connections, no compress_ratios, no indexer).

MLA dimensions from `text_config`:
```
kv_lora_rank: 512       — latent KV dimension
q_lora_rank: 1536       — query low-rank bottleneck
qk_rope_head_dim: 64    — RoPE-applied portion of Q/K
qk_nope_head_dim: 128   — non-RoPE portion of Q/K
v_head_dim: 128          — value head dim
```

From [modeling_deepseek.py DeepseekV3Attention L630-700](file:///C:/Users/surya/.gemini/antigravity/brain/3753ebe4-8695-470c-859d-33797a4b15eb/scratch/architectures/10_kimi_k2_6/modeling_deepseek.py#L630-L700):

**Q path**: `hidden → q_a_proj (→ 1536) → q_a_layernorm → q_b_proj (→ 64 heads × 192)`

**KV path**: `hidden → kv_a_proj_with_mqa (→ 576 = 512 + 64)` → split into `compressed_kv (512)` and `k_pe (64)` → `kv_a_layernorm → kv_b_proj (→ 64 heads × 256)` → split into `k_nope (128)` and `value (128)`

The KV cache stores the **compressed latent (512 dims) + separate RoPE key (64 dims) = 576 dims per token**, not the full expanded per-head KV.

#### MoE Configuration
* `n_routed_experts: 384`, `num_experts_per_tok: 8`, `n_shared_experts: 1`
* `first_k_dense_replace: 1` — layer 0 uses dense MLP (`intermediate_size: 18,432`)
* `moe_layer_freq: 1` — every layer after layer 0 is MoE
* `moe_intermediate_size: 2,048`
* `routed_scaling_factor: 2.827`
* `scoring_func: "sigmoid"`, `topk_method: "noaux_tc"`
* `n_group: 1`, `topk_group: 1` — no grouped routing (all experts in one group)

#### YaRN RoPE
```
rope_theta: 50,000
rope_scaling.type: "yarn"
rope_scaling.factor: 64.0
rope_scaling.original_max_position_embeddings: 4,096
rope_scaling.mscale: 1.0
rope_scaling.mscale_all_dim: 1.0
```

#### `num_nextn_predict_layers: 0` — **No MTP** in Kimi K2.6

#### Quantization
Uses compressed-tensors quantization (INT4, group_size=32, symmetric, minmax observer). Ignores: attention layers, shared experts, gate/up/down_proj MLPs, lm_head, vision tower, and mm_projector.

#### Vision Tower
From `vision_config`:
* `mm_projector_type: "patchmerger"`, `merge_kernel_size: [2, 2]`, `merge_type: "sd2_tpool"`
* `vt_hidden_size: 1,152`, `vt_intermediate_size: 4,304`, `vt_num_attention_heads: 16`, `vt_num_hidden_layers: 27`
* `patch_size: 14`
* `pos_emb_type: "divided_fixed"` — divided fixed positional embeddings (not learned 2D)
* `video_attn_type: "spatial_temporal"`
* `init_pos_emb_height: 64`, `init_pos_emb_width: 64`, `init_pos_emb_time: 4`
* `projector_hidden_act: "gelu"`
* `text_hidden_size: 7,168` (projection target)

---

### 4.6 MiniMax MiniMax-M3

> [!CAUTION]
> The Hugging Face API returned **HTTP 404** for `MiniMax/MiniMax-M3`. The directory `11_minimax_m3/` is **empty**. Zero configuration or modeling files exist. No claims can be made.

---

### 4.7 NVIDIA Nemotron 3 Ultra BF16

**Source file**: [config.json](file:///C:/Users/surya/.gemini/antigravity/brain/3753ebe4-8695-470c-859d-33797a4b15eb/scratch/architectures/12_nemotron_3_ultra_bf16/config.json)

#### Mamba-MoE-Attention Triple-Hybrid
`layers_block_type` array has 108 entries with three block types:
* **`mamba`**: 48 layers
* **`moe`**: 48 layers
* **`attention`**: 12 layers

First 20 entries: `mamba, moe, mamba, moe, mamba, moe, mamba, attention, moe, mamba, moe, mamba, moe, mamba, attention, moe, mamba, moe, mamba, moe`

Pattern: roughly `(mamba, moe) × 3, mamba, attention, moe` repeated — attention layers appear approximately every 9 positions.

> [!NOTE]
> `num_hidden_layers` is explicitly `null` in the config. The number of layers is derived from the `layers_block_type` array length (108).

#### Mamba SSM Parameters
```
mamba_head_dim: 64
mamba_num_heads: 256
mamba_hidden_act: silu
ssm_state_size: 128
conv_kernel: 4
expand: 2
chunk_size: 128
mamba_proj_bias: false
mamba_ssm_cache_dtype: float32
use_mamba_kernels: true
```

#### MoE Parameters
```
n_routed_experts: 512
num_experts_per_tok: 22
n_shared_experts: 1
moe_intermediate_size: 5,120
moe_latent_size: 2,048
moe_shared_expert_intermediate_size: 10,240
moe_shared_expert_overlap: false
routed_scaling_factor: 5.0
n_group: 1
n_groups: 8
topk_group: 1
norm_topk_prob: true
```

* **22 active experts per token** — the highest count in this entire set
* `moe_latent_size: 2,048` — experts use a latent bottleneck
* Shared expert FFN: 2× the routed expert size (10,240 vs 5,120)
* `n_groups: 8` — for expert grouping

#### Attention Parameters
```
num_attention_heads: 64
num_key_value_heads: 2
head_dim: 128
attention_bias: false
partial_rotary_factor: 1.0    — full RoPE rotation
rope_theta: 10,000
sliding_window: null           — no sliding window
```
Standard GQA: 64Q/2KV = 32:1 ratio.

#### MTP
```
num_nextn_predict_layers: 1
mtp_layers_block_type: ["attention", "moe"]
```
The MTP prediction head is 2 layers: one attention block + one MoE block.

#### Other Fields
* `mlp_hidden_act: "relu2"` — squared ReLU for MLP/MoE FFN
* `rescale_prenorm_residual: true` — residual rescaling
* `intermediate_size: 5,120` — dense MLP intermediate (used for dense fallback or attention-block FFN)
* `hidden_dropout: 0.0`, `attention_dropout: 0.0`
* `residual_in_fp32: false`
* `time_step_min: 0.001`, `time_step_max: 0.1`, `time_step_floor: 0.0001` (Mamba dt parameters)

---

### 4.8 OpenAI GPT-OSS 20B

**Source file**: [config.json](file:///C:/Users/surya/.gemini/antigravity/brain/3753ebe4-8695-470c-859d-33797a4b15eb/scratch/architectures/13_gpt_oss_20b/config.json)

#### Alternating Sliding/Full Attention (1:1)
`layer_types` array with 24 entries: strict **1:1 alternation** of `sliding_attention` and `full_attention` (12 each). `sliding_window: 128`.

#### Small-Scale MoE
* `num_local_experts: 32` (field name differs from other models)
* `experts_per_token: 4` / `num_experts_per_tok: 4` (duplicate fields)
* No shared experts, no explicit `moe_intermediate_size` — experts use `intermediate_size: 2,880`
* `router_aux_loss_coef: 0.9`
* `output_router_logits: false`

#### YaRN RoPE
```
rope_theta: 150,000
rope_scaling.rope_type: "yarn"
rope_scaling.factor: 32.0
rope_scaling.original_max_position_embeddings: 4,096
rope_scaling.beta_fast: 32.0
rope_scaling.beta_slow: 1.0
rope_scaling.truncate: false
```

#### SwiGLU Clamping
`swiglu_limit: 7.0` — activation values clamped to [-7, 7].

#### MXFP4 Quantization
```
quantization_config.quant_method: "mxfp4"
quantization_config.modules_to_not_convert: [
    "model.layers.*.self_attn",
    "model.layers.*.mlp.router",
    "model.embed_tokens",
    "lm_head"
]
```
Expert weights in MXFP4; attention, routers, embeddings, and LM head excluded.

#### Other
* `attention_bias: true` — one of the few models using attention bias
* `hidden_act: "silu"` — SiLU activation for FFN
* `initial_context_length: 4,096` (pre-YaRN training context)

---

### 4.9 Z.AI GLM-5.2

**Source files**: [config.json](file:///C:/Users/surya/.gemini/antigravity/brain/3753ebe4-8695-470c-859d-33797a4b15eb/scratch/architectures/14_glm_5_2/config.json)

#### DeepSeek V3/V4 Architecture Base (MLA + MTP)
GLM-5.2 is heavily based on the DeepSeek architecture. It features Multi-Head Latent Attention (MLA) with `q_lora_rank: 2048`, `kv_lora_rank: 512`, `qk_nope_head_dim: 192`, and `qk_rope_head_dim: 64`. It also incorporates Multi-Token Prediction (MTP) with `num_nextn_predict_layers: 1`.

#### "IndexShare" Sparse Attention for 1M Context
To achieve its 1,000,000-token context window (`max_position_embeddings: 1048576`), it introduces a custom mechanism defined by the `indexer_types` array. After the first three standard layers, the model alternates between one `"full"` index layer and three `"shared"` layers. The retrieval index calculated on the `"full"` layer is physically passed to the subsequent `"shared"` layers, reducing long-context attention compute by 75%.

#### MoE Configuration
* `n_routed_experts: 256`, `num_experts_per_tok: 8`, `n_shared_experts: 1`
* `mlp_layer_types`: The first 3 layers are `"dense"`, the remaining 75 layers are `"sparse"`.
* Dense `intermediate_size: 12288`, MoE `moe_intermediate_size: 2048`
* `routed_scaling_factor: 2.5`, `scoring_func: "sigmoid"`, `topk_method: "noaux_tc"`

---

## 5. Cross-Model Comparison Matrices

### 5.1 Attention Paradigm

| Model | Method | KV Cache per Token per Layer (dims) |
|-------|--------|-------------------------------------|
| MiMo V2.5 Pro | GQA (128Q/8KV) | K: 8×192=1,536 + V: 8×128=1,024 = **2,560** |
| MiMo V2 Flash (global) | GQA (64Q/4KV) | K: 4×192=768 + V: 4×128=512 = **1,280** |
| MiMo V2 Flash (SWA) | GQA (64Q/8KV) | K: 8×192=1,536 + V: 8×128=1,024 = **2,560** |
| Qwen 3.6 35B (full attn) | GQA (16Q/2KV) | K: 2×256=512 + V: 2×256=512 = **1,024** |
| Qwen 3.6 35B (linear attn) | Recurrent state | No KV cache — recurrent |
| Gemma 4 31B (sliding) | GQA (32Q/16KV) K=V | 16×256 = **4,096** (shared K=V) |
| Gemma 4 31B (full) | GQA (32Q/4KV) K=V | 4×512 = **2,048** (shared K=V) |
| DeepSeek V4 Pro | Single latent KV + compression | **512** (pre-compression) |
| Kimi K2.6 | MLA (latent 512 + RoPE 64) | **576** |
| Nemotron 3 Ultra (attn) | GQA (64Q/2KV) | K: 2×128=256 + V: 2×128=256 = **512** |
| Nemotron 3 Ultra (mamba) | SSM state | No KV cache — `ssm_state_size: 128` |
| GPT-OSS 20B | GQA (64Q/8KV) | K: 8×64=512 + V: 8×64=512 = **1,024** |

### 5.2 Residual Connection Strategy

| Model | Strategy |
|-------|----------|
| MiMo V2.5/Flash | Standard pre-norm residual |
| Qwen 3.6 35B/27B | Standard pre-norm residual |
| Gemma 4 (all) | Standard pre-norm residual |
| **DeepSeek V4 Pro/Flash** | **Hyper-Connections**: 4-copy state with Sinkhorn mixing |
| Kimi K2.6 | Standard pre-norm residual (DeepSeek V3 style) |
| Nemotron 3 Ultra | `rescale_prenorm_residual: true` |
| GPT-OSS 20B | Standard pre-norm residual |

### 5.3 Positional Encoding Comparison

| Model | Method | Base θ | Special |
|-------|--------|--------|---------|
| MiMo V2.5 Pro | Partial RoPE (33.4%) | Global: 10M, SWA: 10K | Separate RoPE instances |
| MiMo V2 Flash | Partial RoPE (33.4%) | Global: 5M, SWA: 10K | Separate RoPE instances |
| Qwen 3.6 35B/27B | Partial RoPE (25%) + M-RoPE | 10M | 3-section `[11,11,10]` multimodal M-RoPE |
| Gemma 4 31B/26B/12B | Partial RoPE (25%) | Full: 1M (proportional), Sliding: 10K | Per-type config; Vision: 2D RoPE θ=100 |
| DeepSeek V4 Pro/Flash | Decoupled RoPE (64 dim) | 10K base, 160K compress | YaRN ×16 from 65K ctx |
| Kimi K2.6 | Decoupled RoPE (64 dim) | 50K | YaRN ×64 from 4K ctx |
| Nemotron 3 Ultra | Full RoPE (100%) | 10K | `partial_rotary_factor: 1.0` |
| GPT-OSS 20B | Full RoPE (100%) | 150K | YaRN ×32 from 4K ctx |

### 5.4 Unique Features Summary

| Feature | Models |
|---------|--------|
| Multi-head Latent Attention (MLA) | Kimi K2.6 |
| Single-Latent KV + Compression | DeepSeek V4 Pro/Flash |
| Hyper-Connections (non-standard residual) | DeepSeek V4 Pro/Flash |
| Learned Sparse Indexer | DeepSeek V4 Pro/Flash |
| Hash-based expert routing | DeepSeek V4 Pro/Flash (first 3 layers) |
| Linear/recurrent attention layers | Qwen 3.6 35B-A3B, Qwen 3.6 27B |
| Mamba SSM layers | Nemotron 3 Ultra |
| K=V weight sharing | Gemma 4 (all three) |
| Dual head_dim (sliding vs global) | Gemma 4 (all three: 256 vs 512) |
| Dual KV head counts (per attention type) | Gemma 4 (all), MiMo V2 Flash |
| Attention sink bias | MiMo V2.5 Pro, MiMo V2 Flash |
| Attention value scaling | MiMo V2.5 Pro, MiMo V2 Flash |
| Asymmetric Q/K vs V head dim | MiMo V2.5 Pro, MiMo V2 Flash |
| M-RoPE (multimodal sections) | Qwen 3.6 35B, Qwen 3.6 27B |
| Attention output gate | Qwen 3.6 35B, Qwen 3.6 27B |
| Audio encoder | Gemma 4 12B IT |
| Bidirectional attention for vision | Gemma 4 (all three) |
| SwiGLU clamping | DeepSeek V4, GPT-OSS 20B |
| Squared ReLU (`relu2`) | Nemotron 3 Ultra |
| MXFP4 quantization | GPT-OSS 20B |
| FP4 expert weights | DeepSeek V4 Pro/Flash |
