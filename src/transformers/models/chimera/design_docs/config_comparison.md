# Architecture Config Comparison Table

| Model Name | Hugging Face Repo ID | Model Type | Vocab Size | Hidden Size | Layers | Heads (Q/KV) | Max Context | MoE | Unique Features / Notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Xiaomi MiMo-V2.5-Pro | `XiaomiMiMo/MiMo-V2.5-Pro` | `mimo_v2` | 152,576 | 6144 | 70 | 128/8 | 1,048,576 | Yes (8/384 active) | Alternating SWA / Global Attention |
| Xiaomi MiMo-V2-Flash | `XiaomiMiMo/MiMo-V2-Flash` | `mimo_v2_flash` | 152,576 | 4096 | 48 | 64/4 | 262,144 | Yes (8/256 active) | Alternating SWA / Global Attention |
| Alibaba Qwen 3.6 35B-A3B | `Qwen/Qwen3.6-35B-A3B` | `qwen3_5_moe` | 248,320 | 2048 | 40 | 16/2 | 262,144 | Yes (8/256 active) | Linear Attention (Gated DeltaNet), Multi-Token Prediction (MTP), Vision Encoder |
| Alibaba Qwen 3.6 27B | `Qwen/Qwen3.6-27B` | `qwen3_5` | 248,320 | 5120 | 64 | 24/4 | 262,144 | No | Linear Attention (Gated DeltaNet), Multi-Token Prediction (MTP), Vision Encoder |
| Google Gemma 4 31B IT | `google/gemma-4-31b-it` | `gemma4` | 262,144 | 5376 | 60 | 32/16 | 262,144 | Yes (None/None active) | Per-Layer Embeddings (PLE), Alternating Sliding Window Attention (SWA), Vision Encoder |
| Google Gemma 4 26B A4B IT | `google/gemma-4-26b-a4b-it` | `gemma4` | 262,144 | 2816 | 30 | 16/8 | 262,144 | Yes (8/128 active) | Per-Layer Embeddings (PLE), Alternating Sliding Window Attention (SWA), Vision Encoder |
| Google Gemma 4 12B IT | `google/gemma-4-12B-it` | `gemma4_unified` | 262,144 | 3840 | 48 | 16/8 | 262,144 | Yes (None/None active) | Per-Layer Embeddings (PLE), Alternating Sliding Window Attention (SWA), Vision Encoder, Audio Encoder |
| DeepSeek V4 Pro | `deepseek-ai/DeepSeek-V4-Pro` | `deepseek_v4` | 129,280 | 7168 | 61 | 128/1 | 1,048,576 | Yes (6/384 active) | Multi-Head Latent Attention (MLA), Compressed Sparse Attention (CSA), Manifold-Constrained Hyper-Connections (mHC), Multi-Token Prediction (MTP) |
| DeepSeek V4 Flash | `deepseek-ai/DeepSeek-V4-Flash` | `deepseek_v4` | 129,280 | 4096 | 43 | 64/1 | 1,048,576 | Yes (6/256 active) | Multi-Head Latent Attention (MLA), Compressed Sparse Attention (CSA), Manifold-Constrained Hyper-Connections (mHC), Multi-Token Prediction (MTP) |
| Moonshot AI Kimi K2.6 | `moonshotai/Kimi-K2.6` | `kimi_k25` | 163,840 | 7168 | 61 | 64/64 | 262,144 | Yes (8/384 active) | Multi-Head Latent Attention (MLA), Vision Encoder |
| MiniMax MiniMax-M3 | `MiniMax/MiniMax-M3` | *Failed* | - | - | - | - | - | - | Not Found / 404 (Weights unpublished/restricted) |
| NVIDIA Nemotron 3 Ultra BF16 | `nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16` | `nemotron_h` | 131,072 | 8192 | - | 64/2 | 262,144 | Yes (22/512 active) | Mamba-MoE-Attn Hybrid (48 Mamba, 48 MoE, 12 Attn layers), Multi-Token Prediction (MTP) |
| GPT-OSS 20B (OpenAI) | `openai/gpt-oss-20b` | `gpt_oss` | 201,088 | 2880 | 24 | 64/8 | 131,072 | Yes (4/32 active) | Alternating Sliding Window Attention (SWA) |
