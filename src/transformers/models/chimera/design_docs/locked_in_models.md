# Locked-In Model Architectures for Analysis

This document records the locked-in model architectures selected from the Artificial Analysis open weights leaderboard, mapped to their verified Hugging Face repository IDs.

---

## Final Selected Models & Hugging Face Repo IDs

### 1. Xiaomi MiMo-V2.5-Pro
*   **HF Repo ID:** `XiaomiMiMo/MiMo-V2.5-Pro`
*   **Architecture Type:** Sparse MoE (42B active / 1.02T total parameters)
*   **Key Feature:** Alternating Sliding Window Attention (SWA) and global attention backbone to reduce KV cache size (1M context length) without MTP layers directly built into its target weights.

### 2. Xiaomi MiMo-V2-Flash
*   **HF Repo ID:** `XiaomiMiMo/MiMo-V2-Flash`
*   **Architecture Type:** Speculative MoE model (8B active / 309B total parameters)
*   **Key Feature:** Houses the custom **DFlash MTP head** (featuring SWA + MTP attention layers) to speculate 8-to-16 token blocks in parallel in a single forward pass, utilizing projected target hidden state KV-cache injection.

### 3. Alibaba Qwen 3.6 35B-A3B
*   **HF Repo ID:** `Qwen/Qwen3.6-35B-A3B`
*   **Architecture Type:** Hybrid MoE (3B active / 35B total parameters)
*   **Key Feature:** Interleaves **Gated DeltaNet (linear attention)** with standard causal attention in a 3:1 ratio, scaling context natively to 262k with a minimal KV cache memory footprint.

### 4. Alibaba Qwen 3.6 27B
*   **HF Repo ID:** `Qwen/Qwen3.6-27B` (or `Qwen/Qwen3.6-27B-Instruct`)
*   **Architecture Type:** Dense Causal LLM (27B parameters)
*   **Key Feature:** Flagship dense causal baseline model from the Qwen series.

### 5. Google Gemma 4 31B (IT)
*   **HF Repo ID:** `google/gemma-4-31b-it`
*   **Architecture Type:** Dense Reasoning Causal LLM (31B parameters)
*   **Key Feature:** Google's flagship dense reasoning model utilizing **Per-Layer Embeddings (PLE)** to stabilize representation spaces at depth.

### 6. Google Gemma 4 26B A4B (IT)
*   **HF Repo ID:** `google/gemma-4-26b-a4b-it`
*   **Architecture Type:** Sparse MoE Reasoning Model (3.8B active / 25.2B total parameters)
*   **Key Feature:** High-reasoning sparse MoE from the Gemma 4 unified multimodal series.

### 7. DeepSeek DeepSeek V4 Pro
*   **HF Repo ID:** `deepseek-ai/DeepSeek-V4-Pro`
*   **Architecture Type:** Sparse MoE (37B active / 671B total parameters)
*   **Key Feature:** Utilizes **Multi-Head Latent Attention (MLA)** to compress key and value projections into a low-rank latent vector, minimizing KV cache memory footprint.

### 8. DeepSeek V4 Flash
*   **HF Repo ID:** `deepseek-ai/DeepSeek-V4-Flash` (or `deepseek-ai/DeepSeek-V4-Flash-Instruct`)
*   **Architecture Type:** High-efficiency MoE (13B active / 284B total parameters)
*   **Key Feature:** Uses a **Hybrid Attention Architecture** combining **Compressed Sparse Attention (CSA)** and **Heavily Compressed Attention (HCA)**, along with **Manifold-Constrained Hyper-Connections (mHC)** to stabilize signal propagation.

### 9. Moonshot AI Kimi K2.6
*   **HF Repo ID:** `moonshotai/Kimi-K2.6`
*   **Architecture Type:** Sparse MoE (32B active / 1T total parameters)
*   **Key Feature:** Optimized specifically for multi-step agentic planning and long-context processing without logic degradation.

### 10. MiniMax MiniMax-M3
*   **HF Repo ID:** `MiniMax/MiniMax-M3` (Canonical reservation ID)
*   **Architecture Type:** Unified Omnimodal Byte Transformer
*   **Key Feature:** Processes raw projected byte/tile coordinates of audio, video, and text directly in unified transformer layers, omitting separate modality encoders.

### 11. NVIDIA Nemotron 3 Ultra (BF16)
*   **HF Repo ID:** `nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16`
*   **Architecture Type:** Sparse MoE (55B active / 550B total parameters)
*   **Key Feature:** High-capacity sparse MoE optimized for safety-constrained expert routing.

### 12. GPT-OSS 20B (OpenAI)
*   **HF Repo ID:** `openai/gpt-oss-20b`
*   **Architecture Type:** Dense Causal Baseline LLM (20B parameters)
*   **Key Feature:** Open-source baseline template model.
