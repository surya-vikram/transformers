# Chimera Architecture Implementation Task List

This document outlines the requirements and tasks for implementing the Chimera-10B architecture using true SOTA components rather than older library baselines.

## Background

The initial attempt at building `modeling_chimera.py` used older baseline components (`LlamaAttention` and `DeepseekV2MoE`) from the Hugging Face `transformers` repository. 

I have left my initial implementation in this experimental branch (`exp`) along with all the SOTA architecture research files. Your objective is to review this base critically, create a new branch from here, and implement the architecture in a much better way from scratch, utilizing the true bleeding-edge mechanics from the **actual SOTA models** we analyzed.

All necessary research reports and raw reference code for the SOTA models (DeepSeek-V4, Qwen-3.6, MiMo-V2, Kimi-K2.6, etc.) have been copied into this directory for your convenience:
- **`design_docs/`**: Contains the full architectural comparison reports, benchmark analysis, and config cross-comparisons.
- **`reference_code/`**: Contains the raw downloaded PyTorch files (e.g., `modeling_mimo_v2.py`, DeepSeek-V4's `inference/model.py`, `modeling_kimi_k25.py`, etc.)

## Tasks

- [ ] **1. Review the Architecture Reports**
  - Read `design_docs/architecture_comparison_report.md` to familiarize yourself with the 4 cutting-edge paradigms (DeepSeek-V4 Latent KV Compression, MiMo-V2 Sliding-Window Alternation, Qwen-3.6 Linear Attention, Kimi K2.6 MLA).
  
- [ ] **2. Select and Define the Target SOTA Paradigm**
  - Coordinate with the team to finalize which of the bleeding-edge paradigms from the `reference_code/` directory will serve as the foundation for Chimera. 
  
- [ ] **3. Write `configuration_chimera.py` from Scratch**
  - Define `ChimeraConfig` to include the specialized hyperparameter fields required by your chosen paradigm (e.g., `compress_ratios` and `hc_mult` for DeepSeek-V4, or `hybrid_layer_pattern` for MiMo-V2).
  - Retain the baseline 10B target constraints (1M RoPE theta, no tie weights).

- [ ] **4. Write `modeling_chimera.py` from Scratch**
  - Build the PyTorch math from the ground up by extracting and integrating the math from the corresponding files in `reference_code/`.
  - Ensure the file is a **100% self-contained monolith** with zero external architectural imports (like importing `LlamaAttention` or `DeepseekMoE`), protecting it from future Hugging Face library updates.

- [ ] **5. Registration and Verification**
  - Create `__init__.py` to export the config and model classes.
  - Re-register the model in `transformers/models/auto/auto_mappings.py` and `transformers/models/auto/modeling_auto.py`.
  - Run AST syntax checks on your finalized `modeling_chimera.py`.
  - Run a dummy forward-pass instantiation test to confirm shape compatibility.
