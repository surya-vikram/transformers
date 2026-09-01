#!/usr/bin/env python3
# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
"""Export a self-contained Chimera Hugging Face model directory."""

import argparse
import json
import shutil
import tarfile
import tempfile
import zipfile
from importlib import resources
from pathlib import Path

from transformers import ChimeraConfig, ChimeraForCausalLM, GenerationConfig, PreTrainedTokenizerFast
from transformers.models.chimera.configuration_chimera import CHIMERA_CONTEXT_PHASES


TOKENIZER_FILES = ("training_report.json",)
CHIMERA_CHAT_TEMPLATE = (
    "{% for message in messages %}"
    "{{ '<start_of_turn>' + message['role'] + '\\n' + message['content']|trim + '<end_of_turn>\\n' }}"
    "{% endfor %}"
    "{% if add_generation_prompt %}"
    "{{ '<start_of_turn>assistant\\n' }}"
    "{% endif %}"
)
CHIMERA_ADDITIONAL_SPECIAL_TOKENS = [
    "<start_of_turn>",
    "<end_of_turn>",
    "<DUMMY_2>",
    "<DUMMY_3>",
    "<DUMMY_4>",
    "<DUMMY_5>",
    "<DUMMY_6>",
    "<DUMMY_7>",
    "<DUMMY_8>",
    "<DUMMY_9>",
]
CHIMERA_TOKEN_IDS = {
    "<BOS>": 0,
    "<EOS>": 1,
    "<start_of_turn>": 2,
    "<end_of_turn>": 3,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="Destination Hugging Face model directory.")
    parser.add_argument(
        "--tokenizer-archive",
        type=Path,
        default=None,
        help="Tokenizer archive. Supports tar/tar.gz archives and zip files. Defaults to the bundled tokenizer.",
    )
    parser.add_argument(
        "--tokenizer-dir",
        type=Path,
        default=None,
        help="Directory containing tokenizer.json. Defaults to the bundled tokenizer.",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--no-weights", action="store_true", help="Write config/tokenizer files only.")
    mode.add_argument(
        "--meta-init",
        action="store_true",
        help="Instantiate the model on meta device and skip weights.",
    )
    mode.add_argument("--random-init", action="store_true", help="Instantiate random weights and save them.")
    parser.add_argument(
        "--load-with-bias",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use the checkpointed router bias during expert selection. The bias tensors are always saved.",
    )
    parser.add_argument(
        "--profile",
        choices=("full", "tiny"),
        default="full",
        help="Export the full canonical architecture or its reduced canonical smoke-test profile.",
    )
    parser.add_argument(
        "--context-phase",
        choices=tuple(CHIMERA_CONTEXT_PHASES),
        default="8k",
        help="Canonical YaRN context phase represented by the exported configuration.",
    )
    parser.add_argument("--dtype", default="bfloat16", choices=("float32", "float16", "bfloat16"))
    parser.add_argument("--max-shard-size", default="5GB", help="Shard size used when --random-init saves weights.")
    return parser.parse_args()


def find_tokenizer_root(root: Path) -> Path:
    matches = list(root.rglob("tokenizer.json"))
    if not matches:
        raise FileNotFoundError(f"No tokenizer.json found after extracting {root}")
    if len(matches) > 1:
        matches = sorted(matches)
    return matches[0].parent


def extract_tokenizer_archive(archive: Path, target: Path) -> Path:
    if not archive.exists():
        raise FileNotFoundError(f"Tokenizer archive not found: {archive}")

    try:
        with tarfile.open(archive) as tar:
            tar.extractall(target, filter="data")
            return find_tokenizer_root(target)
    except tarfile.TarError:
        pass

    try:
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(target)
            return find_tokenizer_root(target)
    except zipfile.BadZipFile as exc:
        raise ValueError(f"Unsupported tokenizer archive format: {archive}") from exc


def bundled_tokenizer_root() -> Path:
    script_root = Path(__file__).resolve().parents[1]
    for tokenizer_root in (script_root / "tokenizer", script_root):
        if (tokenizer_root / "tokenizer.json").exists():
            return tokenizer_root

    try:
        package_root = resources.files("transformers.models.chimera")
        tokenizer_resource = package_root.joinpath("tokenizer")
        tokenizer_path = Path(str(tokenizer_resource))
    except (ModuleNotFoundError, TypeError):
        tokenizer_path = Path()

    if (tokenizer_path / "tokenizer.json").exists():
        return tokenizer_path

    raise FileNotFoundError(
        "Bundled Chimera tokenizer not found. Pass --tokenizer-dir or --tokenizer-archive explicitly."
    )


def update_json_file(path: Path, updates: dict) -> None:
    data = {}
    if path.exists():
        data = json.loads(path.read_text())
    data.update(updates)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def copy_tokenizer_artifacts(
    tokenizer_root: Path, output: Path, model_max_length: int
) -> PreTrainedTokenizerFast:
    for name in TOKENIZER_FILES:
        src = tokenizer_root / name
        if src.exists():
            shutil.copy2(src, output / name)

    tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer_root)
    if len(tokenizer) != 50176:
        raise ValueError(f"Expected tokenizer length 50176, found {len(tokenizer)}")
    for token, expected_id in CHIMERA_TOKEN_IDS.items():
        actual_id = tokenizer.convert_tokens_to_ids(token)
        if actual_id != expected_id:
            raise ValueError(f"Expected {token!r} id {expected_id}, found {actual_id}")
        if tokenizer.encode(token, add_special_tokens=False) != [expected_id]:
            raise ValueError(f"Expected {token!r} to encode as one token with id {expected_id}")
    if tokenizer.unk_token is not None:
        raise ValueError(f"Chimera byte-level BPE must not define an unknown token: {tokenizer.unk_token!r}")
    configured_special_tokens = json.loads((tokenizer_root / "tokenizer_config.json").read_text()).get(
        "additional_special_tokens", []
    )
    if configured_special_tokens != CHIMERA_ADDITIONAL_SPECIAL_TOKENS:
        raise ValueError(f"Unexpected Chimera additional special tokens: {configured_special_tokens}")
    missing_special_tokens = [
        token for token in CHIMERA_ADDITIONAL_SPECIAL_TOKENS if token not in tokenizer.all_special_tokens
    ]
    if missing_special_tokens:
        raise ValueError(f"Tokens are not registered as special: {missing_special_tokens}")

    tokenizer.bos_token = "<BOS>"
    tokenizer.eos_token = "<EOS>"
    tokenizer.pad_token = "<EOS>"
    tokenizer.chat_template = CHIMERA_CHAT_TEMPLATE
    tokenizer.model_max_length = model_max_length

    tokenizer.save_pretrained(output)
    (output / "chat_template.jinja").write_text(CHIMERA_CHAT_TEMPLATE + "\n", encoding="utf-8")
    update_json_file(
        output / "tokenizer_config.json",
        {
            "bos_token": "<BOS>",
            "eos_token": "<EOS>",
            "model_max_length": model_max_length,
            "pad_token": "<EOS>",
            "additional_special_tokens": CHIMERA_ADDITIONAL_SPECIAL_TOKENS,
            "chat_template": CHIMERA_CHAT_TEMPLATE,
            "tokenizer_class": "PreTrainedTokenizerFast",
        },
    )
    update_json_file(
        output / "special_tokens_map.json",
        {
            "additional_special_tokens": CHIMERA_ADDITIONAL_SPECIAL_TOKENS,
            "bos_token": "<BOS>",
            "eos_token": "<EOS>",
            "pad_token": "<EOS>",
        },
    )
    return tokenizer


def build_config(
    *, load_with_bias: bool = True, profile: str = "full", context_phase: str = "8k"
) -> ChimeraConfig:
    profile_fields = {
        "full": {
            "hidden_size": 2048,
            "intermediate_size": 8192,
            "moe_intermediate_size": 2048,
            "num_hidden_layers": 25,
            "num_attention_heads": 16,
            "num_key_value_heads": 2,
            "head_dim": 256,
            "first_k_dense_replace": 2,
            "n_routed_experts": 32,
            "num_experts_per_tok": 4,
        },
        "tiny": {
            "hidden_size": 512,
            "intermediate_size": 2048,
            "moe_intermediate_size": 256,
            "num_hidden_layers": 8,
            "num_attention_heads": 8,
            "num_key_value_heads": 2,
            "head_dim": 64,
            "first_k_dense_replace": 2,
            "n_routed_experts": 8,
            "num_experts_per_tok": 2,
        },
    }[profile]
    context_geometry = CHIMERA_CONTEXT_PHASES[context_phase]
    config = ChimeraConfig(
        vocab_size=50176,
        bos_token_id=0,
        eos_token_id=1,
        pad_token_id=1,
        tie_word_embeddings=False,
        last_k_dense_replace=0,
        n_shared_experts=0,
        shared_expert_intermediate_size=0,
        context_phase=context_phase,
        position_embedding_type="yarn",
        max_position_embeddings=context_geometry["max_position_embeddings"],
        original_max_position_embeddings=8192,
        rope_theta=10000000.0,
        rope_scaling={
            "type": "yarn",
            "factor": context_geometry["factor"],
            "beta_fast": 32.0,
            "beta_slow": 1.0,
            "mscale": 1.0,
            "mscale_all_dim": 0.0,
            "original_max_position_embeddings": 8192,
            "truncate": False,
        },
        qk_layernorm=True,
        rms_norm_eps=1e-5,
        scoring_func="sigmoid",
        topk_method="noaux_tc",
        norm_topk_prob=True,
        router_aux_loss_coef=0.0,
        router_z_loss_coef=0.001,
        router_bias_update_rate=0.0,
        router_load_balancing_type="quantile_balancing",
        moe_qb_num_bins=1000,
        moe_qb_ema_decay=0.0,
        routed_scaling_factor=2.5,
        n_group=1,
        topk_group=1,
        load_with_bias=load_with_bias,
        **profile_fields,
    )
    config.architectures = ["ChimeraForCausalLM"]
    return config


def write_generation_config(output: Path) -> None:
    generation_config = GenerationConfig(
        bos_token_id=0,
        eos_token_id=1,
        pad_token_id=1,
        max_new_tokens=256,
        do_sample=False,
        temperature=1.0,
        top_p=1.0,
    )
    generation_config.save_pretrained(output)


def write_readme(output: Path) -> None:
    readme = """# Chimera 10B

Chimera is a decoder-only sparse MoE language model configuration.

This export keeps the tokenizer vocabulary size fixed while assigning reserved token IDs to chat markers:

- `vocab_size`: 50176
- `bos_token`: `<BOS>` / id 0
- `eos_token`: `<EOS>` / id 1
- `pad_token`: `<EOS>` / id 1
- `additional_special_tokens`: `<start_of_turn>` / id 2, `<end_of_turn>` / id 3
- `reserved tokens`: `<DUMMY_2>` through `<DUMMY_9>` / ids 4 through 11

`<BOS>` is retained for compatibility but is not inserted by the chat template. `<EOS>` remains the
pretraining document separator, generation EOS, and padding token. Chat inference may additionally stop
on `<end_of_turn>`.

Router expert-bias tensors are always part of the checkpoint. Set `load_with_bias=true` for canonical
parity or `load_with_bias=false` to bypass the frozen bias during expert selection without changing weights.

The chat template is a minimal non-reasoning, non-tool-calling turn format:

```text
<start_of_turn>user
Hello<end_of_turn>
<start_of_turn>assistant
Hi<end_of_turn>
```
"""
    (output / "README.md").write_text(readme)


def copy_scripts(output: Path) -> None:
    scripts_dir = output / "scripts"
    scripts_dir.mkdir(exist_ok=True)
    script_root = Path(__file__).resolve().parent
    for name in ("export_to_hf.py", "infer.py", "replace_tokenizer_tokens.py"):
        src = script_root / name
        if src.exists():
            shutil.copy2(src, scripts_dir / name)


def count_parameters(model: ChimeraForCausalLM) -> int:
    return sum(param.numel() for param in model.parameters())


def main() -> None:
    args = parse_args()
    output = args.output
    output.mkdir(parents=True, exist_ok=True)

    config = build_config(
        load_with_bias=args.load_with_bias,
        profile=args.profile,
        context_phase=args.context_phase,
    )
    config.save_pretrained(output)
    write_generation_config(output)
    write_readme(output)

    with tempfile.TemporaryDirectory() as tmpdir:
        if args.tokenizer_dir is not None:
            tokenizer_root = args.tokenizer_dir
        elif args.tokenizer_archive is not None:
            tokenizer_root = extract_tokenizer_archive(args.tokenizer_archive, Path(tmpdir))
        else:
            tokenizer_root = bundled_tokenizer_root()
        tokenizer = copy_tokenizer_artifacts(
            tokenizer_root, output, config.max_position_embeddings
        )

    model_parameter_count = None
    if args.meta_init:
        import torch

        with torch.device("meta"):
            model = ChimeraForCausalLM(config)
        model_parameter_count = count_parameters(model)
    elif args.random_init:
        import torch

        dtype = getattr(torch, args.dtype)
        model = ChimeraForCausalLM(config).to(dtype=dtype)
        model_parameter_count = count_parameters(model)
        model.save_pretrained(output, safe_serialization=True, max_shard_size=args.max_shard_size)

    report_path = output / "training_report.json"
    report = {}
    if report_path.exists():
        report = json.loads(report_path.read_text())
    report.update(
        {
            "chimera_export": {
                "mode": "random-init" if args.random_init else "meta-init" if args.meta_init else "no-weights",
                "model_parameter_count": model_parameter_count,
                "tokenizer_length": len(tokenizer),
            }
        }
    )
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")

    copy_scripts(output)
    print(f"Wrote Chimera HF directory to {output}")


if __name__ == "__main__":
    main()
