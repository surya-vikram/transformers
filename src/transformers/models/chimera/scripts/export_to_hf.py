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


TOKENIZER_FILES = ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "training_report.json")


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
    tokenizer_root = Path(__file__).resolve().parents[1] / "tokenizer"
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


def copy_tokenizer_artifacts(tokenizer_root: Path, output: Path) -> PreTrainedTokenizerFast:
    for name in TOKENIZER_FILES:
        src = tokenizer_root / name
        if src.exists():
            shutil.copy2(src, output / name)

    tokenizer = PreTrainedTokenizerFast(
        tokenizer_file=str(output / "tokenizer.json"),
        bos_token="<BOS>",
        eos_token="<EOS>",
        pad_token="<EOS>",
        model_max_length=32768,
    )
    if len(tokenizer) != 50176:
        raise ValueError(f"Expected tokenizer length 50176, found {len(tokenizer)}")

    update_json_file(
        output / "tokenizer_config.json",
        {
            "bos_token": "<BOS>",
            "eos_token": "<EOS>",
            "model_max_length": 32768,
            "pad_token": "<EOS>",
            "tokenizer_class": "PreTrainedTokenizerFast",
        },
    )
    update_json_file(
        output / "special_tokens_map.json",
        {"bos_token": "<BOS>", "eos_token": "<EOS>", "pad_token": "<EOS>"},
    )
    tokenizer.save_pretrained(output)
    return tokenizer


def build_config() -> ChimeraConfig:
    return ChimeraConfig(
        vocab_size=50176,
        bos_token_id=0,
        eos_token_id=1,
        pad_token_id=1,
        tie_word_embeddings=False,
        hidden_size=2048,
        num_hidden_layers=28,
        num_attention_heads=16,
        num_key_value_heads=2,
        head_dim=256,
        first_k_dense_replace=1,
        last_k_dense_replace=1,
        intermediate_size=8192,
        n_routed_experts=96,
        num_experts_per_tok=8,
        n_shared_experts=1,
        moe_intermediate_size=704,
        shared_expert_intermediate_size=704,
        max_position_embeddings=32768,
        original_max_position_embeddings=8192,
        rope_theta=10000000.0,
        rope_scaling={"type": "yarn", "factor": 4.0, "original_max_position_embeddings": 8192},
        scoring_func="sigmoid",
        topk_method="noaux_tc",
        norm_topk_prob=True,
        router_aux_loss_coef=0.001,
        routed_scaling_factor=1.0,
        n_group=1,
        topk_group=1,
    )


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
    readme = """# Chimera 12B

Chimera is a decoder-only sparse MoE language model configuration.

This export keeps the base tokenizer vocabulary unchanged:

- `vocab_size`: 50176
- `bos_token`: `<BOS>` / id 0
- `eos_token`: `<EOS>` / id 1
- `pad_token`: `<EOS>` / id 1

The model is intended as a pretraining/base model. No chat, role, reasoning, or tool-use tokens are added.
"""
    (output / "README.md").write_text(readme)


def copy_scripts(output: Path) -> None:
    scripts_dir = output / "scripts"
    scripts_dir.mkdir(exist_ok=True)
    script_root = Path(__file__).resolve().parent
    for name in ("export_to_hf.py", "infer.py"):
        src = script_root / name
        if src.exists():
            shutil.copy2(src, scripts_dir / name)


def count_parameters(model: ChimeraForCausalLM) -> int:
    return sum(param.numel() for param in model.parameters())


def main() -> None:
    args = parse_args()
    output = args.output
    output.mkdir(parents=True, exist_ok=True)

    config = build_config()
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
        tokenizer = copy_tokenizer_artifacts(tokenizer_root, output)

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
