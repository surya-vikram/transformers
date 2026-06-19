#!/usr/bin/env python3
# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
"""Run plain-text generation from a Chimera Hugging Face model directory."""

import argparse
from pathlib import Path

import torch

from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerFast, TextStreamer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True, help="Path to Chimera HF model directory.")
    parser.add_argument("--prompt", default=None, help="Prompt text.")
    parser.add_argument("--prompt-file", type=Path, default=None, help="File containing prompt text.")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--do-sample", action="store_true")
    parser.add_argument("--dtype", default="bfloat16", choices=("float32", "float16", "bfloat16"))
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--stream", action="store_true")
    return parser.parse_args()


def load_tokenizer(model_dir: Path):
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True)
    except Exception:
        tokenizer = PreTrainedTokenizerFast(
            tokenizer_file=str(model_dir / "tokenizer.json"),
            bos_token="<BOS>",
            eos_token="<EOS>",
            pad_token="<EOS>",
            model_max_length=32768,
        )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def resolve_prompt(args: argparse.Namespace) -> str:
    if args.prompt_file is not None:
        return args.prompt_file.read_text()
    if args.prompt is not None:
        return args.prompt
    raise ValueError("Pass either --prompt or --prompt-file.")


def main() -> None:
    args = parse_args()
    tokenizer = load_tokenizer(args.model)
    dtype = getattr(torch, args.dtype)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype, device_map=args.device_map)
    prompt = resolve_prompt(args)
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {key: value.to(model.device) for key, value in inputs.items()}
    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True) if args.stream else None
    outputs = model.generate(
        **inputs,
        max_new_tokens=args.max_new_tokens,
        do_sample=args.do_sample,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        streamer=streamer,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    if not args.stream:
        print(tokenizer.decode(outputs[0], skip_special_tokens=True))


if __name__ == "__main__":
    main()
