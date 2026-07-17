#!/usr/bin/env python3
# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
"""Run plain-text or chat generation from a Chimera Hugging Face model directory."""

import argparse
from pathlib import Path

import torch

from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True, help="Path to Chimera HF model directory.")
    parser.add_argument("--prompt", default=None, help="Prompt text.")
    parser.add_argument("--prompt-file", type=Path, default=None, help="File containing prompt text.")
    parser.add_argument("--chat", action="store_true", help="Render the prompt as a Chimera user turn.")
    parser.add_argument("--system-prompt", default=None, help="Optional system message used with --chat.")
    parser.add_argument(
        "--show-special-tokens",
        action="store_true",
        help="Include tokens such as <end_of_turn> in non-streamed output.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--do-sample", action="store_true")
    parser.add_argument("--dtype", default="bfloat16", choices=("float32", "float16", "bfloat16"))
    parser.add_argument(
        "--device-map",
        default=None,
        help="Optional Transformers device map, such as 'auto'. Requires accelerate when set.",
    )
    parser.add_argument("--stream", action="store_true")
    return parser.parse_args()


def load_tokenizer(model_dir: Path):
    tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True)
    expected_ids = {"<BOS>": 0, "<EOS>": 1, "<start_of_turn>": 2, "<end_of_turn>": 3}
    for token, expected_id in expected_ids.items():
        token_id = tokenizer.convert_tokens_to_ids(token)
        if token_id != expected_id:
            raise ValueError(f"Expected {token!r} id {expected_id}, found {token_id}")
    if tokenizer.pad_token_id != tokenizer.eos_token_id:
        raise ValueError("Chimera requires <EOS> to be both the EOS and padding token")
    return tokenizer


def resolve_prompt(args: argparse.Namespace) -> str:
    if args.prompt_file is not None:
        return args.prompt_file.read_text()
    if args.prompt is not None:
        return args.prompt
    raise ValueError("Pass either --prompt or --prompt-file.")


def prepare_inputs(tokenizer, prompt: str, args: argparse.Namespace):
    if not args.chat:
        return tokenizer(prompt, add_special_tokens=False, return_tensors="pt")

    messages = []
    if args.system_prompt is not None:
        messages.append({"role": "system", "content": args.system_prompt})
    messages.append({"role": "user", "content": prompt})
    rendered = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return tokenizer(rendered, add_special_tokens=False, return_tensors="pt")


def main() -> None:
    args = parse_args()
    tokenizer = load_tokenizer(args.model)
    dtype = getattr(torch, args.dtype)
    model_kwargs = {"dtype": dtype}
    if args.device_map is not None:
        model_kwargs["device_map"] = args.device_map
    model = AutoModelForCausalLM.from_pretrained(args.model, **model_kwargs)
    prompt = resolve_prompt(args)
    inputs = prepare_inputs(tokenizer, prompt, args)
    inputs = {key: value.to(model.device) for key, value in inputs.items()}
    streamer = (
        TextStreamer(
            tokenizer,
            skip_prompt=True,
            skip_special_tokens=not args.show_special_tokens,
        )
        if args.stream
        else None
    )
    stop_ids = [tokenizer.eos_token_id]
    if args.chat:
        stop_ids.append(tokenizer.convert_tokens_to_ids("<end_of_turn>"))
    outputs = model.generate(
        **inputs,
        max_new_tokens=args.max_new_tokens,
        do_sample=args.do_sample,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        streamer=streamer,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=stop_ids,
    )
    if not args.stream:
        generated = outputs[0, inputs["input_ids"].shape[1] :]
        print(tokenizer.decode(generated, skip_special_tokens=not args.show_special_tokens))


if __name__ == "__main__":
    main()
