#!/usr/bin/env python3
"""Replace fixed-vocab Chimera tokenizer entries with special tokens.

Edit REPLACEMENTS to add future fixed-id token swaps. The script keeps the
tokenizer length unchanged by replacing existing vocab entries instead of
appending new tokens.
"""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from pathlib import Path

from transformers import PreTrainedTokenizerFast


EXPECTED_VOCAB_SIZE = 50176

REPLACEMENTS = {
    "<DUMMY_0>": "<start_of_turn>",
    "<DUMMY_1>": "<end_of_turn>",
}

CHAT_TEMPLATE = (
    "{% for message in messages %}"
    "{{ '<start_of_turn>' + message['role'] + '\\n' + message['content']|trim + '<end_of_turn>\\n' }}"
    "{% endfor %}"
    "{% if add_generation_prompt %}"
    "{{ '<start_of_turn>assistant\\n' }}"
    "{% endif %}"
)

EXPECTED_TOKEN_IDS = {
    "<BOS>": 0,
    "<EOS>": 1,
    "<start_of_turn>": 2,
    "<end_of_turn>": 3,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    script_root = Path(__file__).resolve().parents[1]
    bundled_tokenizer = script_root / "tokenizer"
    default_tokenizer_dir = bundled_tokenizer if (bundled_tokenizer / "tokenizer.json").exists() else script_root
    parser.add_argument(
        "--tokenizer-dir",
        type=Path,
        default=default_tokenizer_dir,
        help="Directory containing Chimera tokenizer.json.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate the replacement on a temporary copy without modifying files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Write a finalized copy here instead of modifying --tokenizer-dir in place.",
    )
    return parser.parse_args()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def merge_output(merge: list[str] | str) -> str:
    left, right = merge if isinstance(merge, list) else merge.split()
    return left + right


def replace_tokenizer_json(tokenizer_json: Path) -> dict[str, int]:
    data = read_json(tokenizer_json)
    vocab = data["model"]["vocab"]
    merges = data["model"].get("merges", [])
    replacement_ids = {}
    old_tokens_to_remove = set()

    for old_token, new_token in REPLACEMENTS.items():
        if old_token in vocab:
            if new_token in vocab:
                raise ValueError(f"Both old token {old_token!r} and new token {new_token!r} exist in tokenizer vocab")
            token_id = vocab.pop(old_token)
            vocab[new_token] = token_id
            old_tokens_to_remove.add(old_token)
        elif new_token in vocab:
            token_id = vocab[new_token]
        else:
            raise ValueError(f"Old token {old_token!r} is not in tokenizer vocab")
        replacement_ids[new_token] = token_id

    kept_merges = []
    for merge in merges:
        if merge_output(merge) in old_tokens_to_remove:
            continue
        else:
            kept_merges.append(merge)
    data["model"]["merges"] = kept_merges

    added_tokens = []
    added_contents = set()
    for token in data.get("added_tokens", []):
        token = token.copy()
        content = REPLACEMENTS.get(token.get("content"), token.get("content"))
        token["content"] = content
        if content in replacement_ids:
            token.update({"id": replacement_ids[content], "normalized": False, "special": True})
        if content not in added_contents:
            added_tokens.append(token)
            added_contents.add(content)

    for token, token_id in sorted(replacement_ids.items(), key=lambda item: item[1]):
        if token in added_contents:
            continue
        added_tokens.append(
            {
                "id": token_id,
                "content": token,
                "single_word": False,
                "lstrip": False,
                "rstrip": False,
                "normalized": False,
                "special": True,
            }
        )
    data["added_tokens"] = sorted(added_tokens, key=lambda token: token["id"])

    if len(vocab) != EXPECTED_VOCAB_SIZE:
        raise ValueError(f"Expected model vocab size {EXPECTED_VOCAB_SIZE}, found {len(vocab)}")

    write_json(tokenizer_json, data)
    return replacement_ids


def finalized_additional_special_tokens(config: dict) -> list[str]:
    tokens = [REPLACEMENTS.get(token, token) for token in config.get("additional_special_tokens", [])]
    for token in REPLACEMENTS.values():
        if token not in tokens:
            tokens.append(token)
    return tokens


def update_tokenizer_artifacts(tokenizer_dir: Path, replacement_ids: dict[str, int]) -> list[str]:
    tokenizer_config = tokenizer_dir / "tokenizer_config.json"
    special_tokens_map = tokenizer_dir / "special_tokens_map.json"

    if tokenizer_config.exists():
        config = read_json(tokenizer_config)
    else:
        config = {}
    additional_special_tokens = finalized_additional_special_tokens(config)
    config.update(
        {
            "additional_special_tokens": additional_special_tokens,
            "chat_template": CHAT_TEMPLATE,
            "model_max_length": 32768,
            "tokenizer_class": "PreTrainedTokenizerFast",
        }
    )
    # `from_pretrained` resolves tokenizer.json itself. A relative tokenizer_file
    # entry is passed through as a stale constructor path by recent Transformers.
    config.pop("tokenizer_file", None)
    config.pop("unk_token", None)
    write_json(tokenizer_config, config)

    if special_tokens_map.exists():
        special_map = read_json(special_tokens_map)
    else:
        special_map = {}
    special_map["additional_special_tokens"] = additional_special_tokens
    special_map.pop("unk_token", None)
    write_json(special_tokens_map, special_map)

    (tokenizer_dir / "chat_template.jinja").write_text(CHAT_TEMPLATE + "\n", encoding="utf-8")

    report_path = tokenizer_dir / "training_report.json"
    if report_path.exists():
        report = read_json(report_path)
        dummy_tokens = [token for token in additional_special_tokens if token.startswith("<DUMMY_")]
        vocab = read_json(tokenizer_dir / "tokenizer.json")["model"]["vocab"]
        dummy_token_ids = {token: vocab[token] for token in dummy_tokens}
        current_special_token_ids = {
            "bos_token": EXPECTED_TOKEN_IDS["<BOS>"],
            "eos_token": EXPECTED_TOKEN_IDS["<EOS>"],
            "pad_token": EXPECTED_TOKEN_IDS["<EOS>"],
            "unk_token": None,
            "chat_tokens": replacement_ids,
            "dummy_tokens": dummy_token_ids,
        }
        report.update(
            {
                "special_tokens": ["<BOS>", "<EOS>", *additional_special_tokens],
                "dummy_token_count": len(dummy_tokens),
                "dummy_tokens": dummy_tokens,
                "dummy_token_ids": dummy_token_ids,
                "chat_token_ids": replacement_ids,
                "special_token_ids": current_special_token_ids,
            }
        )
        if "tokenizer_diagnostics" in report:
            report["tokenizer_diagnostics"]["special_token_ids"] = current_special_token_ids
        write_json(report_path, report)

    return additional_special_tokens


def validate_tokenizer(
    tokenizer_dir: Path, replacement_ids: dict[str, int], additional_special_tokens: list[str]
) -> None:
    tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer_dir)
    if len(tokenizer) != EXPECTED_VOCAB_SIZE:
        raise ValueError(f"Expected tokenizer length {EXPECTED_VOCAB_SIZE}, found {len(tokenizer)}")

    for token, expected_id in EXPECTED_TOKEN_IDS.items():
        actual_id = tokenizer.convert_tokens_to_ids(token)
        if actual_id != expected_id:
            raise ValueError(f"Expected {token!r} id {expected_id}, found {actual_id}")
        encoded = tokenizer.encode(token, add_special_tokens=False)
        if encoded != [expected_id]:
            raise ValueError(f"Expected {token!r} to encode as [{expected_id}], found {encoded}")

    if tokenizer.unk_token is not None:
        raise ValueError(f"Expected no unknown token, found {tokenizer.unk_token!r}")
    configured_special_tokens = read_json(tokenizer_dir / "tokenizer_config.json").get(
        "additional_special_tokens", []
    )
    if configured_special_tokens != additional_special_tokens:
        raise ValueError(
            "Unexpected additional special token metadata: "
            f"expected {additional_special_tokens}, found {configured_special_tokens}"
        )
    missing_special_tokens = [
        token for token in additional_special_tokens if token not in tokenizer.all_special_tokens
    ]
    if missing_special_tokens:
        raise ValueError(f"Tokens are not registered as special: {missing_special_tokens}")

    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": "Hello"}, {"role": "assistant", "content": "Hi"}],
        tokenize=False,
        add_generation_prompt=False,
    )
    expected = "<start_of_turn>user\nHello<end_of_turn>\n<start_of_turn>assistant\nHi<end_of_turn>\n"
    if rendered != expected:
        raise ValueError(f"Unexpected chat template rendering: {rendered!r}")

    tokenized = tokenizer.apply_chat_template(
        [{"role": "user", "content": "Hello"}], tokenize=True, add_generation_prompt=True
    )
    if isinstance(tokenized, dict) or hasattr(tokenized, "input_ids"):
        tokenized = tokenized["input_ids"]
    if tokenized and isinstance(tokenized[0], list):
        tokenized = tokenized[0]
    if tokenized[0] != EXPECTED_TOKEN_IDS["<start_of_turn>"]:
        raise ValueError(f"Chat template unexpectedly prefixed tokens: {tokenized}")
    if tokenized.count(EXPECTED_TOKEN_IDS["<start_of_turn>"]) != 2:
        raise ValueError(f"Expected two start-of-turn markers, found tokens: {tokenized}")
    if tokenized.count(EXPECTED_TOKEN_IDS["<end_of_turn>"]) != 1:
        raise ValueError(f"Expected one end-of-turn marker, found tokens: {tokenized}")

    vocab = tokenizer.get_vocab()
    for old_token in REPLACEMENTS:
        if old_token in vocab:
            raise ValueError(f"Replaced token remains in tokenizer vocab: {old_token!r}")


def main() -> None:
    args = parse_args()
    tokenizer_dir = args.tokenizer_dir.resolve()
    if not (tokenizer_dir / "tokenizer.json").exists():
        raise FileNotFoundError(f"Missing tokenizer.json in {tokenizer_dir}")

    if args.dry_run and args.output_dir is not None:
        raise ValueError("--dry-run and --output-dir cannot be used together")

    if args.dry_run:
        with tempfile.TemporaryDirectory() as tmpdir:
            dry_dir = Path(tmpdir) / "tokenizer"
            shutil.copytree(tokenizer_dir, dry_dir)
            replacement_ids = replace_tokenizer_json(dry_dir / "tokenizer.json")
            additional_special_tokens = update_tokenizer_artifacts(dry_dir, replacement_ids)
            validate_tokenizer(dry_dir, replacement_ids, additional_special_tokens)
        print("Dry run passed.")
        return

    output_dir = args.output_dir.resolve() if args.output_dir is not None else tokenizer_dir
    if output_dir != tokenizer_dir:
        if output_dir.exists():
            raise FileExistsError(f"Output directory already exists: {output_dir}")
        shutil.copytree(tokenizer_dir, output_dir)

    replacement_ids = replace_tokenizer_json(output_dir / "tokenizer.json")
    additional_special_tokens = update_tokenizer_artifacts(output_dir, replacement_ids)
    validate_tokenizer(output_dir, replacement_ids, additional_special_tokens)
    print(f"Updated {output_dir}")
    for token, token_id in sorted(replacement_ids.items(), key=lambda item: item[1]):
        print(f"{token_id}: {token}")


if __name__ == "__main__":
    main()
