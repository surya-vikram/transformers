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
    "ĠLena": "<start_of_turn>",
    "Ġintercepted": "<end_of_turn>",
}

CHAT_TEMPLATE = (
    "{% for message in messages %}"
    "{{ '<start_of_turn>' + message['role'] + '\\n' + message['content']|trim + '<end_of_turn>\\n' }}"
    "{% endfor %}"
    "{% if add_generation_prompt %}"
    "{{ '<start_of_turn>assistant\\n' }}"
    "{% endif %}"
)

SPECIAL_TOKENS = list(REPLACEMENTS.values())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tokenizer-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "tokenizer",
        help="Directory containing Chimera tokenizer.json.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate the replacement on a temporary copy without modifying files.",
    )
    return parser.parse_args()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")


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

    removed_merges = []
    kept_merges = []
    for merge in merges:
        if merge_output(merge) in old_tokens_to_remove:
            removed_merges.append(merge)
        else:
            kept_merges.append(merge)
    data["model"]["merges"] = kept_merges

    if len(removed_merges) != len(old_tokens_to_remove):
        raise ValueError(
            f"Expected to remove {len(old_tokens_to_remove)} merge rules, removed {len(removed_merges)}: {removed_merges}"
        )

    added_tokens = [
        token
        for token in data.get("added_tokens", [])
        if token.get("content") not in SPECIAL_TOKENS
    ]
    for token in sorted(SPECIAL_TOKENS, key=replacement_ids.__getitem__):
        added_tokens.append(
            {
                "id": replacement_ids[token],
                "content": token,
                "single_word": False,
                "lstrip": False,
                "rstrip": False,
                "normalized": False,
                "special": True,
            }
        )
    data["added_tokens"] = added_tokens

    write_json(tokenizer_json, data)
    return replacement_ids


def update_tokenizer_configs(tokenizer_dir: Path) -> None:
    tokenizer_config = tokenizer_dir / "tokenizer_config.json"
    special_tokens_map = tokenizer_dir / "special_tokens_map.json"

    if tokenizer_config.exists():
        config = read_json(tokenizer_config)
    else:
        config = {}
    config.update(
        {
            "additional_special_tokens": SPECIAL_TOKENS,
            "chat_template": CHAT_TEMPLATE,
            "tokenizer_class": "PreTrainedTokenizerFast",
        }
    )
    config.pop("unk_token", None)
    write_json(tokenizer_config, config)

    if special_tokens_map.exists():
        special_map = read_json(special_tokens_map)
    else:
        special_map = {}
    special_map["additional_special_tokens"] = SPECIAL_TOKENS
    special_map.pop("unk_token", None)
    write_json(special_tokens_map, special_map)


def validate_tokenizer(tokenizer_dir: Path, replacement_ids: dict[str, int]) -> None:
    tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer_dir)
    if len(tokenizer) != EXPECTED_VOCAB_SIZE:
        raise ValueError(f"Expected tokenizer length {EXPECTED_VOCAB_SIZE}, found {len(tokenizer)}")

    for token, expected_id in replacement_ids.items():
        actual_id = tokenizer.convert_tokens_to_ids(token)
        if actual_id != expected_id:
            raise ValueError(f"Expected {token!r} id {expected_id}, found {actual_id}")
        encoded = tokenizer.encode(token, add_special_tokens=False)
        if encoded != [expected_id]:
            raise ValueError(f"Expected {token!r} to encode as [{expected_id}], found {encoded}")

    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": "Hello"}, {"role": "assistant", "content": "Hi"}],
        tokenize=False,
        add_generation_prompt=False,
    )
    if "<start_of_turn>user\nHello<end_of_turn>\n" not in rendered:
        raise ValueError(f"Unexpected chat template rendering: {rendered!r}")


def main() -> None:
    args = parse_args()
    tokenizer_dir = args.tokenizer_dir.resolve()
    if not (tokenizer_dir / "tokenizer.json").exists():
        raise FileNotFoundError(f"Missing tokenizer.json in {tokenizer_dir}")

    if args.dry_run:
        with tempfile.TemporaryDirectory() as tmpdir:
            dry_dir = Path(tmpdir) / "tokenizer"
            shutil.copytree(tokenizer_dir, dry_dir)
            replacement_ids = replace_tokenizer_json(dry_dir / "tokenizer.json")
            update_tokenizer_configs(dry_dir)
            validate_tokenizer(dry_dir, replacement_ids)
        print("Dry run passed.")
        return

    replacement_ids = replace_tokenizer_json(tokenizer_dir / "tokenizer.json")
    update_tokenizer_configs(tokenizer_dir)
    validate_tokenizer(tokenizer_dir, replacement_ids)
    print(f"Updated {tokenizer_dir}")
    for token, token_id in sorted(replacement_ids.items(), key=lambda item: item[1]):
        print(f"{token_id}: {token}")


if __name__ == "__main__":
    main()
