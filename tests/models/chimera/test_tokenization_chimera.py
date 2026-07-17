# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import unittest
from pathlib import Path

from transformers import PreTrainedTokenizerFast
from transformers.testing_utils import require_tokenizers


TOKENIZER_DIR = Path(__file__).resolve().parents[3] / "src/transformers/models/chimera/tokenizer"
ADDITIONAL_SPECIAL_TOKENS = [
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


@require_tokenizers
class ChimeraTokenizerTest(unittest.TestCase):
    def setUp(self):
        self.tokenizer = PreTrainedTokenizerFast.from_pretrained(TOKENIZER_DIR)

    def test_special_token_contract(self):
        self.assertEqual(len(self.tokenizer), 50176)
        self.assertEqual(self.tokenizer.bos_token_id, 0)
        self.assertEqual(self.tokenizer.eos_token_id, 1)
        self.assertEqual(self.tokenizer.pad_token_id, 1)
        self.assertIsNone(self.tokenizer.unk_token)

        expected_ids = {"<start_of_turn>": 2, "<end_of_turn>": 3}
        for token, token_id in expected_ids.items():
            self.assertEqual(self.tokenizer.convert_tokens_to_ids(token), token_id)
            self.assertEqual(self.tokenizer.encode(token, add_special_tokens=False), [token_id])

        for token in ADDITIONAL_SPECIAL_TOKENS:
            self.assertIn(token, self.tokenizer.all_special_tokens)

        document_tokens = self.tokenizer.encode("A coherent English document.", add_special_tokens=True)
        self.assertNotEqual(document_tokens[0], self.tokenizer.bos_token_id)
        self.assertNotEqual(document_tokens[-1], self.tokenizer.eos_token_id)
        self.assertEqual((document_tokens + [self.tokenizer.eos_token_id])[-1], 1)

        self.assertEqual(len(self.tokenizer.encode("system", add_special_tokens=False)), 1)
        self.assertEqual(len(self.tokenizer.encode("user", add_special_tokens=False)), 1)
        self.assertEqual(len(self.tokenizer.encode("assistant", add_special_tokens=False)), 2)

    def test_chat_template(self):
        messages = [
            {"role": "system", "content": "Be concise."},
            {"role": "user", "content": "Hello"},
        ]
        rendered = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        self.assertEqual(
            rendered,
            "<start_of_turn>system\nBe concise.<end_of_turn>\n"
            "<start_of_turn>user\nHello<end_of_turn>\n"
            "<start_of_turn>assistant\n",
        )

        encoded = self.tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
        if isinstance(encoded, dict) or hasattr(encoded, "input_ids"):
            encoded = encoded["input_ids"]
        if encoded and isinstance(encoded[0], list):
            encoded = encoded[0]

        self.assertEqual(encoded[0], 2)
        self.assertNotIn(self.tokenizer.bos_token_id, encoded)
        self.assertEqual(encoded.count(2), 3)
        self.assertEqual(encoded.count(3), 2)


if __name__ == "__main__":
    unittest.main()
