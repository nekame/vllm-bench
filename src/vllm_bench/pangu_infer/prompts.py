"""Text prompts normalized to the benchmark's fixed context length."""

import json
from pathlib import Path


def load_tokenizer(path):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(path, trust_remote_code=True, local_files_only=True)


def load_prompts(path, tokenizer_path, context_length, vocab_size):
    rows = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(rows, list) or not rows:
        raise ValueError("prompts must be a nonempty JSON array of {id, input} objects")
    seen = set()
    for row in rows:
        if (not isinstance(row, dict) or type(row.get("id")) not in (int, str)
                or not isinstance(row.get("input"), str)):
            raise ValueError("Each prompt requires an integer/string id and a string input")
        key = str(row["id"])
        if key in seen:
            raise ValueError(f"Duplicate prompt id: {row['id']}")
        seen.add(key)
    tokenizer = load_tokenizer(tokenizer_path)
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id
    prompts = []
    for row in rows:
        tokens = tokenizer.encode(row["input"], add_special_tokens=True)
        if any(type(token) is not int or not 0 <= token < vocab_size for token in tokens):
            raise ValueError(f"Prompt {row['id']} has token IDs outside the model vocabulary")
        original = len(tokens)
        tokens = tokens[:context_length]
        padding = context_length - len(tokens)
        if padding:
            if type(pad_id) is not int or not 0 <= pad_id < vocab_size:
                raise ValueError("Short text prompts require a valid tokenizer PAD or EOS token")
            tokens = [pad_id] * padding + tokens
        prompts.append({"id": row["id"], "input": row["input"], "prompt_ids": tokens,
                        "original_tokens": original, "padding_tokens": padding,
                        "truncated_tokens": max(0, original - context_length)})
    return prompts
