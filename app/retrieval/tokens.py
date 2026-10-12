"""Token counting with tiktoken's cl100k_base, as required by Section 4.3."""
from __future__ import annotations

import functools


@functools.lru_cache(maxsize=1)
def _encoding():
    import tiktoken

    return tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    if not text:
        return 0
    return len(_encoding().encode(text, disallowed_special=()))
