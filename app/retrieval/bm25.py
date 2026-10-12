"""A small, dependency-free Okapi BM25 used over the *already filtered* candidate chunks."""
from __future__ import annotations

import math
import re
from collections import Counter

_TOKEN = re.compile(r"[a-z0-9]+(?:[.'][a-z0-9]+)*")
_STOP = frozenset(
    "a an the and or of to in on for is are was were be been by with at as it its this that these those i me my "
    "we our you your do does did can could should would will shall may might how what when where which who whom "
    "there their them they he she his her if than then so not no from about into per any all many much get got have has had "
    "am im i'm tell please know".split()
)


def _stem(tok: str) -> str:
    tok = tok.replace("'", "")
    for suffix in ("ing", "ed"):
        if tok.endswith(suffix) and len(tok) > len(suffix) + 3:
            return tok[: -len(suffix)]
    if tok.endswith("ies") and len(tok) > 4:
        return tok[:-3] + "y"
    if tok.endswith("s") and not tok.endswith("ss") and len(tok) > 3:
        return tok[:-1]
    return tok


def tokenize(text: str) -> list[str]:
    return [_stem(t) for t in _TOKEN.findall(text.lower()) if t not in _STOP]


class BM25:
    def __init__(self, docs: list[list[str]], k1: float = 1.4, b: float = 0.75):
        self.k1, self.b = k1, b
        self.tf = [Counter(d) for d in docs]
        self.len = [len(d) for d in docs]
        self.avg = (sum(self.len) / len(docs)) if docs else 0.0
        df: Counter = Counter()
        for d in docs:
            df.update(set(d))
        n = len(docs)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def scores(self, query: list[str], subset: list[int] | None = None) -> list[float]:
        """Scores for every document, or only for the positions in `subset`."""
        out = []
        q = Counter(query)
        positions = range(len(self.tf)) if subset is None else subset
        for tf, ln in ((self.tf[i], self.len[i]) for i in positions):
            s = 0.0
            for term, qf in q.items():
                f = tf.get(term)
                if not f:
                    continue
                denom = f + self.k1 * (1 - self.b + self.b * ln / (self.avg or 1))
                s += self.idf.get(term, 0.0) * f * (self.k1 + 1) / denom * (1 + 0.2 * (qf - 1))
            out.append(s)
        return out
