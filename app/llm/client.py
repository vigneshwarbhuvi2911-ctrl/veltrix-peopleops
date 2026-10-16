"""Thin LLM client for the OpenAI Chat Completions API (REST, JSON-only replies).

Every call goes through `LLMClient.complete_json`, which:
  * refuses to send more than the hard limit of corpus tokens (Section 4.3),
  * logs provider, model, purpose, latency and the corpus token count of the call,
  * never logs prompt or completion text (they can contain user-typed personal data).
"""
from __future__ import annotations

import json
import logging
import os
import re
import time
from typing import Callable

import requests

from app.logging_setup import log_event

log = logging.getLogger(__name__)

DEFAULT_MODELS = {
    "openai": "gpt-4.1-mini",
}


class LLMError(Exception):
    pass


class CorpusBudgetExceeded(LLMError):
    pass


def extract_json(text: str) -> dict:
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    start = text.find("{")
    depth = 0
    for i in range(start, len(text)) if start >= 0 else []:
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return json.loads(text[start : i + 1])
    raise LLMError("model did not return JSON")


class LLMClient:
    def __init__(self, provider: str, model: str = "", timeout_s: float = 20.0, hard_limit: int = 3000,
                 transport: Callable[[str, str, int], str] | None = None):
        self.provider = provider
        self.model = model or DEFAULT_MODELS.get(provider, "")
        self.timeout_s = timeout_s
        self.hard_limit = hard_limit
        self._transport = transport  # injected in tests: (system, user, max_tokens) -> text
        self._send_temperature = True
        self.session = requests.Session()

    # ------------------------------------------------------------------ public
    def complete_json(self, system: str, user: str, *, corpus_tokens: int, purpose: str,
                      request_id: str, max_tokens: int = 800) -> dict:
        if corpus_tokens > self.hard_limit:
            # Never reached in practice (the retriever packs below budget) but enforced here
            # as the single choke point every LLM call passes through.
            raise CorpusBudgetExceeded(f"{corpus_tokens} corpus tokens exceeds {self.hard_limit}")
        t0 = time.time()
        ok = False
        try:
            text = self._call(system, user, max_tokens)
            try:
                out = extract_json(text)
            except (LLMError, json.JSONDecodeError):
                text = self._call(system, user + "\n\nReturn ONLY the JSON object, nothing else.", max_tokens)
                out = extract_json(text)
            ok = True
            return out
        finally:
            log_event(log, "llm_call", request_id=request_id, purpose=purpose, provider=self.provider,
                      model=self.model, corpus_tokens=corpus_tokens, ok=ok,
                      latency_ms=int((time.time() - t0) * 1000))

    # ----------------------------------------------------------------- private
    def _call(self, system: str, user: str, max_tokens: int) -> str:
        if self._transport is not None:
            return self._transport(system, user, max_tokens)
        fn = {"openai": self._openai}.get(self.provider)
        if fn is None:
            raise LLMError(f"unsupported LLM_PROVIDER '{self.provider}'")
        return fn(system, user, max_tokens)

    def _post(self, url: str, headers: dict, body: dict) -> dict:
        for attempt in range(2):
            started = time.time()
            try:
                resp = self.session.post(url, headers=headers, json=body, timeout=self.timeout_s)
            except requests.RequestException as exc:
                # Network failure or timeout: surface as LLMError so the chat endpoint answers
                # with a polite retry message instead of an HTTP 500.
                raise LLMError(f"{self.provider} unreachable: {type(exc).__name__}") from exc
            if resp.status_code == 400 and self._send_temperature and "temperature" in resp.text.lower():
                # Some models reject sampling parameters; fall back to provider defaults.
                self._send_temperature = False
                body = {k: v for k, v in body.items() if k not in ("temperature", "seed")}
                continue
            # Retry a transient error once, but only if it failed fast: the whole chat request
            # has to finish within 30 seconds.
            if resp.status_code in (429, 500, 502, 503, 529) and attempt == 0 and time.time() - started < 4:
                time.sleep(1.0)
                continue
            if resp.status_code >= 400:
                raise LLMError(f"{self.provider} HTTP {resp.status_code}")
            return resp.json()
        raise LLMError(f"{self.provider} request failed")

    def _openai(self, system: str, user: str, max_tokens: int) -> str:
        key = os.environ.get("OPENAI_API_KEY", "")
        base = (os.environ.get("LLM_BASE_URL") or "https://api.openai.com").rstrip("/")
        body = {"model": self.model, "max_tokens": max_tokens, "response_format": {"type": "json_object"},
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]}
        if self._send_temperature:
            body |= {"temperature": 0, "seed": 7}
        data = self._post(f"{base}/v1/chat/completions",
                          {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}, body)
        return data["choices"][0]["message"]["content"] or ""
