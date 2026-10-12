"""OpenAI adapter: request shape, JSON extraction, temperature fallback."""
import json

import pytest

from app.llm.client import LLMClient, LLMError, extract_json


class FakeResp:
    def __init__(self, status, payload):
        self.status_code, self._payload = status, payload
        self.text = json.dumps(payload)

    def json(self):
        return self._payload


class FakeSession:
    def __init__(self, responses):
        self.responses, self.sent = list(responses), []

    def post(self, url, headers, json, timeout):
        self.sent.append({"url": url, "headers": headers, "body": json})
        return self.responses.pop(0)


OK = {"choices": [{"message": {"content": '{"ok": 1}'}}]}


def test_openai_request_shape(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "k")
    c = LLMClient("openai")
    c.session = FakeSession([FakeResp(200, OK)])
    assert c.complete_json("sys", "user", corpus_tokens=10, purpose="t", request_id="r") == {"ok": 1}
    sent = c.session.sent[0]
    assert sent["url"].endswith("/v1/chat/completions") and sent["headers"]["Authorization"] == "Bearer k"
    assert sent["body"]["temperature"] == 0 and sent["body"]["response_format"] == {"type": "json_object"}
    assert [m["role"] for m in sent["body"]["messages"]] == ["system", "user"]


def test_temperature_rejected_falls_back(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "k")
    c = LLMClient("openai")
    c.session = FakeSession([FakeResp(400, {"error": "temperature is not supported"}), FakeResp(200, OK)])
    assert c.complete_json("s", "u", corpus_tokens=0, purpose="t", request_id="r") == {"ok": 1}
    assert "temperature" not in c.session.sent[1]["body"]


def test_unknown_provider_rejected():
    with pytest.raises(LLMError):
        LLMClient("other").complete_json("s", "u", corpus_tokens=0, purpose="t", request_id="r")


def test_extract_json_variants():
    assert extract_json('{"a": 1}') == {"a": 1}
    assert extract_json('Sure! {"a": {"b": 2}} done') == {"a": {"b": 2}}
    assert extract_json('```json\n{"a": 1}\n```') == {"a": 1}
