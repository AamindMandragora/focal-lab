from types import SimpleNamespace

import pytest

from synthesis.generate.generator import StrategyGenerator


class _FakeMessages:
    def __init__(self):
        self.kwargs = None

    def stream(self, **kwargs):
        self.kwargs = kwargs
        return _FakeStream()


class _FakeStream:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def get_final_message(self):
        return SimpleNamespace(content=[SimpleNamespace(text="strategy body")])


class _FakeClient:
    def __init__(self):
        self.messages = _FakeMessages()


def _anthropic_generator(**kwargs):
    generator = StrategyGenerator(
        model_name=kwargs.pop("model_name", "claude-opus-4-7"),
        backend="anthropic",
        max_new_tokens=kwargs.pop("max_new_tokens", 16000),
        **kwargs,
    )
    generator._client = _FakeClient()
    generator._ensure_backend_loaded = lambda: None
    generator._log_prompt_io = lambda *args, **kwargs: None
    return generator


def test_opus47_auto_anthropic_thinking_uses_adaptive_mode_with_effort():
    generator = _anthropic_generator(
        anthropic_thinking="auto",
        anthropic_effort="high",
        anthropic_thinking_display="omitted",
    )

    assert generator._generate_text("system", "user") == "strategy body"

    kwargs = generator._client.messages.kwargs
    assert kwargs["thinking"] == {"type": "adaptive", "display": "omitted"}
    assert kwargs["output_config"] == {"effort": "high"}
    assert "temperature" not in kwargs
    assert "top_p" not in kwargs


def test_anthropic_thinking_off_omits_thinking_payload():
    generator = _anthropic_generator(anthropic_thinking="off")

    generator._generate_text("system", "user")

    kwargs = generator._client.messages.kwargs
    assert "thinking" not in kwargs
    assert "output_config" not in kwargs


def test_opus47_rejects_manual_enabled_thinking_before_api_call():
    generator = _anthropic_generator(
        anthropic_thinking="enabled",
        anthropic_thinking_budget_tokens=4096,
    )

    with pytest.raises(ValueError, match="manual Anthropic thinking"):
        generator._generate_text("system", "user")
