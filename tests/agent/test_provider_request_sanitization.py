from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Protocol

from agent.message_sanitization import (
    ProviderRequestPayload,
    sanitize_provider_request_payload,
)


class ChatCompletionsClient(Protocol):
    def create(self, **kwargs: Any) -> SimpleNamespace: ...


def _message_content(payload: ProviderRequestPayload, index: int) -> str:
    assert isinstance(payload, dict)
    messages = payload["messages"]
    assert isinstance(messages, list)
    message = messages[index]
    assert isinstance(message, dict)
    content = message["content"]
    assert isinstance(content, str)
    return content


def _message_roles(payload: ProviderRequestPayload) -> list[str]:
    assert isinstance(payload, dict)
    messages = payload["messages"]
    assert isinstance(messages, list)
    roles: list[str] = []
    for message in messages:
        assert isinstance(message, dict)
        role = message["role"]
        assert isinstance(role, str)
        roles.append(role)
    return roles


def test_request_sanitizer_rewrites_hermes_agent_trigger_without_mutation() -> None:
    payload: ProviderRequestPayload = {
        "messages": [
            {"role": "system", "content": "Hermes Agent policy"},
            {"role": "user", "content": "Ask Hermes Agent for help"},
        ],
        "extra_body": {"system": "Hermes Agent metadata"},
    }

    sanitized = sanitize_provider_request_payload(payload)

    assert sanitized is not payload
    assert _message_content(sanitized, 0) == "Hermes runtime policy"
    assert _message_content(sanitized, 1) == "Ask Hermes runtime for help"
    assert isinstance(sanitized, dict)
    extra_body = sanitized["extra_body"]
    assert isinstance(extra_body, dict)
    assert extra_body["system"] == "Hermes runtime metadata"
    assert _message_content(payload, 0) == "Hermes Agent policy"


def test_request_sanitizer_drops_empty_user_messages() -> None:
    payload: ProviderRequestPayload = {
        "messages": [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": ""},
            {"role": "user", "content": "   \n"},
            {"role": "assistant", "content": ""},
            {"role": "user", "content": "real ask"},
        ]
    }

    sanitized = sanitize_provider_request_payload(payload)

    assert _message_roles(sanitized) == ["system", "assistant", "user"]
    assert _message_content(sanitized, -1) == "real ask"
    assert isinstance(payload, dict)
    messages = payload["messages"]
    assert isinstance(messages, list)
    assert len(messages) == 5


def test_request_sanitizer_keeps_clean_payload_identity() -> None:
    payload: ProviderRequestPayload = {"messages": [{"role": "user", "content": "hello"}]}

    assert sanitize_provider_request_payload(payload) is payload


def test_nonstream_dispatch_sanitizes_provider_request_payload() -> None:
    from agent.chat_completion_helpers import _dispatch_nonstreaming_api_request

    captured: dict[str, Any] = {}

    class Completions:
        def create(self, **kwargs: Any) -> SimpleNamespace:
            captured.update(kwargs)
            return SimpleNamespace(choices=[])

    completions: ChatCompletionsClient = Completions()
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    agent = SimpleNamespace(api_mode="chat_completions", provider="zai")
    payload: dict[str, Any] = {
        "messages": [
            {"role": "system", "content": "Hermes Agent system"},
            {"role": "user", "content": ""},
            {"role": "user", "content": "Ask Hermes Agent"},
        ]
    }

    response = _dispatch_nonstreaming_api_request(
        agent,
        payload,
        make_client=lambda _reason: client,
    )

    assert response.choices == []
    assert captured["messages"] == [
        {"role": "system", "content": "Hermes runtime system"},
        {"role": "user", "content": "Ask Hermes runtime"},
    ]
    assert len(payload["messages"]) == 3
