"""General conversation, separate from the video-generation pipeline.

main.py routes a transcript here unless it explicitly asks to make/create
a video (see main.py's intent check). Multiple providers, tried in order
(Settings.chat_provider_order) with fallback to the next on any failure --
a provider with no API key configured is skipped, not attempted.

Claude uses Anthropic's native Messages API. Everything else (ChatGPT,
Groq/Llama, Qwen, Kimi) speaks the same OpenAI-compatible
`/chat/completions` shape, so they all share one HTTP client class --
adding another OpenAI-compatible provider later is a few lines, not a new
class. "ChatGPT realtime response" specifically means OpenAI's separate
voice-to-voice Realtime API (WebSocket audio in, audio out); that's not
used here since we already have our own Whisper/Piper pipeline -- wiring
it would duplicate that for no benefit. This uses standard OpenAI chat
completions instead, spoken through the existing TTS like every other
provider.
"""
from __future__ import annotations

import abc
import logging

import aiohttp

from config.settings import Settings

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are {agent_name}, a voice assistant running locally on the user's PC. "
    "Keep replies short and conversational -- they will be spoken aloud through "
    "text-to-speech, so avoid lists, markdown, code blocks, or anything that "
    "doesn't make sense read out loud."
)

REQUEST_TIMEOUT = aiohttp.ClientTimeout(total=30)


class ChatEngine(abc.ABC):
    name: str = "unknown"

    @abc.abstractmethod
    async def respond(self, agent_name: str, text: str) -> str:
        """Return a spoken-friendly reply to `text`. Raise on failure --
        callers (MultiProviderChatEngine) decide how to handle that."""


class ClaudeChatEngine(ChatEngine):
    name = "claude"
    API_URL = "https://api.anthropic.com/v1/messages"
    MODEL = "claude-haiku-4-5-20251001"  # fast/cheap; voice replies need low latency

    def __init__(self, api_key: str) -> None:
        self._api_key = api_key

    async def respond(self, agent_name: str, text: str) -> str:
        headers = {
            "x-api-key": self._api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }
        body = {
            "model": self.MODEL,
            "max_tokens": 300,
            "system": SYSTEM_PROMPT.format(agent_name=agent_name),
            "messages": [{"role": "user", "content": text}],
        }
        async with aiohttp.ClientSession(timeout=REQUEST_TIMEOUT) as session:
            async with session.post(self.API_URL, headers=headers, json=body) as resp:
                resp.raise_for_status()
                data = await resp.json()
        return "".join(block["text"] for block in data["content"] if block.get("type") == "text").strip()


class OpenAICompatibleChatEngine(ChatEngine):
    """Covers OpenAI/ChatGPT, Groq (Llama and others), Qwen (DashScope
    compatible mode), Kimi (Moonshot), and any other provider that speaks
    the same `/chat/completions` shape -- just point base_url at it."""

    def __init__(self, name: str, api_key: str, base_url: str, model: str) -> None:
        self.name = name
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._model = model

    async def respond(self, agent_name: str, text: str) -> str:
        headers = {"Authorization": f"Bearer {self._api_key}", "Content-Type": "application/json"}
        body = {
            "model": self._model,
            "max_tokens": 300,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT.format(agent_name=agent_name)},
                {"role": "user", "content": text},
            ],
        }
        async with aiohttp.ClientSession(timeout=REQUEST_TIMEOUT) as session:
            async with session.post(f"{self._base_url}/chat/completions", headers=headers, json=body) as resp:
                resp.raise_for_status()
                data = await resp.json()
        return data["choices"][0]["message"]["content"].strip()


class MultiProviderChatEngine(ChatEngine):
    name = "multi"

    def __init__(self, providers: list[ChatEngine]) -> None:
        self._providers = providers

    async def respond(self, agent_name: str, text: str) -> str:
        if not self._providers:
            raise RuntimeError(
                "No chat provider is configured. Set at least one of "
                "ANTHROPIC_API_KEY / OPENAI_API_KEY / GROQ_API_KEY / QWEN_API_KEY / KIMI_API_KEY."
            )
        last_error: Exception | None = None
        for provider in self._providers:
            try:
                return await provider.respond(agent_name, text)
            except Exception as exc:
                last_error = exc
                logger.warning("chat provider %r failed, trying next: %s", provider.name, exc)
        raise RuntimeError(f"all configured chat providers failed; last error: {last_error}")


_PROVIDER_BUILDERS = {
    "claude": lambda s: ClaudeChatEngine(s.anthropic_api_key) if s.anthropic_api_key else None,
    "openai": lambda s: OpenAICompatibleChatEngine(
        "openai", s.openai_api_key, "https://api.openai.com/v1", "gpt-4o-mini"
    ) if s.openai_api_key else None,
    "groq": lambda s: OpenAICompatibleChatEngine(
        "groq", s.groq_api_key, "https://api.groq.com/openai/v1", "llama-3.3-70b-versatile"
    ) if s.groq_api_key else None,
    "qwen": lambda s: OpenAICompatibleChatEngine(
        "qwen", s.qwen_api_key, "https://dashscope.aliyuncs.com/compatible-mode/v1", "qwen-plus"
    ) if s.qwen_api_key else None,
    "kimi": lambda s: OpenAICompatibleChatEngine(
        "kimi", s.kimi_api_key, "https://api.moonshot.ai/v1", "moonshot-v1-8k"
    ) if s.kimi_api_key else None,
}


def build_chat_engine(settings: Settings) -> MultiProviderChatEngine:
    providers: list[ChatEngine] = []
    for name in (n.strip().lower() for n in settings.chat_provider_order.split(",")):
        builder = _PROVIDER_BUILDERS.get(name)
        if builder is None:
            logger.warning("unknown chat provider %r in CHAT_PROVIDER_ORDER, skipping", name)
            continue
        engine = builder(settings)
        if engine is not None:
            providers.append(engine)
    if providers:
        logger.info("chat providers configured (in order): %s", ", ".join(p.name for p in providers))
    else:
        logger.warning("no chat provider API keys set; general conversation will not work")
    return MultiProviderChatEngine(providers)
