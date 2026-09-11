"""Runtime configuration loaded from .env, with live reload on file change.

Changing AGENT_NAME (or any other value) in .env is picked up on the next
poll cycle without restarting the process -- callers can `subscribe()` to be
notified when settings change.
"""
from __future__ import annotations

import logging
import os
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from dotenv import dotenv_values

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ENV_PATH = PROJECT_ROOT / ".env"

RELOAD_POLL_INTERVAL_SECONDS = 2.0


@dataclass(frozen=True)
class Settings:
    agent_name: str = "Jarvis"
    trigger_hotkey: str = "<cmd>+z"
    execution_mode: str = "OFFLINE"

    whisper_model_size: str = "base"
    piper_voice_model: str = "en_US-lessac-medium"

    telegram_bot_token: str = ""
    telegram_chat_id: str = ""

    anthropic_api_key: str = ""

    dashboard_port: int = 8765

    @classmethod
    def _from_env_map(cls, env: dict[str, str | None]) -> "Settings":
        def get(key: str, default: str) -> str:
            value = env.get(key)
            return value if value else default

        defaults = cls()
        return cls(
            agent_name=get("AGENT_NAME", defaults.agent_name),
            trigger_hotkey=get("TRIGGER_HOTKEY", defaults.trigger_hotkey),
            execution_mode=get("EXECUTION_MODE", defaults.execution_mode).upper(),
            whisper_model_size=get("WHISPER_MODEL_SIZE", defaults.whisper_model_size),
            piper_voice_model=get("PIPER_VOICE_MODEL", defaults.piper_voice_model),
            telegram_bot_token=get("TELEGRAM_BOT_TOKEN", defaults.telegram_bot_token),
            telegram_chat_id=get("TELEGRAM_CHAT_ID", defaults.telegram_chat_id),
            anthropic_api_key=get("ANTHROPIC_API_KEY", defaults.anthropic_api_key),
            dashboard_port=int(get("DASHBOARD_PORT", str(defaults.dashboard_port)) or defaults.dashboard_port),
        )


SettingsListener = Callable[[Settings, Settings], None]


class SettingsManager:
    """Thread-safe holder for the current Settings, with change notification."""

    def __init__(self, env_path: Path = ENV_PATH) -> None:
        self._env_path = env_path
        self._lock = threading.Lock()
        self._listeners: list[SettingsListener] = []
        self._last_mtime: float | None = None
        self._settings = self._load()
        self._watch_thread: threading.Thread | None = None
        self._stop_event = threading.Event()

    def _load(self) -> Settings:
        env_map: dict[str, str | None] = {}
        if self._env_path.exists():
            env_map = dotenv_values(self._env_path)
            self._last_mtime = self._env_path.stat().st_mtime
        else:
            logger.warning(".env not found at %s, using defaults", self._env_path)
        # Real process environment variables take priority over .env values.
        merged = {**env_map, **os.environ}
        return Settings._from_env_map(merged)

    @property
    def current(self) -> Settings:
        with self._lock:
            return self._settings

    def reload(self, force: bool = False) -> Settings:
        with self._lock:
            if not force and self._env_path.exists():
                mtime = self._env_path.stat().st_mtime
                if self._last_mtime is not None and mtime == self._last_mtime:
                    return self._settings
            old = self._settings
            new = self._load()
            self._settings = new

        if old != new:
            logger.info("settings changed: agent_name=%r -> %r", old.agent_name, new.agent_name)
            for listener in list(self._listeners):
                try:
                    listener(old, new)
                except Exception:
                    logger.exception("settings listener raised")
        return new

    def subscribe(self, listener: SettingsListener) -> None:
        with self._lock:
            self._listeners.append(listener)

    def start_watching(self, interval: float = RELOAD_POLL_INTERVAL_SECONDS) -> None:
        if self._watch_thread is not None:
            return

        def _poll_loop() -> None:
            while not self._stop_event.wait(interval):
                try:
                    self.reload()
                except Exception:
                    logger.exception("failed to reload settings")

        self._stop_event.clear()
        self._watch_thread = threading.Thread(target=_poll_loop, name="settings-watch", daemon=True)
        self._watch_thread.start()

    def stop_watching(self) -> None:
        self._stop_event.set()
        if self._watch_thread is not None:
            self._watch_thread.join(timeout=1.0)
            self._watch_thread = None


# Process-wide singleton. Import `settings_manager` to subscribe to live
# changes, or `get_settings()` for a one-shot read of the current values.
settings_manager = SettingsManager()


def get_settings() -> Settings:
    return settings_manager.current
