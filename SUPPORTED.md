# Supported (current)

Minimal list of what's actually wired up right now, not the full blueprint scope.

- **Trigger:** wake word (openWakeWord, `hey_jarvis`) by default. Hotkey (`TRIGGER_HOTKEY`) exists but is off by default (`ENABLE_HOTKEY_TRIGGER=false`) -- pynput's X11-based listener generally can't receive keypresses on GNOME/Wayland.
- **Execution mode:** `OFFLINE` only (`ONLINE` is a stub, raises until a provider is wired in).
- **STT:** Faster-Whisper, `base.en` model.
- **TTS:** Piper, `en_US-lessac-medium` voice.
- **Dashboard:** local status page at `http://127.0.0.1:8765/` (agent state, task history, live logs).
- **Social platforms:** YouTube upload only. Facebook/Instagram/TikTok are unimplemented stubs (pending platform app review).
