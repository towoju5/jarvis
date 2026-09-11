"""Telegram-based two-way approval channel.

Uses long polling (not a webhook) -- no public port/URL needed, which is
the practical choice for an agent running on a home PC. Any caller can
`await bridge.request_approval(message)` and get back True/False once the
user taps a button, from anywhere in the app (the watchdog's patch/restart
gate, and later the post-generation publish gate both share this).
"""
from __future__ import annotations

import asyncio
import logging
import uuid

from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.ext import Application, CallbackQueryHandler, ContextTypes

logger = logging.getLogger(__name__)

APPROVE_LABEL = "Approve Task"
REJECT_LABEL = "Reject Task"


class TelegramApprovalBridge:
    def __init__(self, bot_token: str, chat_id: str) -> None:
        self._bot_token = bot_token
        self._chat_id = chat_id
        self._app: Application | None = None
        self._pending: dict[str, asyncio.Future] = {}

    @property
    def is_running(self) -> bool:
        return self._app is not None

    async def start(self) -> None:
        if not self._bot_token or not self._chat_id:
            raise RuntimeError("TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID must both be set")

        self._app = Application.builder().token(self._bot_token).build()
        self._app.add_handler(CallbackQueryHandler(self._on_callback))
        await self._app.initialize()
        await self._app.start()
        await self._app.updater.start_polling()
        logger.info("telegram approval bridge polling started")

    async def stop(self) -> None:
        if self._app is None:
            return
        for future in self._pending.values():
            if not future.done():
                future.cancel()
        self._pending.clear()
        await self._app.updater.stop()
        await self._app.stop()
        await self._app.shutdown()
        self._app = None
        logger.info("telegram approval bridge stopped")

    async def _on_callback(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        query = update.callback_query
        await query.answer()

        try:
            request_id, decision = (query.data or "").split(":", 1)
        except ValueError:
            return

        future = self._pending.pop(request_id, None)
        approved = decision == "approve"
        if future is not None and not future.done():
            future.set_result(approved)

        outcome = "Approved" if approved else "Rejected"
        original = query.message.text if query.message else ""
        try:
            await query.edit_message_text(f"{original}\n\n-> {outcome}", reply_markup=None)
        except Exception:
            logger.debug("could not edit telegram message after decision", exc_info=True)

    async def request_approval(self, message: str, timeout: float | None = None) -> bool:
        """Sends `message` with Approve/Reject buttons; returns True iff approved.

        Returns False on rejection OR on timeout (fail-safe default: don't
        proceed without an explicit approval).
        """
        if self._app is None:
            raise RuntimeError("TelegramApprovalBridge.start() must be called first")

        request_id = uuid.uuid4().hex
        keyboard = InlineKeyboardMarkup([[
            InlineKeyboardButton(APPROVE_LABEL, callback_data=f"{request_id}:approve"),
            InlineKeyboardButton(REJECT_LABEL, callback_data=f"{request_id}:reject"),
        ]])

        future: asyncio.Future[bool] = asyncio.get_running_loop().create_future()
        self._pending[request_id] = future

        await self._app.bot.send_message(chat_id=self._chat_id, text=message, reply_markup=keyboard)
        logger.info("sent approval request %s", request_id)

        try:
            return await asyncio.wait_for(future, timeout=timeout)
        except (asyncio.TimeoutError, asyncio.CancelledError):
            self._pending.pop(request_id, None)
            logger.warning("approval request %s timed out; defaulting to reject", request_id)
            return False
