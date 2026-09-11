"""Self-healing supervisor for main.py.

Guarded auto-patch flow (chosen over a blind auto-restart):

  1. Launch main.py as a subprocess, streaming stdout+stderr to
     logs/runtime.log.
  2. On crash, extract the traceback and the file it points to.
  3. Git-commit the current tree as a rollback checkpoint.
  4. Invoke `claude -p ... --dangerously-skip-permissions` headlessly to
     patch the offending file, and commit whatever it changed as a
     separate commit.
  5. Send a Telegram message with Approve/Reject buttons describing the
     crash and the patch. Restart main.py ONLY on explicit approval;
     otherwise roll back to the pre-patch checkpoint and stop.
  6. Give up after MAX_PATCH_ATTEMPTS consecutive crashes and leave
     main.py stopped for manual intervention, rather than looping forever
     and burning API calls on a bug it can't actually fix.

This intentionally does not restart automatically on its own patch --
that was a deliberate tradeoff (safety over full autonomy): a bad patch
here is a bug in a background agent with mic/hotkey/network access, and a
human should see what changed before it runs again.
"""
from __future__ import annotations

import asyncio
import logging
import re
import sys
from pathlib import Path

from git import InvalidGitRepositoryError, Repo

from communication.notify_bridge import TelegramApprovalBridge
from config.settings import get_settings

logger = logging.getLogger("watchdog")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RUNTIME_LOG_PATH = PROJECT_ROOT / "logs" / "runtime.log"
MAIN_SCRIPT = PROJECT_ROOT / "main.py"

MAX_PATCH_ATTEMPTS = 3
APPROVAL_TIMEOUT_SECONDS = 600  # 10 minutes; no response -> treated as reject

_TRACEBACK_START_RE = re.compile(r"^Traceback \(most recent call last\):\s*$", re.MULTILINE)
_FRAME_RE = re.compile(r'^\s*File "(?P<file>[^"]+)", line (?P<line>\d+)', re.MULTILINE)


def _clear_log() -> None:
    RUNTIME_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    RUNTIME_LOG_PATH.write_text("")


def _extract_last_traceback(log_text: str) -> tuple[str, str] | None:
    """Returns (offending_file_path, full_traceback_text) or None."""
    starts = [m.start() for m in _TRACEBACK_START_RE.finditer(log_text)]
    if not starts:
        return None
    tb_text = log_text[starts[-1]:].strip()

    frames = list(_FRAME_RE.finditer(tb_text))
    if not frames:
        return None

    project_frames = [f for f in frames if str(PROJECT_ROOT) in f.group("file")]
    chosen = project_frames[-1] if project_frames else frames[-1]
    return chosen.group("file"), tb_text


async def _run_child() -> int:
    with open(RUNTIME_LOG_PATH, "ab") as log_file:
        proc = await asyncio.create_subprocess_exec(
            sys.executable, str(MAIN_SCRIPT),
            cwd=str(PROJECT_ROOT),
            stdout=log_file,
            stderr=asyncio.subprocess.STDOUT,
        )
        logger.info("main.py started (pid=%s)", proc.pid)
        return await proc.wait()


def _snapshot(repo: Repo, message: str) -> str:
    """Commit any dirty state as a rollback point; return its SHA (or current HEAD)."""
    if repo.is_dirty(untracked_files=True):
        repo.git.add(A=True)
        repo.index.commit(message)
    return repo.head.commit.hexsha


def _commit_patch(repo: Repo, message: str) -> str | None:
    if not repo.is_dirty(untracked_files=True):
        return None
    repo.git.add(A=True)
    repo.index.commit(message)
    return repo.head.commit.hexsha


async def _invoke_claude_fix(file_path: str, traceback_text: str) -> bool:
    prompt = f"Fix the following runtime bug in file {file_path}:\n\n{traceback_text}"
    try:
        proc = await asyncio.create_subprocess_exec(
            "claude", "-p", prompt, "--dangerously-skip-permissions",
            cwd=str(PROJECT_ROOT),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
    except FileNotFoundError:
        logger.error("`claude` CLI not found on PATH; cannot auto-patch")
        return False

    stdout, stderr = await proc.communicate()
    logger.info(
        "claude fix attempt exit=%s\n--- stdout ---\n%s\n--- stderr ---\n%s",
        proc.returncode, stdout.decode(errors="replace")[:4000], stderr.decode(errors="replace")[:2000],
    )
    return proc.returncode == 0


async def _request_restart_approval(
    approval: TelegramApprovalBridge | None,
    agent_name: str,
    file_path: str,
    traceback_text: str,
    attempt: int,
) -> bool:
    if approval is None or not approval.is_running:
        logger.warning(
            "no Telegram approval bridge configured (set TELEGRAM_BOT_TOKEN/TELEGRAM_CHAT_ID); "
            "defaulting to reject so main.py doesn't restart unattended"
        )
        return False

    last_line = traceback_text.strip().splitlines()[-1] if traceback_text.strip() else "unknown error"
    message = (
        f"{agent_name} watchdog: crash detected in {file_path}\n"
        f"Error: {last_line}\n"
        f"Auto-patch attempt {attempt}/{MAX_PATCH_ATTEMPTS} has been applied and committed to git.\n"
        f"Approve restarting main.py with this patch?"
    )
    return await approval.request_approval(message, timeout=APPROVAL_TIMEOUT_SECONDS)


async def run() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stdout,
    )

    try:
        repo = Repo(PROJECT_ROOT)
    except InvalidGitRepositoryError:
        logger.error("watchdog requires a git repository at %s for its rollback safety net; run `git init` first", PROJECT_ROOT)
        return

    settings = get_settings()
    approval: TelegramApprovalBridge | None = None
    if settings.telegram_bot_token and settings.telegram_chat_id:
        approval = TelegramApprovalBridge(settings.telegram_bot_token, settings.telegram_chat_id)
        try:
            await approval.start()
        except Exception:
            logger.exception("failed to start telegram approval bridge; auto-patches will default to reject")
            approval = None
    else:
        logger.warning("TELEGRAM_BOT_TOKEN/TELEGRAM_CHAT_ID not set; auto-patches will default to reject")

    attempt = 0
    try:
        while True:
            _clear_log()
            exit_code = await _run_child()

            if exit_code == 0:
                logger.info("main.py exited cleanly; watchdog stopping")
                break

            log_text = RUNTIME_LOG_PATH.read_text(errors="replace")
            found = _extract_last_traceback(log_text)
            if found is None:
                logger.error("main.py exited with code %s but no traceback was found in the log; stopping", exit_code)
                break
            file_path, tb_text = found
            logger.error("main.py crashed (exit %s), offending file: %s\n%s", exit_code, file_path, tb_text)

            attempt += 1
            if attempt > MAX_PATCH_ATTEMPTS:
                logger.error("exceeded %d auto-patch attempts; stopping for manual intervention", MAX_PATCH_ATTEMPTS)
                break

            checkpoint_sha = _snapshot(repo, f"watchdog: pre-patch checkpoint (attempt {attempt}) before fixing {file_path}")
            patched = await _invoke_claude_fix(file_path, tb_text)
            if not patched:
                logger.error("claude did not produce a usable patch; stopping for manual intervention")
                break

            patch_sha = _commit_patch(repo, f"watchdog: auto-patch attempt {attempt} for crash in {file_path}")
            if patch_sha is None:
                logger.error("claude ran but made no file changes; stopping for manual intervention")
                break

            approved = await _request_restart_approval(approval, settings.agent_name, file_path, tb_text, attempt)
            if approved:
                logger.info("patch approved; restarting main.py")
                attempt = 0
                continue

            logger.info("patch not approved; rolling back to checkpoint %s and stopping", checkpoint_sha[:8])
            repo.git.reset("--hard", checkpoint_sha)
            break
    finally:
        if approval is not None:
            await approval.stop()


if __name__ == "__main__":
    asyncio.run(run())
