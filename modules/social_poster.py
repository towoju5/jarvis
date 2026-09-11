"""Social platform upload/comment clients.

Every platform here gates public posting behind an approval process that
no amount of code gets around:
  - YouTube: OAuth2 (installed-app flow) + Data API v3 -- no review needed
    for a personal channel, just a Google Cloud OAuth client.
  - TikTok: Content Posting API requires an audited app; unaudited apps
    can only create drafts/private posts.
  - Facebook/Instagram (Graph API): requires Meta App Review + business
    verification before public posting or comment auto-replies work.

Each `upload()`/`reply_to_comments()` call is wrapped so an unexpected API
exception is logged with full context and then re-raised -- it must reach
main.py's stdout/stderr uncaught for the watchdog (core/watchdog.py) to
see the traceback and attempt a patch. Swallowing it here would hide the
crash from that mechanism entirely.
"""
from __future__ import annotations

import abc
import logging
import os
from dataclasses import dataclass
from pathlib import Path

import aiohttp

logger = logging.getLogger(__name__)


@dataclass
class UploadResult:
    platform: str
    post_id: str
    url: str | None = None


class SocialPlatformClient(abc.ABC):
    platform_name: str = "unknown"

    @abc.abstractmethod
    async def upload(self, video_path: Path, title: str, description: str) -> UploadResult:
        ...

    @abc.abstractmethod
    async def reply_to_comments(self, post_id: str, reply_text: str) -> None:
        ...

    async def _guarded(self, coro):
        """Log-and-reraise wrapper -- see module docstring for why this
        does not swallow the exception."""
        try:
            return await coro
        except Exception:
            logger.exception("%s API call failed", self.platform_name)
            raise


class YouTubeClient(SocialPlatformClient):
    platform_name = "youtube"
    UPLOAD_URL = "https://www.googleapis.com/upload/youtube/v3/videos"

    def __init__(self, oauth_access_token: str | None = None) -> None:
        self._access_token = oauth_access_token or os.getenv("YOUTUBE_OAUTH_ACCESS_TOKEN", "")

    async def upload(self, video_path: Path, title: str, description: str) -> UploadResult:
        async def _do():
            if not self._access_token:
                raise RuntimeError("YOUTUBE_OAUTH_ACCESS_TOKEN not set (OAuth2 installed-app flow required)")
            metadata = {
                "snippet": {"title": title, "description": description, "categoryId": "1"},
                "status": {"privacyStatus": "private"},
            }
            headers = {"Authorization": f"Bearer {self._access_token}"}
            async with aiohttp.ClientSession() as session:
                with open(video_path, "rb") as video_file:
                    form = aiohttp.FormData()
                    form.add_field("metadata", str(metadata), content_type="application/json")
                    form.add_field("file", video_file, content_type="video/*")
                    async with session.post(
                        self.UPLOAD_URL, params={"part": "snippet,status", "uploadType": "multipart"},
                        headers=headers, data=form,
                    ) as resp:
                        resp.raise_for_status()
                        data = await resp.json()
            return UploadResult(self.platform_name, data["id"], f"https://youtu.be/{data['id']}")

        return await self._guarded(_do())

    async def reply_to_comments(self, post_id: str, reply_text: str) -> None:
        async def _do():
            raise NotImplementedError("wire YouTube commentThreads.insert here")
        await self._guarded(_do())


class TikTokClient(SocialPlatformClient):
    platform_name = "tiktok"
    # Content Posting API: https://open.tiktokapis.com/v2/post/publish/video/init/
    UPLOAD_INIT_URL = "https://open.tiktokapis.com/v2/post/publish/video/init/"

    def __init__(self, access_token: str | None = None) -> None:
        self._access_token = access_token or os.getenv("TIKTOK_ACCESS_TOKEN", "")

    async def upload(self, video_path: Path, title: str, description: str) -> UploadResult:
        async def _do():
            if not self._access_token:
                raise RuntimeError("TIKTOK_ACCESS_TOKEN not set")
            raise NotImplementedError(
                "TikTok's Content Posting API requires an audited app for public posts; "
                "unaudited apps are limited to draft uploads. Wire the init/upload/publish "
                "sequence here once your app is audited."
            )
        return await self._guarded(_do())

    async def reply_to_comments(self, post_id: str, reply_text: str) -> None:
        async def _do():
            raise NotImplementedError("wire TikTok comment reply endpoint here")
        await self._guarded(_do())


class MetaGraphClient(SocialPlatformClient):
    """Shared base for Facebook Page and Instagram Business posting (Graph API)."""

    GRAPH_API_BASE = "https://graph.facebook.com/v19.0"

    def __init__(self, page_access_token: str | None = None) -> None:
        self._access_token = page_access_token or os.getenv("META_PAGE_ACCESS_TOKEN", "")

    async def reply_to_comments(self, post_id: str, reply_text: str) -> None:
        async def _do():
            if not self._access_token:
                raise RuntimeError("META_PAGE_ACCESS_TOKEN not set")
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.GRAPH_API_BASE}/{post_id}/comments",
                    params={"message": reply_text, "access_token": self._access_token},
                ) as resp:
                    resp.raise_for_status()
        await self._guarded(_do())


class FacebookClient(MetaGraphClient):
    platform_name = "facebook"

    def __init__(self, page_id: str | None = None, page_access_token: str | None = None) -> None:
        super().__init__(page_access_token)
        self._page_id = page_id or os.getenv("FACEBOOK_PAGE_ID", "")

    async def upload(self, video_path: Path, title: str, description: str) -> UploadResult:
        async def _do():
            if not (self._page_id and self._access_token):
                raise RuntimeError("FACEBOOK_PAGE_ID / META_PAGE_ACCESS_TOKEN not set")
            async with aiohttp.ClientSession() as session:
                with open(video_path, "rb") as video_file:
                    form = aiohttp.FormData()
                    form.add_field("description", description)
                    form.add_field("access_token", self._access_token)
                    form.add_field("source", video_file, content_type="video/*")
                    async with session.post(f"{self.GRAPH_API_BASE}/{self._page_id}/videos", data=form) as resp:
                        resp.raise_for_status()
                        data = await resp.json()
            return UploadResult(self.platform_name, data["id"])
        return await self._guarded(_do())


class InstagramClient(MetaGraphClient):
    platform_name = "instagram"

    def __init__(self, ig_business_account_id: str | None = None, page_access_token: str | None = None) -> None:
        super().__init__(page_access_token)
        self._ig_id = ig_business_account_id or os.getenv("INSTAGRAM_BUSINESS_ACCOUNT_ID", "")

    async def upload(self, video_path: Path, title: str, description: str) -> UploadResult:
        async def _do():
            if not (self._ig_id and self._access_token):
                raise RuntimeError("INSTAGRAM_BUSINESS_ACCOUNT_ID / META_PAGE_ACCESS_TOKEN not set")
            raise NotImplementedError(
                "Instagram's Content Publishing API needs the video hosted at a public URL "
                "(container create -> status poll -> publish); wire that flow here. "
                "Requires Meta App Review for a live (non-sandbox) account."
            )
        return await self._guarded(_do())
