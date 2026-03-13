"""Instagram API client with proper error handling and rate limiting."""

import json
import logging
import random
import time
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import requests

logger = logging.getLogger(__name__)


class InstagramAPIError(Exception):
    """Instagram API error."""

    pass


class InstagramAuthError(InstagramAPIError):
    """Instagram authentication error."""

    pass


class InstagramRateLimitError(InstagramAPIError):
    """Instagram rate limit error."""

    pass


class InstagramClient:
    """Instagram API client with caching and rate limiting."""

    BASE_URL = "https://www.instagram.com/api/v1"
    DEFAULT_TIMEOUT = 30
    MAX_RETRIES = 3
    RETRY_DELAY = 2

    def __init__(
        self,
        sessionid: Optional[str] = None,
        raw_cookie: Optional[str] = None,
        user_agent: Optional[str] = None,
    ):
        self.sessionid = sessionid
        self.raw_cookie = raw_cookie
        self.user_agent = (
            user_agent
            or "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"
        )
        self._headers = self._build_headers()
        self._request_count = 0
        self._last_request_time = 0

    def _build_headers(self) -> Dict[str, str]:
        """Build request headers."""
        headers = {
            "User-Agent": self.user_agent,
            "Accept": "application/json, text/plain, */*",
            "Accept-Language": "en-US,en;q=0.5",
            "X-IG-App-ID": "936619743392459",
            "X-Requested-With": "XMLHttpRequest",
            "Origin": "https://www.instagram.com",
            "Referer": "https://www.instagram.com/saved/",
            "Sec-Fetch-Dest": "empty",
            "Sec-Fetch-Mode": "cors",
            "Sec-Fetch-Site": "same-origin",
        }

        if self.raw_cookie:
            headers["Cookie"] = self.raw_cookie
        elif self.sessionid:
            headers["Cookie"] = f"sessionid={self.sessionid}"

        return headers

    def _rate_limit(self) -> None:
        """Implement rate limiting."""
        current_time = time.time()
        time_since_last = current_time - self._last_request_time

        if time_since_last < 1:  # Max 1 request per second
            sleep_time = 1 - time_since_last
            time.sleep(sleep_time)

        self._last_request_time = time.time()
        self._request_count += 1

    def _make_request(
        self,
        method: str,
        endpoint: str,
        params: Optional[Dict] = None,
        retries: int = 0,
    ) -> Dict:
        """Make rate-limited request with retry logic."""
        self._rate_limit()

        url = f"{self.BASE_URL}/{endpoint}"

        try:
            if method == "GET":
                response = requests.get(
                    url,
                    headers=self._headers,
                    params=params,
                    timeout=self.DEFAULT_TIMEOUT,
                )
            else:
                raise ValueError(f"Unsupported method: {method}")

            if response.status_code == 429:
                if retries < self.MAX_RETRIES:
                    delay = self.RETRY_DELAY * (2**retries)
                    logger.warning(f"Rate limited. Retrying in {delay}s...")
                    time.sleep(delay)
                    return self._make_request(method, endpoint, params, retries + 1)
                raise InstagramRateLimitError("Rate limit exceeded")

            if response.status_code == 401:
                raise InstagramAuthError("Invalid or expired session")

            if response.status_code != 200:
                raise InstagramAPIError(
                    f"HTTP {response.status_code}: {response.text[:200]}"
                )

            return response.json()

        except requests.exceptions.Timeout:
            if retries < self.MAX_RETRIES:
                return self._make_request(method, endpoint, params, retries + 1)
            raise InstagramAPIError("Request timeout")

        except requests.exceptions.RequestException as e:
            if retries < self.MAX_RETRIES:
                return self._make_request(method, endpoint, params, retries + 1)
            raise InstagramAPIError(f"Request failed: {e}")

    def validate_session(self) -> Tuple[bool, str]:
        """Validate Instagram session."""
        try:
            data = self._make_request("GET", "feed/saved/posts/", {"max_id": "0"})

            if "items" in data or "status" in data:
                return True, "Session valid"

            return False, "Unexpected response structure"

        except InstagramAuthError:
            return False, "Invalid or expired session"
        except InstagramAPIError as e:
            return False, str(e)

    def fetch_saved_posts(
        self,
        limit: Optional[int] = None,
        existing_ids: Optional[set] = None,
    ) -> List[Dict]:
        """Fetch saved posts with pagination."""
        saved_posts = []
        next_max_id = None
        existing_ids = existing_ids or set()
        consecutive_existing = 0
        MAX_CONSECUTIVE = 5

        target_count = limit or float("inf")

        while len(saved_posts) < target_count:
            params = {}
            if next_max_id:
                params["max_id"] = next_max_id

            try:
                data = self._make_request("GET", "feed/saved/posts/", params)
            except InstagramAPIError as e:
                logger.error(f"Failed to fetch posts: {e}")
                break

            items = data.get("items", [])
            if not items:
                break

            for item in items:
                post = self._extract_post_data(item)
                if post and post.get("id"):
                    post_id = post["id"]

                    if post_id in existing_ids:
                        consecutive_existing += 1
                    else:
                        consecutive_existing = 0

                    saved_posts.append(post)

                    if len(saved_posts) >= target_count:
                        break

                    if consecutive_existing >= MAX_CONSECUTIVE and not limit:
                        logger.info(
                            f"Stopping early: found {MAX_CONSECUTIVE} existing posts"
                        )
                        return saved_posts

            next_max_id = data.get("next_max_id")
            if not next_max_id:
                break

        logger.info(f"Fetched {len(saved_posts)} posts")
        return saved_posts

    def fetch_collections(self) -> Dict[str, str]:
        """Fetch user's saved collections."""
        collections = {}
        next_max_id = None

        while True:
            params = {}
            if next_max_id:
                params["max_id"] = next_max_id

            try:
                data = self._make_request("GET", "collections/list/", params)
            except InstagramAPIError as e:
                logger.error(f"Failed to fetch collections: {e}")
                break

            for item in data.get("items", []):
                c_id = item.get("collection_id") or item.get("id")
                c_name = item.get("collection_name") or item.get("title")
                if c_id and c_name:
                    collections[str(c_id)] = c_name

            next_max_id = data.get("next_max_id")
            if not data.get("more_available") and not next_max_id:
                break

        return collections

    def _extract_post_data(self, item: Dict) -> Optional[Dict]:
        """Extract post data from API response."""
        try:
            media = item.get("media", item)
            if not media:
                return None

            shortcode = media.get("code") or media.get("shortcode", "")
            is_video = media.get("media_type") == 2

            # Extract URLs
            thumbnail_url = ""
            media_url = ""
            video_url = ""

            if "image_versions2" in media:
                candidates = media["image_versions2"].get("candidates", [])
                if candidates:
                    thumbnail_url = candidates[0].get("url", "")
                    media_url = candidates[-1].get("url", thumbnail_url)

            if is_video and "video_versions" in media:
                candidates = media.get("video_versions", [])
                if candidates:
                    video_url = candidates[0].get("url", "")
                    media_url = video_url or media_url

            post = {
                "id": media.get("id"),
                "shortcode": shortcode,
                "username": media.get("user", {}).get("username", "unknown"),
                "timestamp": datetime.fromtimestamp(
                    media.get("taken_at", 0), tz=timezone.utc
                ).isoformat(),
                "url": f"https://www.instagram.com/p/{shortcode}/" if shortcode else "",
                "likes": media.get("like_count", 0),
                "comments": media.get("comment_count", 0),
                "caption": media.get("caption", {}).get("text", "")
                if media.get("caption")
                else "",
                "media_type": media.get("media_type", 1),
                "is_video": is_video,
                "thumbnail_url": thumbnail_url,
                "media_url": media_url,
                "video_url": video_url,
                "saved_collection_ids": item.get("saved_collection_ids", [])
                or item.get("collection_ids", []),
            }

            # Handle carousel posts
            if media.get("media_type") == 8 and "carousel_media" in media:
                post["carousel_count"] = len(media["carousel_media"])
                if not post["thumbnail_url"]:
                    post["thumbnail_url"] = (
                        media["carousel_media"][0]
                        .get("image_versions2", {})
                        .get("candidates", [{}])[0]
                        .get("url", "")
                    )
                if not post["media_url"]:
                    post["media_url"] = post["thumbnail_url"]

            return post

        except Exception as e:
            logger.error(f"Error extracting post data: {e}")
            return None
