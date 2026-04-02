"""AI analysis service for Instagram posts."""

import json
import logging
import os
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import requests
from openai import AzureOpenAI

try:
    from google import genai
    from google.genai import types as genai_types
except ImportError:  # pragma: no cover - optional dependency in tests
    genai = None
    genai_types = None

from config import config

logger = logging.getLogger(__name__)

PRIMARY_CATEGORIES = [
    "AI & Tech",
    "Business & Career",
    "Health & Fitness",
    "Productivity & Systems",
    "Self-Development",
    "Finance",
    "Design & Creativity",
    "Travel",
    "Food",
    "Fashion & Beauty",
    "Home & Lifestyle",
    "Entertainment & Culture",
    "Relationships",
    "Other",
]

CATEGORY_ALIASES = {
    "technology": "AI & Tech",
    "ai": "AI & Tech",
    "tech": "AI & Tech",
    "business": "Business & Career",
    "career": "Business & Career",
    "fitness": "Health & Fitness",
    "health": "Health & Fitness",
    "wellness": "Health & Fitness",
    "productivity": "Productivity & Systems",
    "systems": "Productivity & Systems",
    "self development": "Self-Development",
    "self-development": "Self-Development",
    "personal growth": "Self-Development",
    "personal": "Self-Development",
    "money": "Finance",
    "design": "Design & Creativity",
    "art": "Design & Creativity",
    "creativity": "Design & Creativity",
    "travel": "Travel",
    "food": "Food",
    "fashion": "Fashion & Beauty",
    "beauty": "Fashion & Beauty",
    "home": "Home & Lifestyle",
    "lifestyle": "Home & Lifestyle",
    "entertainment": "Entertainment & Culture",
    "culture": "Entertainment & Culture",
    "relationships": "Relationships",
    "relationship": "Relationships",
    "education": "Self-Development",
    "other": "Other",
}


class AnalysisError(Exception):
    """Analysis error."""

    pass


class AIAnalyzer:
    """AI-powered content analyzer."""

    def __init__(
        self,
        azure_endpoint: Optional[str] = None,
        azure_key: Optional[str] = None,
        model: Optional[str] = None,
        api_version: Optional[str] = None,
        gemini_api_key: Optional[str] = None,
        gemini_model: Optional[str] = None,
    ):
        self.client: Optional[AzureOpenAI] = None
        self.gemini_client = None
        self.azure_endpoint = azure_endpoint or config.azure.endpoint
        self.azure_key = azure_key or config.azure.api_key
        self.api_version = api_version or config.azure.api_version
        self.model = model or config.azure.model
        self.gemini_api_key = gemini_api_key or config.gemini.api_key
        self.gemini_model = gemini_model or config.gemini.model
        self._init_client()
        self._init_gemini_client()

    def _init_client(self) -> None:
        """Initialize Azure OpenAI client."""
        if not (self.azure_endpoint and self.azure_key):
            return

        try:
            self.client = AzureOpenAI(
                api_key=self.azure_key,
                api_version=self.api_version,
                azure_endpoint=self.azure_endpoint,
            )
            logger.info("Azure OpenAI client initialized")
        except Exception as e:
            logger.error(f"Failed to initialize Azure client: {e}")

    def _init_gemini_client(self) -> None:
        """Initialize Gemini client when available."""
        if not self.gemini_api_key or genai is None:
            return

        try:
            self.gemini_client = genai.Client(api_key=self.gemini_api_key)
            logger.info("Gemini client initialized")
        except Exception as e:
            logger.error(f"Failed to initialize Gemini client: {e}")

    def is_available(self) -> bool:
        """Check if analyzer is available."""
        return self.gemini_client is not None or self.client is not None

    def validate_chat_deployment(self) -> Tuple[bool, str]:
        """Validate that the configured chat deployment exists and is callable."""
        if not self.client:
            return False, "Azure OpenAI client not available"

        try:
            self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": "ping"}],
                max_completion_tokens=8,
            )
            return True, "Azure deployment is valid"
        except Exception as e:
            return False, str(e)

    def validate_analysis_backend(self) -> Tuple[bool, str]:
        """Validate the configured analysis backend."""
        if self.gemini_client is not None:
            try:
                response = self.gemini_client.models.generate_content(
                    model=self.gemini_model,
                    contents="ping",
                )
                return True, f"Gemini analysis model is valid ({self.gemini_model})"
            except Exception as e:
                return False, str(e)

        return self.validate_chat_deployment()

    def download_media(self, url: str, suffix: str = ".mp4") -> Optional[str]:
        """Download media to temp file."""
        try:
            response = requests.get(url, stream=True, timeout=60)
            if response.status_code == 200:
                fd, path = tempfile.mkstemp(suffix=suffix)
                try:
                    with os.fdopen(fd, "wb") as f:
                        for chunk in response.iter_content(chunk_size=8192):
                            f.write(chunk)
                    return path
                except Exception:
                    os.close(fd)
                    return None
            return None
        except Exception as e:
            logger.error(f"Failed to download media: {e}")
            return None

    def extract_audio(self, video_path: str) -> Optional[str]:
        """Extract audio from video using ffmpeg."""
        audio_path = video_path.replace(".mp4", ".mp3")

        try:
            cmd = [
                "ffmpeg",
                "-i",
                video_path,
                "-q:a",
                "0",
                "-map",
                "a",
                audio_path,
                "-y",
                "-loglevel",
                "error",
            ]

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=60,
            )

            if result.returncode == 0 and os.path.exists(audio_path):
                return audio_path

            logger.error(f"ffmpeg failed: {result.stderr}")
            return None

        except subprocess.TimeoutExpired:
            logger.error("ffmpeg timeout")
            return None
        except Exception as e:
            logger.error(f"Failed to extract audio: {e}")
            return None

    def transcribe_audio(self, audio_path: str) -> Optional[str]:
        """Transcribe audio using Whisper."""
        if not self.client:
            return None

        for attempt in range(2):
            try:
                with open(audio_path, "rb") as f:
                    result = self.client.audio.transcriptions.create(
                        model="whisper",
                        file=f,
                    )
                return result.text
            except Exception as e:
                logger.warning(f"Transcription attempt {attempt + 1} failed: {e}")
                if attempt < 1:
                    time.sleep(2)

        return None

    def analyze_post(self, post: Dict) -> Tuple[Optional[Dict], int]:
        """Analyze a single post using AI."""
        if self.gemini_client is not None:
            return self._analyze_post_with_gemini(post)

        if not self.client:
            raise AnalysisError("No analysis client is available")

        # Process video if present
        transcript = None
        if post.get("is_video") and post.get("video_url"):
            video_path = self.download_media(post["video_url"])
            if video_path:
                try:
                    audio_path = self.extract_audio(video_path)
                    if audio_path:
                        try:
                            transcript = self.transcribe_audio(audio_path)
                        finally:
                            try:
                                os.remove(audio_path)
                            except OSError:
                                pass
                finally:
                    try:
                        os.remove(video_path)
                    except OSError:
                        pass

        # Build prompt
        prompt = self._build_prompt(post, transcript)

        # Prepare content
        user_content = [{"type": "text", "text": prompt}]
        if post.get("thumbnail_url"):
            user_content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": post["thumbnail_url"]},
                }
            )

        messages = [
            {
                "role": "system",
                "content": "You are a sophisticated knowledge extraction AI.",
            },
            {"role": "user", "content": user_content},
        ]

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_completion_tokens=1500,
                response_format={"type": "json_object"},
            )

            content = response.choices[0].message.content or ""
            tokens_used = response.usage.total_tokens if response.usage else 0

            # Parse JSON
            content = content.strip()
            if content.startswith("```json"):
                content = content[7:]
            if content.endswith("```"):
                content = content[:-3]

            result = json.loads(content.strip())
            result = self._normalize_result(result)

            if transcript:
                result["video_transcript"] = transcript

            return result, tokens_used

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse AI response: {e}")
            return None, 0
        except Exception as e:
            logger.error(f"AI analysis failed: {e}")
            return None, 0

    def _analyze_post_with_gemini(self, post: Dict) -> Tuple[Optional[Dict], int]:
        """Analyze a post with Gemini multimodal input."""
        if self.gemini_client is None:
            raise AnalysisError("Gemini client not available")

        prompt = self._build_prompt(post, transcript=None)
        contents: List = [prompt]
        video_path = None
        image_path = None
        uploaded_file = None

        try:
            if post.get("is_video") and post.get("video_url"):
                video_path = self.download_media(post["video_url"], suffix=".mp4")
                if video_path:
                    uploaded_file = self.gemini_client.files.upload(file=video_path)
                    uploaded_file = self._wait_for_uploaded_file(uploaded_file)
                    contents = [uploaded_file, prompt]
            elif post.get("thumbnail_url"):
                image_path = self.download_media(post["thumbnail_url"], suffix=".jpg")
                if image_path and genai_types is not None:
                    with open(image_path, "rb") as handle:
                        image_part = genai_types.Part.from_bytes(
                            data=handle.read(),
                            mime_type="image/jpeg",
                        )
                    contents = [image_part, prompt]

            response = self.gemini_client.models.generate_content(
                model=self.gemini_model,
                contents=contents,
                config={"response_mime_type": "application/json"},
            )

            content = getattr(response, "text", "") or ""
            result = json.loads(content.strip())
            result = self._normalize_result(result)
            tokens_used = self._extract_gemini_tokens_used(response)
            return result, tokens_used
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse Gemini response: {e}")
            return None, 0
        except Exception as e:
            logger.error(f"Gemini analysis failed: {e}")
            return None, 0
        finally:
            if uploaded_file is not None:
                try:
                    self.gemini_client.files.delete(name=uploaded_file.name)
                except Exception:
                    pass
            for path in (video_path, image_path):
                if path:
                    try:
                        os.remove(path)
                    except OSError:
                        pass

    def _wait_for_uploaded_file(self, uploaded_file):
        """Wait until a Gemini file upload is ready to use."""
        file_name = getattr(uploaded_file, "name", None)
        if not file_name:
            return uploaded_file

        for _ in range(60):
            current = self.gemini_client.files.get(name=file_name)
            state = getattr(getattr(current, "state", None), "name", None) or getattr(
                current, "state", None
            )
            if state in {None, "ACTIVE", "SUCCEEDED", "READY"}:
                return current
            if state in {"FAILED", "ERROR"}:
                raise AnalysisError(f"Gemini file upload failed with state: {state}")
            time.sleep(2)

        raise AnalysisError("Timed out waiting for Gemini file processing")

    def _extract_gemini_tokens_used(self, response) -> int:
        """Best-effort extraction of Gemini token usage."""
        usage = getattr(response, "usage_metadata", None)
        if usage is None:
            return 0

        for attr in ("total_token_count", "total_tokens"):
            value = getattr(usage, attr, None)
            if value is not None:
                return int(value)
        return 0

    def _build_prompt(self, post: Dict, transcript: Optional[str]) -> str:
        """Build analysis prompt."""
        prompt = f"""
Deeply analyze this Instagram content to create a high-value knowledge base entry.

Content Details:
- Type: {"Video" if post.get("is_video") else "Photo"}
- User: {post.get("username", "unknown")}
- Caption: {post.get("caption", "")}
- Likes/Comments: {post.get("likes", 0)} / {post.get("comments", 0)}
- Timestamp: {post.get("timestamp", "")}
"""
        if transcript:
            prompt += f"\nVideo Transcript:\n{transcript}\n"

        prompt += """
For videos, use the full video itself to reason about speech, on-screen text, visuals, and sequence of events.
Choose one primary category from this controlled vocabulary only:
- AI & Tech
- Business & Career
- Health & Fitness
- Productivity & Systems
- Self-Development
- Finance
- Design & Creativity
- Travel
- Food
- Fashion & Beauty
- Home & Lifestyle
- Entertainment & Culture
- Relationships
- Other

Category rules:
- Pick the most useful knowledge category for why the reel was likely saved.
- Use Self-Development for mindset, habits, life advice, and personal growth content.
- Use Other only if nothing else clearly fits.
- Do not default to vague or personal buckets for educational/advice content.
- Produce 4-8 specific topics grounded in the actual caption, spoken content, OCR text, and visuals.
- Topics should be short phrases like "meal prep", "founder storytelling", "sleep hygiene", not generic words like "motivation".

Provide analysis in this exact JSON format:
{
    "category": "one of the controlled categories above",
    "sentiment": {"score": float -1 to 1, "label": "Positive/Negative/Neutral"},
    "credibility_score": integer 0-100,
    "topics": ["specific", "tags"],
    "learning_points": ["bullet", "points"],
    "action_items": ["Concrete actions"],
    "ocr_text": "Text visible in image/video",
    "visual_description": "Detailed visual description",
    "video_transcript": "Spoken content summary or transcript when applicable",
    "video_summary": "Video summary if applicable"
}

Focus on: 1) Actionability 2) Knowledge extraction 3) Completeness
"""
        return prompt

    def classify_saved_content(
        self,
        post: Dict,
        analysis_fields: Dict[str, Optional[str]],
    ) -> Optional[Dict[str, List[str] | str]]:
        """Refresh category and topic tags from stored text without re-downloading media."""
        prompt = f"""
Reclassify this already-analyzed Instagram save using only the stored text artifacts.

Choose one primary category from this vocabulary only:
{json.dumps(PRIMARY_CATEGORIES, ensure_ascii=False)}

Instructions:
- Choose the category that best explains the practical value of the save.
- Use Other rarely.
- Generate 4 to 8 highly specific topic tags grounded in the provided text.
- Prefer useful subject tags over broad mood words.

Post context:
- User: {post.get("username", "unknown")}
- Caption: {post.get("caption", "")}
- OCR text: {analysis_fields.get("ocr_text", "")}
- Transcript: {analysis_fields.get("video_transcript", "")}
- Visual description: {analysis_fields.get("visual_description", "")}
- Video summary: {analysis_fields.get("video_summary", "")}

Return strict JSON only:
{{
  "category": "one primary category",
  "topics": ["tag 1", "tag 2"]
}}
""".strip()

        try:
            if self.gemini_client is not None:
                response = self.gemini_client.models.generate_content(
                    model=self.gemini_model,
                    contents=prompt,
                    config={"response_mime_type": "application/json"},
                )
                payload = json.loads((getattr(response, "text", "") or "{}").strip())
                return self._normalize_classification(payload)

            if self.client is None:
                return None

            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You classify saved short-form content into a strict taxonomy and respond with JSON only.",
                    },
                    {"role": "user", "content": prompt},
                ],
                max_completion_tokens=400,
                response_format={"type": "json_object"},
            )
            payload = json.loads((response.choices[0].message.content or "{}").strip())
            return self._normalize_classification(payload)
        except Exception as exc:
            logger.error(f"Category refresh failed: {exc}")
            return None

    def _normalize_classification(self, payload: Dict) -> Dict[str, List[str] | str]:
        """Normalize refreshed category/tag output."""
        category = self._normalize_category(payload.get("category"))
        topics = self._clean_list(payload.get("topics"), limit=8)
        return {"category": category, "topics": topics}

    def _normalize_result(self, result: Dict) -> Dict:
        """Normalize analysis result."""
        sentiment = result.get("sentiment", {})
        score = float(sentiment.get("score", 0)) if sentiment else 0.0
        score = max(-1.0, min(1.0, score))

        label = sentiment.get("label", "Neutral") if sentiment else "Neutral"
        label_map = {
            "positive": "Positive",
            "negative": "Negative",
            "neutral": "Neutral",
        }
        normalized_label = label_map.get(str(label).lower(), "Neutral")

        # Determine label from score if missing
        if normalized_label == "Neutral" and score != 0:
            normalized_label = (
                "Positive"
                if score > 0.15
                else "Negative"
                if score < -0.15
                else "Neutral"
            )

        credibility = result.get("credibility_score")
        if credibility is not None:
            try:
                credibility = max(0, min(100, int(credibility)))
            except (ValueError, TypeError):
                credibility = None

        return {
            "category": self._normalize_category(result.get("category")),
            "sentiment": {"score": score, "label": normalized_label},
            "credibility_score": credibility,
            "topics": self._clean_list(result.get("topics"), limit=8),
            "learning_points": self._clean_list(result.get("learning_points"), limit=6),
            "action_items": self._clean_list(result.get("action_items"), limit=6),
            "ocr_text": str(result.get("ocr_text", ""))[:2000],
            "video_transcript": str(result.get("video_transcript", ""))[:5000],
            "visual_description": str(result.get("visual_description", ""))[:2000],
            "video_summary": str(result.get("video_summary", ""))[:1000],
        }

    def _normalize_category(self, value: Optional[str]) -> str:
        """Map category variants to the controlled vocabulary."""
        text = str(value or "").strip()
        if not text:
            return "Other"
        for category in PRIMARY_CATEGORIES:
            if text.lower() == category.lower():
                return category
        alias = CATEGORY_ALIASES.get(text.lower())
        if alias:
            return alias
        lowered = text.lower().replace("&", "and")
        for alias_key, mapped in CATEGORY_ALIASES.items():
            if alias_key in lowered:
                return mapped
        return "Other"

    def _clean_list(self, value, limit: int = 8) -> List[str]:
        """Clean, dedupe, and cap list output."""
        if isinstance(value, list):
            items = [str(item).strip() for item in value if str(item).strip()]
        elif isinstance(value, str):
            items = [value.strip()] if value.strip() else []
        else:
            items = []

        deduped: List[str] = []
        seen = set()
        for item in items:
            key = item.lower()
            if key in seen:
                continue
            seen.add(key)
            deduped.append(item[:120])
            if len(deduped) >= limit:
                break
        return deduped

    def generate_psychometric_profile(self, posts_data: List[Dict]) -> Optional[Dict]:
        """Generate psychometric profile from analyzed posts."""
        if not posts_data:
            return None

        # Aggregate data
        categories = {}
        all_topics = []
        sentiment_labels = []

        for post in posts_data:
            cat = post.get("category", "Other")
            categories[cat] = categories.get(cat, 0) + 1
            all_topics.extend(post.get("topics", []))

            sentiment = post.get("sentiment", {})
            if sentiment:
                sentiment_labels.append(sentiment.get("label", "Neutral"))

        top_categories = sorted(categories.items(), key=lambda x: x[1], reverse=True)[
            :3
        ]

        prompt = f"""
Based on this user's Instagram saved content, generate a psychometric profile.

Data:
- Top Interests: {", ".join([c[0] for c in top_categories])}
- Common Topics: {", ".join(all_topics[:50])}
- Total Posts: {len(posts_data)}

Return ONLY valid JSON:
{{
    "archetype": "Title (e.g. The Curious Creator)",
    "one_liner": "Short punchy description",
    "traits": ["Trait 1", "Trait 2", "Trait 3"],
    "subconscious_motivations": "What drives them",
    "content_dna": {{"primary_focus": "Main", "secondary_focus": "Secondary"}},
    "growth_areas": ["Area 1", "Area 2"]
}}
"""

        try:
            if self.gemini_client is not None:
                response = self.gemini_client.models.generate_content(
                    model=self.gemini_model,
                    contents=prompt,
                    config={"response_mime_type": "application/json"},
                )
                content = getattr(response, "text", "") or "{}"
                return json.loads(content.strip())

            if not self.client:
                return None

            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert psychologist analyzing digital content.",
                    },
                    {"role": "user", "content": prompt},
                ],
                max_completion_tokens=500,
                response_format={"type": "json_object"},
            )

            content = response.choices[0].message.content or "{}"
            return json.loads(content.strip())

        except Exception as e:
            logger.error(f"Failed to generate profile: {e}")
            return None
