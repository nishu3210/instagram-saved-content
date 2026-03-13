"""AI analysis service for Instagram posts."""

import json
import logging
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import requests
from openai import AzureOpenAI

from config import config

logger = logging.getLogger(__name__)


class AnalysisError(Exception):
    """Analysis error."""

    pass


class AIAnalyzer:
    """AI-powered content analyzer."""

    def __init__(self):
        self.client: Optional[AzureOpenAI] = None
        self.model = config.azure.model
        self._init_client()

    def _init_client(self) -> None:
        """Initialize Azure OpenAI client."""
        if not config.azure.is_configured():
            logger.warning("Azure credentials not configured")
            return

        try:
            self.client = AzureOpenAI(
                api_key=config.azure.api_key,
                api_version=config.azure.api_version,
                azure_endpoint=config.azure.endpoint,
            )
            logger.info("Azure OpenAI client initialized")
        except Exception as e:
            logger.error(f"Failed to initialize Azure client: {e}")

    def is_available(self) -> bool:
        """Check if analyzer is available."""
        return self.client is not None

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
                    import time

                    time.sleep(2)

        return None

    def analyze_post(self, post: Dict) -> Tuple[Optional[Dict], int]:
        """Analyze a single post using AI."""
        if not self.client:
            raise AnalysisError("Azure OpenAI client not available")

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
                max_tokens=1500,
                temperature=0.3,
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
Provide analysis in this exact JSON format:
{
    "category": "one of: [Technology, Travel, Food, Fashion, Fitness, Art, Education, Entertainment, Business, Personal, Other]",
    "sentiment": {"score": float -1 to 1, "label": "Positive/Negative/Neutral"},
    "credibility_score": integer 0-100,
    "topics": ["specific", "tags"],
    "learning_points": ["bullet", "points"],
    "action_items": ["Concrete actions"],
    "ocr_text": "Text visible in image/video",
    "visual_description": "Detailed visual description",
    "video_summary": "Video summary if applicable"
}

Focus on: 1) Actionability 2) Knowledge extraction 3) Completeness
"""
        return prompt

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

        def clean_list(value) -> List[str]:
            if isinstance(value, list):
                return [str(item).strip() for item in value if str(item).strip()]
            if isinstance(value, str):
                return [value.strip()] if value.strip() else []
            return []

        return {
            "category": str(result.get("category", "Other"))[:50],
            "sentiment": {"score": score, "label": normalized_label},
            "credibility_score": credibility,
            "topics": clean_list(result.get("topics")),
            "learning_points": clean_list(result.get("learning_points")),
            "action_items": clean_list(result.get("action_items")),
            "ocr_text": str(result.get("ocr_text", ""))[:2000],
            "video_transcript": str(result.get("video_transcript", ""))[:5000],
            "visual_description": str(result.get("visual_description", ""))[:2000],
            "video_summary": str(result.get("video_summary", ""))[:1000],
        }

    def generate_psychometric_profile(self, posts_data: List[Dict]) -> Optional[Dict]:
        """Generate psychometric profile from analyzed posts."""
        if not self.client or not posts_data:
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
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert psychologist analyzing digital content.",
                    },
                    {"role": "user", "content": prompt},
                ],
                max_tokens=500,
                temperature=0.7,
                response_format={"type": "json_object"},
            )

            content = response.choices[0].message.content or "{}"
            return json.loads(content.strip())

        except Exception as e:
            logger.error(f"Failed to generate profile: {e}")
            return None
