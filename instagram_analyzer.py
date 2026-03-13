#!/usr/bin/env python3
"""
Instagram Saved Posts AI Analyzer
Fetches saved posts from Instagram and analyzes them using Azure OpenAI
"""

import os
import json
import requests
import time
import random
from pathlib import Path
from dotenv import load_dotenv
from openai import AzureOpenAI
from datetime import datetime, timezone
import logging
from flask import Flask
from database import db, Post, Analysis, get_db_uri
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import numpy as np
import browser_cookie3
import tempfile
import subprocess


# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _first_non_empty(*values):
    for value in values:
        if value:
            return value
    return None

class InstagramAnalyzer:
    def __init__(self, config=None):
        # Always load environment variables first
        load_dotenv('.env.temp')
        
        # If no config provided, create from environment
        if not config:
            config = {
                'sessionid': os.getenv('INSTAGRAM_SESSIONID'),
                'azure_endpoint': _first_non_empty(
                    os.getenv('AZURE_OPENAI_API_BASE'),
                    os.getenv('AZURE_ENDPOINT')
                ),
                'azure_key': _first_non_empty(
                    os.getenv('AZURE_OPENAI_API_KEY'),
                    os.getenv('AZURE_KEY')
                ),
                'model': os.getenv('MODEL', 'DeepSeek-V3.2'),
                'max_posts': int(os.getenv('MAX_POSTS', 20)),
                'raw_cookie': os.getenv('RAW_COOKIE'),
                'user_agent': os.getenv('USER_AGENT'),
                'browser': 'none'
            }

        self.config = config
        self.sessionid = config.get('sessionid')
        self.raw_cookie = config.get('raw_cookie')
        self.browser_name = config.get('browser')
        self.user_agent = config.get('user_agent', 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:109.0) Gecko/20100101 Firefox/119.0')
        
        # Azure OpenAI setup
        self.azure_endpoint = _first_non_empty(
            config.get('azure_endpoint'),
            os.getenv('AZURE_OPENAI_API_BASE'),
            os.getenv('AZURE_ENDPOINT')
        )
        self.azure_key = _first_non_empty(
            config.get('azure_key'),
            os.getenv('AZURE_OPENAI_API_KEY'),
            os.getenv('AZURE_KEY')
        )
        self.model = config.get('model') or os.getenv('MODEL', 'DeepSeek-V3.2')
        self.max_posts = int(config.get('max_posts') or os.getenv('MAX_POSTS', 20))
        
        if self.azure_endpoint and self.azure_key:
            try:
                self.client = AzureOpenAI(
                    api_key=self.azure_key,
                    api_version="2024-02-15-preview",
                    azure_endpoint=self.azure_endpoint
                )
                logger.info("Azure OpenAI client initialized")
            except Exception as e:
                logger.error(f"Failed to initialize Azure OpenAI client: {e}")
                self.client = None
        else:
            logger.warning("Azure credentials missing - analysis will be disabled")
            self.client = None
        
        # Headers for Instagram requests
        self.headers = {
            'User-Agent': self.user_agent,
            'Accept': 'application/json, text/plain, */*',
            'Accept-Language': 'en-US,en;q=0.5',
            'X-IG-App-ID': '936619743392459',
            'X-Requested-With': 'XMLHttpRequest',
            'Origin': 'https://www.instagram.com',
            'Referer': 'https://www.instagram.com/saved/',
            'Sec-Fetch-Dest': 'empty',
            'Sec-Fetch-Mode': 'cors',
            'Sec-Fetch-Site': 'same-origin',
        }
        
        # If browser is specified, load cookies directly (CLI-style)
        if self.browser_name and self.browser_name != 'none':
            self.load_cookies_from_browser(self.browser_name)
        elif self.raw_cookie:
            self.headers['Cookie'] = self.raw_cookie
            logger.info('Loaded RAW_COOKIE into headers')
        elif self.sessionid:
            self.headers['Cookie'] = f'sessionid={self.sessionid}'
            logger.info(f"Set sessionid cookie: {str(self.sessionid)[:10]}...")

        # Initialize DB context
        self.app = Flask(__name__)
        self.app.config['SQLALCHEMY_DATABASE_URI'] = get_db_uri()
        self.app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
        db.init_app(self.app)

    @staticmethod
    def _clean_list(value):
        if isinstance(value, (list, tuple, set)):
            return [str(item).strip() for item in value if str(item).strip()]

        if isinstance(value, str):
            text = value.strip()
            if not text:
                return []
            try:
                parsed = json.loads(text)
                if isinstance(parsed, str):
                    parsed = json.loads(parsed)
                if isinstance(parsed, list):
                    return [str(item).strip() for item in parsed if str(item).strip()]
            except Exception:
                return [text]

        return []

    @staticmethod
    def _parse_timestamp(value):
        if isinstance(value, datetime):
            return value
        if isinstance(value, (int, float)):
            return datetime.fromtimestamp(value)
        if isinstance(value, str):
            try:
                return datetime.fromisoformat(value)
            except ValueError:
                return datetime.now(timezone.utc)
        return datetime.now(timezone.utc)

    @staticmethod
    def _normalize_sentiment(label, score):
        label_map = {
            'positive': 'Positive',
            'neutral': 'Neutral',
            'negative': 'Negative'
        }
        normalized = label_map.get(str(label or '').strip().lower())
        if normalized:
            return normalized

        try:
            numeric = float(score)
        except Exception:
            return 'Neutral'

        if numeric > 0.15:
            return 'Positive'
        if numeric < -0.15:
            return 'Negative'
        return 'Neutral'

    def _normalize_analysis_result(self, raw_result):
        if not isinstance(raw_result, dict):
            return None

        sentiment = raw_result.get('sentiment') or {}
        try:
            sentiment_score = float(sentiment.get('score', 0))
        except Exception:
            sentiment_score = 0.0
        sentiment_score = max(-1.0, min(1.0, sentiment_score))

        credibility_score = raw_result.get('credibility_score')
        try:
            credibility_score = int(credibility_score) if credibility_score is not None else None
        except Exception:
            credibility_score = None
        if credibility_score is not None:
            credibility_score = max(0, min(100, credibility_score))

        return {
            'category': str(raw_result.get('category') or 'Other')[:50],
            'sentiment': {
                'score': sentiment_score,
                'label': self._normalize_sentiment(sentiment.get('label'), sentiment_score)
            },
            'credibility_score': credibility_score,
            'topics': self._clean_list(raw_result.get('topics')),
            'learning_points': self._clean_list(raw_result.get('learning_points')),
            'action_items': self._clean_list(raw_result.get('action_items')),
            'ocr_text': raw_result.get('ocr_text') or '',
            'video_transcript': raw_result.get('video_transcript') or '',
            'visual_description': raw_result.get('visual_description') or '',
            'video_summary': raw_result.get('video_summary') or ''
        }

    def fetch_collections(self):
        """Fetch user's saved collections"""
        try:
            logger.info("Fetching saved collections...")
            # Use updated working endpoint
            url = 'https://www.instagram.com/api/v1/collections/list/'
            collections_map = {}
            next_max_id = None
            
            while True:
                params = {}
                if next_max_id:
                    params['max_id'] = next_max_id
                    
                response = requests.get(url, headers=self.headers, params=params, timeout=30)
                
                if response.status_code != 200:
                    logger.warning(f"Failed to fetch collections page: {response.status_code}")
                    # Keep what we have if a page fails
                    break
                    
                data = response.json()
                
                # The new endpoint returns collections in 'items' list
                for item in data.get('items', []):
                    # New structure uses collection_id and collection_name
                    c_id = item.get('collection_id') or item.get('id')
                    c_name = item.get('collection_name') or item.get('title')
                    if c_id and c_name:
                        collections_map[str(c_id)] = c_name
                
                next_max_id = data.get('next_max_id')
                if not data.get('more_available') and not next_max_id:
                    break
                    
                import time, random
                time.sleep(random.uniform(1, 3))
                        
            logger.info(f"Found {len(collections_map)} collections")
            return collections_map
            
        except Exception as e:
            logger.error(f"Error fetching collections: {e}")
            return {}

    def download_media(self, url, suffix='.mp4'):
        """Download media to a temporary file"""
        try:
            response = requests.get(url, stream=True, timeout=60)
            if response.status_code == 200:
                fd, path = tempfile.mkstemp(suffix=suffix)
                with os.fdopen(fd, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                return path
            return None
        except Exception as e:
            logger.error(f"Failed to download media: {e}")
            return None

    def extract_audio(self, video_path):
        """Extract audio from video using ffmpeg"""
        try:
            audio_path = video_path.replace('.mp4', '.mp3')
            # ffmpeg -i input.mp4 -q:a 0 -map a output.mp3 -y
            cmd = [
                'ffmpeg', '-i', video_path, 
                '-q:a', '0', '-map', 'a', 
                audio_path, '-y',
                '-loglevel', 'error'
            ]
            subprocess.run(cmd, check=True)
            if os.path.exists(audio_path):
                return audio_path
            return None
        except Exception as e:
            logger.error(f"Failed to extract audio: {e}")
            return None

    def transcribe_audio(self, audio_path):
        """Transcribe audio using OpenAI Whisper (via Azure or OpenAI)"""
        try:
            if not self.client: return None
            
            # Simple retry mechanism
            for _ in range(2):
                try:
                    with open(audio_path, "rb") as audio_file:
                        # Analyze using Azure OpenAI Whisper if available, or fallback/fail
                        # Azure OpenAI uses a deployment for Whisper
                        # For now assuming standard interface or checking if model is set
                        result = self.client.audio.transcriptions.create(
                            model="whisper", # This might need to be configurable
                            file=audio_file,
                        )
                    return result.text
                except Exception as e:
                    logger.warning(f"Transcription attempt failed: {e}")
                    time.sleep(2)
            return None
        except Exception as e:
            logger.error(f"Transcription failed: {e}")
            return None

    def load_cookies_from_browser(self, browser_name):
        """Load cookies directly from browser (CLI style)"""
        try:
            logger.info(f"Loading cookies directly from {browser_name}...")
            cj = None
            
            if browser_name.lower() == 'chrome':
                cj = browser_cookie3.chrome(domain_name='.instagram.com')
            elif browser_name.lower() == 'firefox':
                cj = browser_cookie3.firefox(domain_name='.instagram.com')
            elif browser_name.lower() == 'safari':
                cj = browser_cookie3.safari(domain_name='.instagram.com')
            elif browser_name.lower() == 'edge':
                cj = browser_cookie3.edge(domain_name='.instagram.com')
            
            if cj:
                cookie_parts = [f"{c.name}={c.value}" for c in cj]
                self.headers['Cookie'] = "; ".join(cookie_parts)
                logger.info(f"Successfully loaded {len(cookie_parts)} cookies from {browser_name}")
            else:
                logger.warning(f"No cookies found for {browser_name}")
                
        except Exception as e:
            logger.error(f"Failed to load browser cookies: {e}")

    def validate_session(self):
        """Validate that the current session/cookie works"""
        try:
            # Use the actual saved posts endpoint for validation
            url = 'https://www.instagram.com/api/v1/feed/saved/posts/'
            response = requests.get(url, headers=self.headers, timeout=10)
            
            # Log the response for debugging
            logger.info(f"Session validation: status={response.status_code}")
            
            # If we get HTML (login page), session is invalid
            if "<!DOCTYPE html>" in response.text or "<html" in response.text:
                logger.error("Session validation failed: Got HTML login page")
                return False, "Session expired or invalid (Instagram login page detected)"
            
            # Check if we can decode JSON
            try:
                data = response.json()
                # If we got JSON, validate it has the expected structure
                if 'items' in data or 'status' in data:
                    logger.info("Session validation passed")
                    return True, "Session valid"
                else:
                    logger.error(f"Session validation failed: Unexpected JSON structure: {list(data.keys())}")
                    return False, "Unexpected response from Instagram"
            except json.JSONDecodeError:
                logger.error("Session validation failed: Cannot decode JSON response")
                return False, "Invalid response from Instagram (not JSON)"
                
        except requests.exceptions.Timeout:
            logger.error("Session validation failed: Request timeout")
            return False, "Request timeout - Instagram might be down or blocking requests"
        except Exception as e:
            logger.error(f"Session validation failed with exception: {e}")
            return False, f"Validation error: {str(e)}"

    def get_saved_posts(self, limit=None, existing_ids=None):
        """Fetch saved posts from Instagram"""
        saved_posts = []
        next_max_id = None
        existing_ids = existing_ids or set()
        consecutive_existing_count = 0
        MAX_CONSECUTIVE_EXISTING = 5  # Stop after seeing 5 already-saved posts in a row
        
        # Use configured max_posts if no limit specified
        target_count = int(limit) if limit else 999999  # Essentially unlimited
        
        logger.info(f"Fetching saved posts{'...' if not limit else f' (limit: {target_count})...'}")
        
        while len(saved_posts) < target_count:
            try:
                # API endpoint for saved posts
                url = 'https://www.instagram.com/api/v1/feed/saved/posts/'
                params = {}
                if next_max_id:
                    params['max_id'] = next_max_id
                
                response = requests.get(
                    url, 
                    headers=self.headers, 
                    params=params,
                    timeout=30
                )
                
                if response.status_code != 200:
                    logger.error(f"Error fetching posts: {response.status_code} - {response.text[:200]}")
                    break
                    
                try:
                    data = response.json()
                except json.JSONDecodeError:
                    logger.error(f"Failed to decode JSON. Response text: {response.text[:500]}")
                    break
                    
                items = data.get('items', [])
                
                if not items:
                    break
                    
                for item in items:
                    post = self.extract_post_data(item)
                    if post and post.get('id'):
                        post_id = post.get('id')
                        
                        # Early stopping logic
                        if post_id in existing_ids:
                            consecutive_existing_count += 1
                            logger.debug(f"Found existing post {post_id}. Consecutive count: {consecutive_existing_count}")
                        else:
                            consecutive_existing_count = 0  # Reset if we find a new one
                        
                        saved_posts.append(post)
                        
                        if len(saved_posts) >= target_count:
                            break
                            
                        if consecutive_existing_count >= MAX_CONSECUTIVE_EXISTING and not limit:
                            logger.info(f"Hit existing posts threshold ({MAX_CONSECUTIVE_EXISTING}). Stopping fetch early.")
                            return saved_posts
                
                next_max_id = data.get('next_max_id')
                if not next_max_id:
                    break
                    
                # Be nice to the API
                time.sleep(random.uniform(1, 3))
                
            except Exception as e:
                logger.error(f"Exception fetching posts: {e}")
                break

        logger.info(f"Fetched {len(saved_posts)} saved posts")
        return saved_posts

    def extract_post_data(self, item):
        """Extract relevant data from Instagram post item"""
        try:
            # Extract post data
            # The structure is item -> media -> ...
            media = item.get('media', item)  # Handle both wrapped and unwrapped cases
            if not media:
                return None

            shortcode = media.get('code') or media.get('shortcode') or ''
            timestamp = datetime.fromtimestamp(media.get('taken_at', 0)).isoformat()
            thumbnail_url = ''
            media_url = ''
            video_url = ''

            # Extract image URL
            if 'image_versions2' in media:
                candidates = media['image_versions2'].get('candidates', [])
                if candidates:
                    thumbnail_url = candidates[0].get('url', '')
                    media_url = candidates[-1].get('url', thumbnail_url)

            # Extract video URL
            is_video = media.get('media_type') == 2
            if is_video and 'video_versions' in media:
                candidates = media.get('video_versions', [])
                if candidates:
                    video_url = candidates[0].get('url', '')
                    media_url = video_url or media_url
            
            post = {
                'id': media.get('id'),
                'shortcode': shortcode,
                'username': media.get('user', {}).get('username', 'unknown'),
                'timestamp': timestamp,
                'url': f"https://www.instagram.com/p/{shortcode}/" if shortcode else '',
                'likes': media.get('like_count', 0),
                'comments': media.get('comment_count', 0),
                'caption': media.get('caption', {}).get('text', '') if media.get('caption') else '',
                'media_type': media.get('media_type', 1),  # 1=photo, 2=video, 8=carousel
                'is_video': is_video,
                'thumbnail_url': thumbnail_url,
                'media_url': media_url,
                'video_url': video_url
            }
            
            # Handle carousel posts
            if media.get('media_type') == 8 and 'carousel_media' in media:
                post['carousel_count'] = len(media['carousel_media'])
                if not post['thumbnail_url']:
                    post['thumbnail_url'] = media['carousel_media'][0].get('image_versions2', {}).get('candidates', [{}])[0].get('url', '')
                if not post['media_url']:
                    post['media_url'] = post['thumbnail_url']

            # Store collection IDs if present
            post['saved_collection_ids'] = item.get('saved_collection_ids', []) or item.get('collection_ids', [])

            return post

        except Exception as e:
            logger.error(f"Error extracting post data: {e}")
            return None

    def analyze_post_with_ai(self, post):
        """Analyze a single post using Azure OpenAI with enhanced extraction"""
        try:
            if not self.client:
                logger.error("Azure OpenAI client not available")
                return None, 0

            # 1. Video Processing (Transcription)
            transcript = None
            if post['is_video'] and post.get('video_url'):
                logger.info("Processing video for transcription...")
                video_path = self.download_media(post['video_url'])
                if video_path:
                    audio_path = self.extract_audio(video_path)
                    if audio_path:
                        transcript = self.transcribe_audio(audio_path)
                        # Cleanup
                        try:
                            os.remove(audio_path)
                        except: pass
                    try:
                        os.remove(video_path)
                    except: pass
            
            # 2. Prepare Prompt
            prompt = f"""
Deeply analyze this Instagram content to create a high-value knowledge base entry.

Content Details:
- Type: {'Video' if post['is_video'] else 'Photo'}
- User: {post['username']}
- Caption: {post['caption']}
- Likes/Comments: {post['likes']} / {post['comments']}
- Timestamp: {post['timestamp']}
"""
            if transcript:
                prompt += f"\nVideo Transcript:\n{transcript}\n"
            
            prompt += """
Your Goal: Extract knowledge and actionable value.

Please provide analysis in this exact JSON format:
{
    "category": "one of: [Technology, Travel, Food, Fashion, Fitness, Art, Education, Entertaiment, Business, Personal, Other]",
    "sentiment": {
        "score": float -1 to 1,
        "label": "Positive/Negative/Neutral"
    },
    "credibility_score": integer 0-100,
    "topics": ["specific", "tags"],
    "learning_points": ["bullet", "points", "of", "core", "knowledge"],
    "action_items": ["Concrete actions the user can take", "e.g. 'Read book X', 'Visit restaurant Y', 'Try tool Z'"],
    "ocr_text": "Full text visible in the image/video frames (if any)",
    "visual_description": "Detailed description of the visual content",
    "video_summary": "Summary of what happens in the video (if applicable)"
}

Focus heavily on:
1. **Actionability**: What can the user DO with this?
2. **Knowledge Extraction**: Don't just describe, explain the value.
3. **Completeness**: If there is text in the image, transcribe it in ocr_text.
"""

            # Build user content list — image_url part is optional
            user_content: list = [{"type": "text", "text": prompt}]
            if post.get('thumbnail_url'):
                user_content.append({
                    "type": "image_url",
                    "image_url": {"url": post['thumbnail_url']}
                })

            messages: list = [
                {"role": "system", "content": "You are a sophisticated knowledge extraction AI. Turn social media noise into structured, actionable knowledge."},
                {"role": "user", "content": user_content}
            ]

            response = self.client.chat.completions.create(
                model=str(self.model),
                messages=messages,
                max_tokens=1500,
                temperature=0.3,
                response_format={"type": "json_object"}
            )

            analysis_text = (response.choices[0].message.content or "").strip()
            tokens_used = response.usage.total_tokens if response.usage else 0

            # Clean up response
            if analysis_text.startswith('```json'):
                analysis_text = analysis_text[7:]
            if analysis_text.endswith('```'):
                analysis_text = analysis_text[:-3]

            analysis_result = self._normalize_analysis_result(json.loads(analysis_text.strip()))
            if not analysis_result:
                return None, tokens_used
            
            # Inject transcript if we had one
            if transcript:
                analysis_result['video_transcript'] = transcript

            return analysis_result, tokens_used

        except Exception as e:
            logger.error(f"Error analyzing post {post.get('id', 'unknown')}: {e}")
            return None, 0

    def perform_lda_analysis(self, posts):
        """Perform LDA Topic Modeling on post captions"""
        try:
            if not posts:
                return []

            # Combine caption and topics for richer text
            documents = []
            for post in posts:
                # Handle both dict and object access if needed, but here we expect dicts from analyzed_posts
                text = post.get('caption', '')
                if 'analysis' in post and 'topics' in post['analysis']:
                    text += " " + " ".join(post['analysis']['topics'])
                documents.append(text)

            if not documents:
                return []

            # Vectorize
            vectorizer = CountVectorizer(max_df=0.95, min_df=2, stop_words='english')
            dtm = vectorizer.fit_transform(documents)
            
            if dtm.shape[1] == 0:
                return []

            # LDA
            lda = LatentDirichletAllocation(n_components=5, random_state=42)
            lda.fit(dtm)

            # Extract topics
            feature_names = vectorizer.get_feature_names_out()
            topics = []
            for topic_idx, topic in enumerate(lda.components_):
                top_words = [feature_names[i] for i in topic.argsort()[:-6:-1]]
                topics.append({
                    "id": topic_idx + 1,
                    "keywords": top_words
                })
            
            return topics
        except Exception as e:
            logger.error(f"LDA Analysis failed: {e}")
            return []

    def analyze_user_profile(self, posts, lda_topics):
        """Generate psychoanalysis of the user based on saved content"""
        try:
            if not posts:
                return {}

            if not self.client:
                logger.warning("OpenAI client not available; skipping psychoanalysis")
                return {}

            # Prepare summary for AI
            categories = {}
            all_topics = []
            for post in posts:
                cat = post['analysis']['category']
                categories[cat] = categories.get(cat, 0) + 1
                all_topics.extend(post['analysis']['topics'])

            prompt = f"""
Based on this user's saved Instagram content, provide a psychoanalysis profile.

Data:
- Total Saved Posts: {len(posts)}
- Top Categories: {json.dumps(categories)}
- Key Topics: {", ".join(all_topics[:50])}
- LDA Latent Themes: {json.dumps(lda_topics)}

Provide a JSON profile with:
{{
    "personality_traits": ["trait1", "trait2", "trait3"],
    "interests": ["deep interest 1", "interest 2"],
    "suggested_hobbies": ["hobby1", "hobby2"],
    "career_potential": "Based on these interests, they might enjoy...",
    "psychological_summary": "A brief paragraph describing their content consumption personality."
}}
"""
            response = self.client.chat.completions.create(
                model=str(self.model),
                messages=[
                    {"role": "system", "content": "You are an expert psychologist and data analyst. Provide deep insights in JSON format."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=500,
                temperature=0.7
            )

            raw_content = response.choices[0].message.content or ""
            content = raw_content.strip()
            if content.startswith('```json'): content = content[7:]
            if content.endswith('```'): content = content[:-3]
            return json.loads(content.strip())

        except Exception as e:
            logger.error(f"Psychoanalysis failed: {e}")
            return {}

    def generate_summary(self, analyzed_posts):
        """Generate summary statistics from analyzed posts"""
        if not analyzed_posts:
            return {}

        total_posts = len(analyzed_posts)
        total_tokens = sum(post.get('tokens_used', 0) for post in analyzed_posts)

        # Calculate average sentiment
        sentiments = [post['analysis']['sentiment']['score'] for post in analyzed_posts]
        avg_sentiment = sum(sentiments) / len(sentiments) if sentiments else 0

        # Category breakdown
        categories = {}
        for post in analyzed_posts:
            cat = post['analysis']['category']
            categories[cat] = categories.get(cat, 0) + 1

        # Sentiment distribution
        sentiment_dist = {'positive': 0, 'neutral': 0, 'negative': 0}
        for post in analyzed_posts:
            label = post['analysis']['sentiment']['label'].lower()
            if label in sentiment_dist:
                sentiment_dist[label] += 1

        # Top topics
        all_topics = []
        for post in analyzed_posts:
            all_topics.extend(post['analysis']['topics'])

        topic_counts = {}
        for topic in all_topics:
            topic_counts[topic] = topic_counts.get(topic, 0) + 1

        top_topics = sorted(topic_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        top_topics = [topic for topic, count in top_topics]

        # Estimate cost (rough calculation for gpt-4o-mini)
        cost_per_1k_tokens = 0.00015  # Approximate for gpt-4o-mini
        total_cost = (total_tokens / 1000) * cost_per_1k_tokens

        # Advanced Analysis
        logger.info("Performing LDA Topic Modeling...")
        lda_topics = self.perform_lda_analysis(analyzed_posts)
        
        logger.info("Generating Psychoanalysis Profile...")
        user_profile = self.analyze_user_profile(analyzed_posts, lda_topics)

        return {
            'total_posts': total_posts,
            'avg_sentiment': round(avg_sentiment, 2),
            'categories': categories,
            'sentiment_dist': sentiment_dist,
            'top_topics': top_topics,
            'lda_topics': lda_topics,
            'user_profile': user_profile,
            'total_tokens': total_tokens,
            'estimated_cost': round(total_cost, 4)
        }



    def fetch_and_sync_posts(self, limit=None, progress_callback=None):
        """Fetch posts from Instagram and sync to DB (no analysis)"""
        try:
            logger.info("Syncing Instagram saved posts...")
            
            # Validate session first
            is_valid, msg = self.validate_session()
            if not is_valid:
                raise Exception(f"Instagram Connection Failed: {msg}")
                
            if progress_callback: progress_callback(10, "Fetching saved posts...")

            # Fetch collections first
            collections_map = self.fetch_collections()
            
            new_count = 0
            with self.app.app_context():
                db.create_all()
                
                # Get existing post IDs for early stopping
                existing_records = db.session.query(Post.id).all()
                existing_ids = {r[0] for r in existing_records}
                
                saved_posts = self.get_saved_posts(limit, existing_ids=existing_ids)
                
                if not saved_posts:
                    return {'count': 0, 'new': 0}
                
                for i, post_data in enumerate(saved_posts):
                    if progress_callback:
                        percent = 10 + int((i / len(saved_posts)) * 80)
                        progress_callback(percent, f"Syncing post {i+1}/{len(saved_posts)}...")

                    post_id = post_data.get('id')
                    if not post_id:
                        continue

                    existing_post = Post.query.get(post_id)
                    
                    # Extract collections for this post
                    post_collections = []
                    saved_ids = post_data.get('saved_collection_ids', [])
                    for c_id in saved_ids:
                        key = str(c_id)
                        if key in collections_map:
                            post_collections.append(collections_map[key])
                    
                    if not existing_post:
                        timestamp = self._parse_timestamp(post_data.get('timestamp'))
                        new_post = Post(
                            id=post_id,
                            shortcode=post_data.get('shortcode'),
                            username=post_data.get('username', 'unknown'),
                            caption=post_data.get('caption') or '',
                            timestamp=timestamp,
                            thumbnail_url=post_data.get('thumbnail_url') or '',
                            media_url=post_data.get('media_url') or post_data.get('thumbnail_url') or '',
                            media_type=post_data.get('media_type', 1),
                            is_video=bool(post_data.get('is_video')),
                            likes=post_data.get('likes', 0) or 0,
                            comments=post_data.get('comments', 0) or 0,
                            collections=json.dumps(post_collections)
                        )
                        db.session.add(new_post)
                        new_count += 1
                    else:
                        existing_post.shortcode = post_data.get('shortcode') or existing_post.shortcode
                        existing_post.username = post_data.get('username', existing_post.username)
                        existing_post.caption = post_data.get('caption') or existing_post.caption
                        existing_post.timestamp = self._parse_timestamp(post_data.get('timestamp'))
                        existing_post.thumbnail_url = post_data.get('thumbnail_url') or existing_post.thumbnail_url
                        existing_post.media_url = post_data.get('media_url') or existing_post.media_url
                        existing_post.media_type = post_data.get('media_type', existing_post.media_type)
                        existing_post.is_video = bool(post_data.get('is_video', existing_post.is_video))
                        existing_post.likes = post_data.get('likes', existing_post.likes) or 0
                        existing_post.comments = post_data.get('comments', existing_post.comments) or 0
                        # Update collections for existing posts if changed
                        existing_post.collections = json.dumps(post_collections)
                
                db.session.commit()
                
            if progress_callback: progress_callback(100, "Sync complete!")
            return {'count': len(saved_posts), 'new': new_count}
            
        except Exception as e:
            logger.error(f"Sync failed: {e}")
            raise e

    def analyze_batch(self, batch_size=None, progress_callback=None):
        """Analyze unanalyzed posts from DB in parallel"""
        analyzed_posts = []
        total_tokens = 0
        
        # Import here to avoid circular dependency if any
        from concurrent.futures import ThreadPoolExecutor, as_completed
        
        with self.app.app_context():
            # Get unanalyzed posts
            query = Post.query.filter(Post.analysis == None)
            if batch_size:
                query = query.limit(batch_size)
            
            posts_to_analyze = query.all()
            
            if not posts_to_analyze:
                return {'analyzed_count': 0, 'message': 'No unanalyzed posts found'}

            logger.info(f"Starting parallel analysis for {len(posts_to_analyze)} posts")
            
            # Prepare data for parallel processing (detach from DB session)
            post_data_list = [{'post': p.to_dict(), 'db_id': p.id} for p in posts_to_analyze]
            
            # Function to run in thread
            def analyze_single(item):
                result, tokens = self.analyze_post_with_ai(item['post'])
                return {'db_id': item['db_id'], 'result': result, 'tokens': tokens, 'post_data': item['post']}

            # Run in parallel
            max_workers = 5 # Limit concurrency to avoid rate limits
            completed_count = 0
            
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_post = {executor.submit(analyze_single, item): item for item in post_data_list}
                
                for future in as_completed(future_to_post):
                    completed_count += 1
                    if progress_callback:
                        percent = int((completed_count / len(post_data_list)) * 100)
                        progress_callback(percent, f"Analyzing {completed_count}/{len(post_data_list)}...")
                    
                    try:
                        data = future.result()
                        analysis_result = data['result']
                        tokens = data['tokens']
                        db_id = data['db_id']
                        post_dict = data['post_data']
                        
                        total_tokens += tokens
                        
                        if analysis_result:
                            normalized = self._normalize_analysis_result(analysis_result)
                            if not normalized:
                                continue

                            # Save to DB (must be done in main thread with app context)
                            analysis = Analysis(
                                post_id=db_id,
                                category=normalized['category'],
                                sentiment_score=normalized['sentiment']['score'],
                                sentiment_label=normalized['sentiment']['label'],
                                credibility_score=normalized.get('credibility_score'),
                                topics=normalized['topics'],
                                learning_points=normalized.get('learning_points', []),
                                action_items=normalized.get('action_items', []),
                                raw_analysis=json.dumps(normalized, ensure_ascii=False),
                                ocr_text=normalized.get('ocr_text'),
                                video_transcript=normalized.get('video_transcript'),
                                visual_description=normalized.get('visual_description')
                            )
                            db.session.add(analysis)
                            db.session.commit()
                            
                            post_dict['analysis'] = normalized
                            post_dict['tokens_used'] = tokens
                            analyzed_posts.append(post_dict)
                            
                    except Exception as e:
                        logger.error(f"Parallel analysis failed for a post: {e}")

        # Generate summary
        summary = self.generate_summary(analyzed_posts)
        
        return {
            'analyzed_count': len(analyzed_posts),
            'total_tokens': total_tokens,
            'summary': summary
        }

    def generate_psychometric_profile(self, posts_data):
        """Generate a psychometric profile based on analyzed posts"""
        try:
            # Aggregate data for the prompt
            categories = {}
            all_topics = []
            sentiment_labels = []
            
            for p in posts_data:
                # Count categories
                cat = p.get('category', 'Other')
                categories[cat] = categories.get(cat, 0) + 1
                
                # Collect topics
                all_topics.extend(p.get('topics', []))
                
                # Collect sentiment
                if 'sentiment' in p:
                    sentiment_labels.append(p['sentiment'].get('label', 'Neutral'))

            # Top 3 categories
            top_categories = sorted(categories.items(), key=lambda x: x[1], reverse=True)[:3]
            
            # Simple prompt construction
            prompt = f"""
            Based on this user's Instagram saved content, generate a 'Psychometric Profile' and 'Digital Persona'.
            
            Data:
            - Top Interests: {', '.join([c[0] for c in top_categories])}
            - Common Topics: {', '.join(all_topics[:50])}
            - Sentiment Vibe: {max(set(sentiment_labels), key=sentiment_labels.count) if sentiment_labels else 'Unknown'}
            - Total Posts Saved: {len(posts_data)}

            Return ONLY a valid JSON object with this structure:
            {{
                "archetype": "Title (e.g. The Curious Creator)",
                "one_liner": "A short, punchy description of who they are.",
                "traits": ["Trait 1", "Trait 2", "Trait 3", "Trait 4"],
                "subconscious_motivations": "What drives them? (1-2 sentences)",
                "content_dna": {{
                    "primary_focus": "Main interest",
                    "secondary_focus": "Secondary interest"
                }},
                "growth_areas": ["Area 1", "Area 2"]
            }}
            """

            if not self.client:
                logger.warning("OpenAI client not available; skipping psychometric profile")
                return None

            response = self.client.chat.completions.create(
                model=str(self.model),
                messages=[
                    {"role": "system", "content": "You are a world-class psychologist and data scientist. Analyze the user's content diet to reveal their digital persona."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=500,
                temperature=0.7,
                response_format={ "type": "json_object" }
            )

            return json.loads(response.choices[0].message.content or "{}")

        except Exception as e:
            logger.error(f"Error generating profile: {e}")
            return None

    def run_analysis(self, progress_callback=None):
        """Main analysis workflow"""
        try:
            logger.info("Starting Instagram saved posts analysis...")
            if progress_callback:
                progress_callback(5, "Validating Instagram session...")

            is_valid, msg = self.validate_session()
            if not is_valid:
                raise Exception(f"Instagram Connection Failed: {msg}")

            if progress_callback:
                progress_callback(10, "Fetching saved posts...")

            saved_posts = self.get_saved_posts(limit=self.max_posts)

            if not saved_posts:
                raise Exception("No saved posts found. Check Instagram credentials.")

            analyzed_posts = []
            total_tokens = 0
            
            with self.app.app_context():
                db.create_all()
                total_posts = len(saved_posts)

                for i, post in enumerate(saved_posts, start=1):
                    post_id = post.get('id')
                    if not post_id:
                        continue

                    if progress_callback:
                        percent = 15 + int((i / total_posts) * 80)
                        progress_callback(percent, f"Analyzing post {i}/{total_posts}...")

                    try:
                        existing_post = Post.query.get(post_id)

                        if not existing_post:
                            existing_post = Post(
                                id=post_id,
                                shortcode=post.get('shortcode'),
                                username=post.get('username', 'unknown'),
                                timestamp=self._parse_timestamp(post.get('timestamp')),
                                thumbnail_url=post.get('thumbnail_url') or '',
                                media_url=post.get('media_url') or post.get('thumbnail_url') or '',
                                caption=post.get('caption') or '',
                                media_type=post.get('media_type', 1),
                                is_video=bool(post.get('is_video')),
                                likes=post.get('likes', 0) or 0,
                                comments=post.get('comments', 0) or 0,
                                collections=json.dumps([], ensure_ascii=False)
                            )
                            db.session.add(existing_post)
                        else:
                            existing_post.shortcode = post.get('shortcode') or existing_post.shortcode
                            existing_post.username = post.get('username', existing_post.username)
                            existing_post.timestamp = self._parse_timestamp(post.get('timestamp'))
                            existing_post.thumbnail_url = post.get('thumbnail_url') or existing_post.thumbnail_url
                            existing_post.media_url = post.get('media_url') or existing_post.media_url
                            existing_post.caption = post.get('caption') or existing_post.caption
                            existing_post.media_type = post.get('media_type', existing_post.media_type)
                            existing_post.is_video = bool(post.get('is_video', existing_post.is_video))
                            existing_post.likes = post.get('likes', existing_post.likes) or 0
                            existing_post.comments = post.get('comments', existing_post.comments) or 0

                        if existing_post.analysis:
                            analyzed_posts.append(existing_post.to_dict())
                            db.session.commit()
                            continue

                        analysis_result, tokens_used = self.analyze_post_with_ai(post)
                        total_tokens += tokens_used
                        if not analysis_result:
                            db.session.commit()
                            continue

                        normalized = self._normalize_analysis_result(analysis_result)
                        if not normalized:
                            db.session.commit()
                            continue

                        new_analysis = Analysis(
                            post_id=post_id,
                            category=normalized['category'],
                            sentiment_score=normalized['sentiment']['score'],
                            sentiment_label=normalized['sentiment']['label'],
                            credibility_score=normalized.get('credibility_score'),
                            topics=normalized.get('topics', []),
                            learning_points=normalized.get('learning_points', []),
                            action_items=normalized.get('action_items', []),
                            raw_analysis=json.dumps(normalized, ensure_ascii=False),
                            ocr_text=normalized.get('ocr_text'),
                            video_transcript=normalized.get('video_transcript'),
                            visual_description=normalized.get('visual_description')
                        )
                        db.session.add(new_analysis)
                        db.session.commit()

                        result_post = post.copy()
                        result_post['analysis'] = normalized
                        result_post['tokens_used'] = tokens_used
                        analyzed_posts.append(result_post)

                        time.sleep(1)
                    except Exception as e:
                        logger.error(f"Error processing post {post_id}: {e}")
                        db.session.rollback()

            summary = self.generate_summary(analyzed_posts)
            summary['total_tokens'] = total_tokens

            results = {
                'analyzed_posts': analyzed_posts,
                'summary': summary,
                'timestamp': datetime.now().isoformat(),
                'model_used': self.model
            }

            output_file = Path('output/analyzed_results.json')
            output_file.parent.mkdir(exist_ok=True)

            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)

            logger.info(f"Analysis complete! Results saved to {output_file}")
            logger.info(f"Analyzed {len(analyzed_posts)} posts, used approximately {total_tokens:.0f} tokens")
            if progress_callback:
                progress_callback(100, "Analysis complete!")

        except Exception as e:
            logger.error(f"Analysis failed: {e}")
            raise

def main():
    """Main entry point"""
    analyzer = InstagramAnalyzer()
    analyzer.run_analysis()

if __name__ == '__main__':
    main()
