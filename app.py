"""Refactored Flask backend with compatibility for the existing UI."""

import hashlib
import json
import logging
import threading
import uuid
import warnings
from datetime import datetime, timedelta, timezone
from io import StringIO
from pathlib import Path
from typing import Any, Dict, List, Optional

import browser_cookie3
import pandas as pd
import requests
from flask import Flask, Response, abort, jsonify, request, send_file
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from flask_talisman import Talisman
from sqlalchemy import or_

from config import config
from database import (
    ActionTask,
    Analysis,
    Conversation,
    Message,
    Post,
    PostVerification,
    WorkspaceProfile,
    db,
    get_db_uri,
)
from migrations import run_migrations
from schemas import (
    AnalysisRequest,
    BatchAnalysisRequest,
    ChatRequest,
    InstagramAuthRequest,
    ProfileSettingsRequest,
    PostsFilterRequest,
    TaskBootstrapRequest,
    TaskCreateRequest,
    TaskUpdateRequest,
    TasksFilterRequest,
    VerificationRequest,
)
from services.ai_analyzer import AIAnalyzer
from services.instagram_client import (
    InstagramAPIError,
    InstagramAuthError,
    InstagramClient,
)
from services.planner_service import PlannerService
from services.rag_service import RAGService
from services.verification_service import VerificationService, VerificationServiceError
from status_manager import analysis_status, task_job_status

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Create Flask app
app = Flask(__name__)
app.config["SQLALCHEMY_DATABASE_URI"] = get_db_uri()
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
app.config["SECRET_KEY"] = config.app.secret_key

# Initialize extensions
db.init_app(app)

# Security headers
content_security_policy = {
    "default-src": ["'self'"],
    "script-src": ["'self'", "'unsafe-inline'", "https://cdn.jsdelivr.net", "https://cdn.tailwindcss.com"],
    "style-src": ["'self'", "'unsafe-inline'", "https://fonts.googleapis.com"],
    "font-src": ["'self'", "https://fonts.gstatic.com", "data:"],
    "img-src": ["'self'", "data:", "blob:", "https:"],
    "connect-src": ["'self'", "https:"],
    "frame-ancestors": ["'self'"],
    "object-src": ["'none'"],
}

Talisman(app, force_https=False, content_security_policy=content_security_policy)

# Rate limiting
if config.app.rate_limit_storage_uri == "memory://":
    warnings.filterwarnings(
        "ignore",
        message="Using the in-memory storage for tracking rate limits",
        category=UserWarning,
        module="flask_limiter",
    )

limiter = Limiter(
    app=app,
    key_func=get_remote_address,
    default_limits=["200 per day", "50 per hour"],
    storage_uri=config.app.rate_limit_storage_uri,
)

# CORS
CORS(
    app,
    resources={
        r"/api/*": {
            "origins": [
                "http://localhost:5000",
                "http://127.0.0.1:5000",
                "http://localhost:5001",
                "http://127.0.0.1:5001",
            ],
            "methods": ["GET", "POST", "OPTIONS", "PATCH"],
            "allow_headers": ["Content-Type", "Authorization"],
        }
    },
)

# Initialize services
rag_service = RAGService()
ai_analyzer = AIAnalyzer()
verification_service = VerificationService()
planner_service = PlannerService()
frontend_index_file = Path(app.static_folder or "static") / "app" / "index.html"

# Auth storage (per-session)
auth_storage = threading.local()


def get_auth() -> Dict[str, str]:
    """Get stored auth credentials."""
    return getattr(auth_storage, "auth", {})


def set_auth(data: Dict[str, str]) -> None:
    """Store auth credentials."""
    auth_storage.auth = data


def parse_iso_datetime(value: Optional[str]) -> Optional[datetime]:
    """Parse ISO datetime strings safely."""
    if not value:
        return None
    if isinstance(value, datetime):
        return value
    try:
        text = str(value).strip()
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        return datetime.fromisoformat(text)
    except (TypeError, ValueError):
        return None


def normalize_status(value: Optional[str]) -> str:
    """Normalize task status."""
    status = str(value or "pending").strip().lower()
    return status if status in {"pending", "in_progress", "done", "archived"} else "pending"


def normalize_priority(value: Optional[int]) -> int:
    """Normalize task priority."""
    try:
        priority = int(value)
    except (TypeError, ValueError):
        return 2
    return max(1, min(3, priority))


def normalize_effort(value: Optional[str]) -> str:
    """Normalize structured task effort."""
    effort = str(value or "medium").strip().lower()
    return effort if effort in {"quick", "medium", "deep"} else "medium"


def normalize_impact(value: Optional[str]) -> str:
    """Normalize structured task impact."""
    impact = str(value or "medium").strip().lower()
    return impact if impact in {"low", "medium", "high"} else "medium"


def normalize_horizon(value: Optional[str]) -> str:
    """Normalize structured task horizon."""
    horizon = str(value or "this_week").strip().lower()
    return horizon if horizon in {"today", "this_week", "later"} else "this_week"


def coerce_utc_datetime(value: Optional[datetime]) -> Optional[datetime]:
    """Treat naive datetimes as UTC for consistent comparisons."""
    if value is None:
        return None
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def serialize_status(manager) -> Dict[str, Any]:
    """Serialize thread-safe status manager output."""
    status = manager.get_status()
    return {
        "running": status.running,
        "progress": status.progress,
        "message": status.message,
        "error": status.error,
        "results": status.results,
    }


def build_ai_analyzer(data: Optional[Dict[str, Any]] = None) -> AIAnalyzer:
    """Build analyzer using request credentials when provided."""
    data = data or {}
    return AIAnalyzer(
        azure_endpoint=data.get("azure_endpoint") or config.azure.endpoint,
        azure_key=data.get("azure_key") or config.azure.api_key,
        model=data.get("model") or config.azure.model,
        api_version=config.azure.api_version,
        gemini_api_key=data.get("gemini_api_key") or config.gemini.api_key,
        gemini_model=data.get("gemini_model") or config.gemini.model,
    )


def prepare_raw_analysis_payload(result: Dict[str, Any], tokens_used: int) -> str:
    """Store normalized analysis plus token usage."""
    payload = dict(result)
    payload["tokens_used"] = tokens_used
    return json.dumps(payload, ensure_ascii=False)


def save_analysis_for_post(post: Post, result: Dict[str, Any], tokens_used: int) -> Analysis:
    """Create or update the structured analysis for a post."""
    analysis = post.analysis or Analysis(post_id=post.id)
    analysis.category = result["category"]
    analysis.sentiment_score = result["sentiment"]["score"]
    analysis.sentiment_label = result["sentiment"]["label"]
    analysis.credibility_score = result.get("credibility_score")
    analysis.topics = result.get("topics", [])
    analysis.learning_points = result.get("learning_points", [])
    analysis.action_items = result.get("action_items", [])
    analysis.raw_analysis = prepare_raw_analysis_payload(result, tokens_used)
    analysis.ocr_text = result.get("ocr_text")
    analysis.video_transcript = result.get("video_transcript")
    analysis.visual_description = result.get("visual_description")
    analysis.video_summary = result.get("video_summary")
    db.session.add(analysis)
    return analysis


def analyze_existing_post(post: Post, data: Optional[Dict[str, Any]] = None) -> Analysis:
    """Analyze an existing stored post when verification or batch flows need it."""
    if post.analysis:
        return post.analysis

    analyzer = build_ai_analyzer(data)
    if not analyzer.is_available():
        raise ValueError("No analysis model credentials are configured")

    result, tokens = analyzer.analyze_post(post.to_dict())
    if not result:
        raise ValueError("Analysis failed for this post")

    analysis = save_analysis_for_post(post, result, tokens)
    db.session.commit()
    rag_service.index_posts([post])
    return analysis


def build_system_overview() -> Dict[str, Any]:
    """Summarize analysis and verification health from the database."""
    total_posts = Post.query.count()
    analysis_count = Analysis.query.count()
    verification_count = PostVerification.query.filter_by(status="completed").count()
    rag_stats = rag_service.get_stats()
    index_vector_count = int(rag_stats.get("index_vector_count", rag_stats.get("total_indexed", 0)) or 0)
    index_metadata_count = int(rag_stats.get("index_metadata_count", 0) or 0)
    placeholder_post_count = Post.query.filter(Post.caption.like("Post by %")).count()
    integrity_warnings: List[str] = []
    if rag_stats.get("index_integrity_warning"):
        integrity_warnings.append(str(rag_stats["index_integrity_warning"]))
    if index_metadata_count != analysis_count:
        integrity_warnings.append(
            "Indexed metadata count does not match analyzed post rows in the database."
        )

    stale_data_detected = bool(index_vector_count > 0 and analysis_count == 0) or bool(
        placeholder_post_count and analysis_count == 0
    )
    stale_data_warning = None
    if stale_data_detected:
        stale_data_warning = (
            "Persisted RAG/index artifacts exist without matching structured analysis rows. "
            "Use fresh analysis or rebuild the index before trusting profile, chat, or task generation."
        )
        integrity_warnings.append(stale_data_warning)

    verification_coverage = (
        round((verification_count / analysis_count) * 100, 1) if analysis_count else 0.0
    )
    index_integrity_warning = " ".join(dict.fromkeys(integrity_warnings)) if integrity_warnings else None

    return {
        "total_posts": total_posts,
        "analysis_count": analysis_count,
        "analyzed_count": analysis_count,
        "verification_count": verification_count,
        "verification_coverage": verification_coverage,
        "embedded_count": index_vector_count,
        "index_vector_count": index_vector_count,
        "index_metadata_count": index_metadata_count,
        "index_integrity_status": "warning" if index_integrity_warning else "ok",
        "index_integrity_warning": index_integrity_warning,
        "placeholder_post_count": placeholder_post_count,
        "stale_data_detected": stale_data_detected,
        "stale_data_warning": stale_data_warning,
    }


def get_recent_analyzed_posts(limit: int = 30) -> List[Post]:
    """Get analyzed posts from the last 30 days, with a recent fallback."""
    threshold = datetime.now(timezone.utc) - timedelta(days=30)
    recent_posts = (
        Post.query.join(Analysis)
        .filter(Post.timestamp != None, Post.timestamp >= threshold)
        .order_by(Post.timestamp.desc())
        .limit(limit)
        .all()
    )
    if recent_posts:
        return recent_posts
    return (
        Post.query.join(Analysis)
        .order_by(Post.timestamp.is_(None), Post.timestamp.desc(), Post.created_at.desc())
        .limit(limit)
        .all()
    )


def get_workspace_profile() -> WorkspaceProfile:
    """Load or create the singleton workspace profile."""
    profile = WorkspaceProfile.get_singleton()
    db.session.flush()
    return profile


def refresh_workspace_profile_snapshot(
    data: Optional[Dict[str, Any]] = None,
) -> WorkspaceProfile:
    """Regenerate and persist the psychometric profile snapshot."""
    posts_data = []
    for post in Post.query.join(Analysis).all():
        analysis = post.analysis.to_dict() if post.analysis else {}
        posts_data.append(analysis)

    if not posts_data:
        raise ValueError("No analyzed posts found")

    profile_data = build_ai_analyzer(data).generate_psychometric_profile(posts_data)
    if not profile_data:
        raise ValueError("Failed to generate profile")

    workspace_profile = get_workspace_profile()
    workspace_profile.psychometric_profile = profile_data
    workspace_profile.profile_refreshed_at = datetime.now(timezone.utc)
    db.session.add(workspace_profile)
    db.session.commit()
    return workspace_profile


def build_analysis_results() -> Dict[str, Any]:
    """Build analysis results payload from the database."""
    posts = Post.query.order_by(Post.timestamp.desc()).all()
    analyzed_posts: List[Dict[str, Any]] = []
    total_tokens = 0
    categories: Dict[str, int] = {}
    sentiments = {"positive": 0, "neutral": 0, "negative": 0}
    topic_counts: Dict[str, int] = {}
    sentiment_scores: List[float] = []

    for post in posts:
        if not post.analysis:
            continue
        post_dict = post.to_dict()
        analyzed_posts.append(post_dict)

        analysis = post.analysis
        try:
            raw_analysis = json.loads(analysis.raw_analysis) if analysis.raw_analysis else {}
        except (TypeError, ValueError, json.JSONDecodeError):
            raw_analysis = {}
        total_tokens += int(raw_analysis.get("tokens_used", 0) or 0)
        category = analysis.category or "Other"
        categories[category] = categories.get(category, 0) + 1

        label = str(analysis.sentiment_label or "Neutral").strip().lower()
        if label not in sentiments:
            label = "neutral"
        sentiments[label] += 1
        sentiment_scores.append(float(analysis.sentiment_score or 0.0))

        for topic in analysis.topics or []:
            topic_counts[topic] = topic_counts.get(topic, 0) + 1

    avg_sentiment = sum(sentiment_scores) / len(sentiment_scores) if sentiment_scores else 0.0
    top_topics = [
        topic for topic, _ in sorted(topic_counts.items(), key=lambda item: item[1], reverse=True)[:10]
    ]

    return {
        "analyzed_posts": analyzed_posts,
        "summary": {
            "total_posts": len(analyzed_posts),
            "avg_sentiment": round(avg_sentiment, 2),
            "categories": categories,
            "sentiment_dist": sentiments,
            "top_topics": top_topics,
            "total_tokens": total_tokens,
        },
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model_used": config.azure.model,
    }


def suggested_due_date_for_horizon(horizon: str, due_days: int) -> Optional[datetime]:
    """Assign a reasonable due date from planner horizon."""
    now = datetime.now(timezone.utc)
    normalized = normalize_horizon(horizon)
    if normalized == "today":
        return datetime(now.year, now.month, now.day, 18, 0, tzinfo=timezone.utc)
    if normalized == "this_week":
        return now + timedelta(days=min(max(due_days, 1), 7))
    return now + timedelta(days=max(due_days, 14))


def task_source_key(post_id: Optional[str], title: str) -> str:
    """Generate unique key for task deduplication."""
    normalized = f"{post_id or 'none'}::{title.strip().lower()}"
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def is_allowed_media_host(hostname: Optional[str]) -> bool:
    """Validate Instagram media hosts for proxying."""
    if not hostname:
        return False
    host = hostname.lower()
    allowed_suffixes = ("cdninstagram.com", "fbcdn.net", "instagram.com")
    return any(host == suffix or host.endswith(f".{suffix}") for suffix in allowed_suffixes)


def extract_og_image_url(html: str) -> Optional[str]:
    """Extract og:image URL from Instagram HTML."""
    marker = 'property="og:image"'
    if marker not in html:
        marker = "property='og:image'"
    if marker not in html:
        return None
    content_marker = 'content="'
    start = html.find(content_marker, html.find(marker))
    if start == -1:
        content_marker = "content='"
        start = html.find(content_marker, html.find(marker))
    if start == -1:
        return None
    start += len(content_marker)
    end = html.find(content_marker[0], start)
    if end == -1:
        return None
    return html[start:end]


# Error handlers
@app.errorhandler(429)
def rate_limit_handler(e):
    """Handle rate limit errors."""
    return jsonify({"error": "Rate limit exceeded", "retry_after": e.description}), 429


@app.errorhandler(500)
def server_error_handler(e):
    """Handle server errors."""
    logger.error(f"Server error: {e}")
    return jsonify({"error": "Internal server error"}), 500


# Health check
@app.route("/api/health", methods=["GET"])
def health():
    """Health check endpoint."""
    return jsonify(
        {
            "status": "ok",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "version": "2.0.0",
        }
    )


# Configuration
@app.route("/api/config", methods=["GET"])
def get_config():
    """Get public configuration."""
    analysis_provider = ""
    analysis_model = ""
    if config.gemini.is_configured():
        analysis_provider = "gemini"
        analysis_model = config.gemini.model
    elif config.azure.is_configured():
        analysis_provider = "azure"
        analysis_model = config.azure.model

    return jsonify(
        {
            "analysis_provider": analysis_provider,
            "analysis_model": analysis_model,
            "azure_endpoint": config.azure.endpoint or "",
            "has_azure_key": bool(config.azure.api_key),
            "max_posts": config.app.max_posts,
            "model": config.azure.model,
            "gemini_model": config.gemini.model,
            "has_gemini_key": bool(config.gemini.api_key),
            "verification_provider": config.verification.provider,
            "verification_model": config.verification.model,
            "has_verification_key": bool(config.verification.api_key),
            "has_tavily_key": bool(config.verification.tavily_api_key),
            "verification_max_claims": config.verification.max_claims,
            "verification_max_sources": config.verification.max_sources,
            "verification_providers": verification_service.get_provider_names(),
        }
    )


# Status endpoints
@app.route("/api/status", methods=["GET"])
@limiter.exempt
def get_status():
    """Get analysis status."""
    payload = serialize_status(analysis_status)
    payload["system"] = build_system_overview()
    return jsonify(payload)


@app.route("/api/tasks/status", methods=["GET"])
@limiter.exempt
def get_task_status():
    """Get task job status."""
    payload = serialize_status(task_job_status)
    payload["system"] = build_system_overview()
    return jsonify(payload)


@app.route("/", methods=["GET"])
def index():
    """Serve the main UI."""
    if frontend_index_file.exists():
        return send_file(frontend_index_file)
    return send_file("index-ui.html")


@app.route("/legacy", methods=["GET"])
def legacy_index():
    """Serve the legacy single-file UI explicitly."""
    return send_file("index-ui.html")


@app.route("/<path:path>", methods=["GET"])
def spa_fallback(path: str):
    """Serve the built SPA for client-side routes while leaving APIs untouched."""
    if path.startswith("api/"):
        abort(404)
    if frontend_index_file.exists():
        return send_file(frontend_index_file)
    return send_file("index-ui.html")


# Statistics
@app.route("/api/stats", methods=["GET"])
def get_stats():
    """Get analysis statistics."""
    try:
        overview = build_system_overview()
        analyzed_count = overview["analysis_count"]

        # Average sentiment
        avg_sentiment = (
            db.session.query(db.func.avg(Analysis.sentiment_score)).scalar() or 0.0
        )

        # Categories
        categories = {}
        for cat, count in (
            db.session.query(Analysis.category, db.func.count(Analysis.category))
            .group_by(Analysis.category)
            .all()
        ):
            if cat:
                categories[cat] = categories.get(cat, 0) + count

        # Sentiment distribution
        sentiments = {"Positive": 0, "Neutral": 0, "Negative": 0}
        for label, count in (
            db.session.query(
                Analysis.sentiment_label, db.func.count(Analysis.sentiment_label)
            )
            .group_by(Analysis.sentiment_label)
            .all()
        ):
            key = str(label or "").strip().title()
            if key in sentiments:
                sentiments[key] = count
            else:
                sentiments["Neutral"] += count

        return jsonify(
            {
                **overview,
                "analyzed_count": analyzed_count,
                "avg_sentiment": round(float(avg_sentiment), 2),
                "categories": categories,
                "sentiments": sentiments,
            }
        )
    except Exception as e:
        logger.error(f"Stats error: {e}")
        return jsonify({"error": str(e)}), 500


# Instagram authentication
@app.route("/api/auth/browser-cookies", methods=["POST"])
def get_browser_cookies():
    """Extract cookies from browser."""
    try:
        data = request.get_json() or {}
        browser_name = data.get("browser", "all")

        browsers = []
        if browser_name == "all":
            browsers = [
                ("Chrome", browser_cookie3.chrome),
                ("Firefox", browser_cookie3.firefox),
                ("Safari", browser_cookie3.safari),
                ("Edge", browser_cookie3.edge),
            ]
        elif hasattr(browser_cookie3, browser_name.lower()):
            browsers = [(browser_name, getattr(browser_cookie3, browser_name.lower()))]

        for name, func in browsers:
            try:
                cj = func(domain_name=".instagram.com")
                raw_cookie = "; ".join(
                    f"{cookie.name}={cookie.value}" for cookie in cj if cookie.domain.endswith("instagram.com")
                )
                for cookie in cj:
                    if cookie.name == "sessionid":
                        result = {
                            "sessionid": cookie.value,
                            "raw_cookie": raw_cookie,
                            "browser": name,
                            "user_agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
                        }
                        set_auth(result)
                        return jsonify(result)
            except Exception as e:
                logger.debug(f"Could not load from {name}: {e}")

        return jsonify({"error": "No Instagram session found"}), 404

    except Exception as e:
        logger.error(f"Browser cookie error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/test-connection", methods=["POST"])
def test_connection():
    """Test Instagram connection."""
    try:
        data = InstagramAuthRequest(**request.get_json()).model_dump()
        set_auth(data)

        client = InstagramClient(
            sessionid=data.get("sessionid"),
            raw_cookie=data.get("raw_cookie"),
            user_agent=data.get("user_agent"),
        )

        is_valid, message = client.validate_session()

        if is_valid:
            return jsonify({"success": True, "message": message})
        else:
            return jsonify({"success": False, "error": message}), 400

    except InstagramAuthError as e:
        return jsonify({"success": False, "error": str(e)}), 401
    except Exception as e:
        logger.error(f"Connection test error: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/test-azure", methods=["POST"])
def test_azure():
    """Validate Azure OpenAI deployment configuration."""
    try:
        payload = request.get_json() or {}
        analyzer = build_ai_analyzer(payload)
        if analyzer.client is None:
            return jsonify({"success": False, "error": "Azure OpenAI credentials are not configured"}), 400

        ok, message = analyzer.validate_chat_deployment()
        return jsonify({"success": ok, "message": message if ok else None, "error": None if ok else message}), (200 if ok else 400)
    except Exception as e:
        logger.error(f"Azure test error: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


# Posts endpoints
@app.route("/api/posts", methods=["GET"])
def get_posts():
    """Get posts with filtering and pagination."""
    try:
        filters = PostsFilterRequest(
            page=request.args.get("page", 1, type=int),
            per_page=request.args.get("per_page", 50, type=int),
            sort=request.args.get("sort", "newest"),
            category=request.args.get("category", "all"),
            collection=request.args.get("collection", "all"),
            sentiment=request.args.get("sentiment", "all"),
            verification=request.args.get("verification", "all"),
        )

        query = Post.query
        analysis_joined = False
        # Apply filters
        if filters.category != "all":
            query = query.join(Analysis)
            analysis_joined = True
            query = query.filter(Analysis.category == filters.category)

        if filters.sentiment != "all":
            if not analysis_joined:
                query = query.join(Analysis)
                analysis_joined = True
            query = query.filter(Analysis.sentiment_label == filters.sentiment)

        if filters.collection != "all":
            query = query.filter(Post.collections.contains(filters.collection))

        if filters.verification != "all":
            query = query.outerjoin(PostVerification)
            if filters.verification == "verified":
                query = query.filter(PostVerification.status == "completed")
            elif filters.verification == "failed":
                query = query.filter(PostVerification.status == "failed")
            elif filters.verification == "unverified":
                query = query.filter(
                    or_(
                        PostVerification.id == None,
                        PostVerification.status == "pending",
                    )
                )

        # Apply sorting
        if filters.sort in {"sentiment_desc", "sentiment_asc"} and not analysis_joined:
            query = query.outerjoin(Analysis)
            analysis_joined = True
        sort_options = {
            "newest": Post.timestamp.desc(),
            "oldest": Post.timestamp.asc(),
            "sentiment_desc": Analysis.sentiment_score.desc(),
            "sentiment_asc": Analysis.sentiment_score.asc(),
        }
        query = query.order_by(sort_options.get(filters.sort, Post.timestamp.desc()))

        # Paginate
        pagination = query.paginate(
            page=filters.page,
            per_page=filters.per_page,
            error_out=False,
        )

        return jsonify(
            {
                "posts": [post.to_dict() for post in pagination.items],
                "total": pagination.total,
                "pages": pagination.pages,
                "current_page": pagination.page,
                "has_next": pagination.has_next,
                "has_prev": pagination.has_prev,
            }
        )

    except Exception as e:
        logger.error(f"Posts error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/admin/refresh-categories", methods=["POST"])
def refresh_categories():
    """Reclassify stored analyses using saved text fields only."""
    try:
        analyzer = build_ai_analyzer(request.get_json() or {})
        if not analyzer.is_available():
            raise ValueError("No analysis model credentials are configured")

        analyses = (
            Analysis.query.join(Post)
            .order_by(Post.timestamp.is_(None), Post.timestamp.desc(), Post.created_at.desc())
            .all()
        )
        refreshed = 0
        for analysis in analyses:
            post = analysis.post
            if post is None:
                continue
            classification = analyzer.classify_saved_content(
                post.to_dict(),
                {
                    "ocr_text": analysis.ocr_text,
                    "video_transcript": analysis.video_transcript,
                    "visual_description": analysis.visual_description,
                    "video_summary": analysis.video_summary,
                },
            )
            if not classification:
                continue
            analysis.category = classification["category"]
            analysis.topics = classification["topics"]
            if analysis.raw_analysis:
                try:
                    payload = json.loads(analysis.raw_analysis)
                except (TypeError, ValueError, json.JSONDecodeError):
                    payload = {}
                payload["category"] = analysis.category
                payload["topics"] = analysis.topics
                analysis.raw_analysis = json.dumps(payload, ensure_ascii=False)
            db.session.add(analysis)
            refreshed += 1

        db.session.commit()
        return jsonify(
            {
                "status": "completed",
                "refreshed_count": refreshed,
                "analysis_count": Analysis.query.count(),
            }
        )
    except Exception as e:
        db.session.rollback()
        logger.error(f"Category refresh error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/posts/<post_id>/verification", methods=["GET"])
def get_post_verification(post_id: str):
    """Return the latest verification report for a post."""
    post = db.session.get(Post, post_id)
    if post is None:
        return jsonify({"error": "Post not found"}), 404
    if post.verification is None:
        return jsonify({"error": "Verification not found"}), 404
    return jsonify(post.verification.to_dict())


@app.route("/api/posts/<post_id>/verify", methods=["POST"])
def verify_post(post_id: str):
    """Run grounded verification for a single post."""
    post = db.session.get(Post, post_id)
    if post is None:
        return jsonify({"error": "Post not found"}), 404
    if (
        not post.analysis
        and post.caption
        and post.caption.startswith("Post by ")
        and not post.thumbnail_url
        and not post.media_url
    ):
        return (
            jsonify(
                {
                    "error": (
                        "This post only has recovered index metadata. Sync or import real post data "
                        "before trying to analyze or verify it."
                    )
                }
            ),
            400,
        )

    try:
        data = VerificationRequest(**(request.get_json() or {})).model_dump(
            exclude_none=True
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 400

    verification = post.verification or PostVerification(post_id=post.id, provider="", status="pending")
    db.session.add(verification)

    try:
        analysis = analyze_existing_post(post, data)
        report = verification_service.verify_post(post, analysis, data)
        verification.provider = report["provider"]
        verification.model = report["model"]
        verification.status = report["status"]
        verification.verdict = report["verdict"]
        verification.confidence = report["confidence"]
        verification.claims = report["claims"]
        verification.source_links = report["source_links"]
        verification.evidence_summary = report["evidence_summary"]
        verification.raw_report = report["raw_report"]
        verification.last_error = None
        db.session.add(verification)
        db.session.commit()
        return jsonify(verification.to_dict())
    except (ValueError, VerificationServiceError) as e:
        db.session.rollback()
        verification = PostVerification.query.filter_by(post_id=post.id).first() or PostVerification(post_id=post.id)
        settings = verification_service.build_settings(data)
        verification.provider = settings.provider or config.verification.provider
        verification.model = settings.model or config.verification.model
        verification.status = "failed"
        verification.last_error = str(e)
        verification.raw_report = None
        db.session.add(verification)
        db.session.commit()
        return jsonify({"error": str(e), "verification": verification.to_dict()}), 400
    except Exception as e:
        db.session.rollback()
        logger.error(f"Verification error for {post_id}: {e}")
        verification = PostVerification.query.filter_by(post_id=post.id).first() or PostVerification(post_id=post.id)
        settings = verification_service.build_settings(data)
        verification.provider = settings.provider or config.verification.provider
        verification.model = settings.model or config.verification.model
        verification.status = "failed"
        verification.last_error = str(e)
        verification.raw_report = None
        db.session.add(verification)
        db.session.commit()
        return jsonify({"error": str(e), "verification": verification.to_dict()}), 500


@app.route("/api/admin/import-analysis-cache", methods=["POST"])
@limiter.limit("2 per minute")
def import_analysis_cache():
    """Import cached analyzed results JSON into the database."""
    try:
        db.create_all()
        cache_path = config.app.output_dir / "analyzed_results.json"
        if not cache_path.exists():
            return jsonify({"error": "analyzed_results.json not found"}), 404

        with cache_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)

        analyzed_posts = payload.get("analyzed_posts") or []
        imported_posts = 0
        imported_analyses = 0

        for item in analyzed_posts:
            post_id = item.get("id")
            if not post_id:
                continue

            post = Post.query.get(post_id)
            if post is None:
                post = Post(id=post_id)
                db.session.add(post)
                imported_posts += 1

            post.shortcode = item.get("shortcode") or post.shortcode
            post.username = item.get("username") or post.username
            post.timestamp = parse_iso_datetime(item.get("timestamp")) or post.timestamp
            post.thumbnail_url = item.get("thumbnail_url") or post.thumbnail_url
            post.media_url = item.get("media_url") or post.media_url
            post.caption = item.get("caption") or post.caption
            post.media_type = item.get("media_type", post.media_type or 1)
            post.is_video = bool(item.get("is_video", post.is_video))
            post.likes = item.get("likes", post.likes or 0) or 0
            post.comments = item.get("comments", post.comments or 0) or 0
            post.collections = item.get("collections") or post.collections or []

            analysis_data = item.get("analysis") or {}
            if analysis_data:
                analysis = post.analysis
                if analysis is None:
                    analysis = Analysis(post_id=post.id)
                    db.session.add(analysis)
                    imported_analyses += 1

                sentiment = analysis_data.get("sentiment") or {}
                analysis.category = analysis_data.get("category") or analysis.category
                analysis.sentiment_score = float(sentiment.get("score", analysis.sentiment_score or 0.0) or 0.0)
                analysis.sentiment_label = sentiment.get("label") or analysis.sentiment_label or "Neutral"
                analysis.credibility_score = analysis_data.get("credibility_score")
                analysis.topics = analysis_data.get("topics") or []
                analysis.learning_points = analysis_data.get("learning_points") or []
                analysis.action_items = analysis_data.get("action_items") or []
                analysis.ocr_text = analysis_data.get("ocr_text")
                analysis.video_transcript = analysis_data.get("video_transcript")
                analysis.visual_description = analysis_data.get("visual_description")
                analysis.video_summary = analysis_data.get("video_summary")
                analysis.raw_analysis = prepare_raw_analysis_payload(
                    analysis_data,
                    int(analysis_data.get("tokens_used", 0) or 0),
                )

        db.session.commit()

        analyzed_posts = Post.query.filter(Post.analysis != None).all()
        if analyzed_posts:
            rag_service.index_posts(analyzed_posts)

        return jsonify(
            {
                "status": "ok",
                "imported_posts": imported_posts,
                "imported_analyses": imported_analyses,
                "total_posts": Post.query.count(),
                "total_analyses": Analysis.query.count(),
            }
        )
    except Exception as e:
        db.session.rollback()
        logger.error(f"Import analysis cache error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/admin/import-index-metadata", methods=["POST"])
@limiter.limit("2 per minute")
def import_index_metadata():
    """Restore placeholder posts from indexed metadata when SQL rows are missing."""
    try:
        db.create_all()
        metadata_path = config.app.output_dir / "indexed_posts.json"
        if not metadata_path.exists():
            return jsonify({"error": "indexed_posts.json not found"}), 404

        with metadata_path.open("r", encoding="utf-8") as handle:
            items = json.load(handle)

        imported_posts = 0
        for item in items:
            post_id = item.get("id")
            if not post_id or Post.query.get(post_id):
                continue

            preview = item.get("context_preview") or ""
            username = None
            caption = preview
            if preview.startswith("Post by "):
                remainder = preview[len("Post by "):]
                username = remainder.split(" on ", 1)[0].strip() or None

            post = Post(
                id=post_id,
                username=username,
                caption=caption[:2000] if caption else None,
                collections=[],
            )
            db.session.add(post)
            imported_posts += 1

        db.session.commit()
        return jsonify(
            {
                "status": "ok",
                "imported_posts": imported_posts,
                "total_posts": Post.query.count(),
            }
        )
    except Exception as e:
        db.session.rollback()
        logger.error(f"Import index metadata error: {e}")
        return jsonify({"error": str(e)}), 500


# Background job runners
def run_analysis_job(data: Dict[str, Any]):
    """Run analysis in background."""
    try:
        analysis_status.start("Initializing...")
        analyzer = build_ai_analyzer(data)
        if not analyzer.is_available():
            raise ValueError("No analysis model credentials are configured")

        # Validate Instagram connection
        client = InstagramClient(
            sessionid=data.get("sessionid"),
            raw_cookie=data.get("raw_cookie"),
            user_agent=data.get("user_agent"),
        )

        is_valid, msg = client.validate_session()
        if not is_valid:
            raise InstagramAuthError(msg)

        analysis_status.update_progress(10, "Fetching posts...")

        # Fetch posts
        existing_ids = {row[0] for row in db.session.query(Post.id).all()}
        posts = client.fetch_saved_posts(
            limit=data.get("max_posts", config.app.max_posts),
            existing_ids=existing_ids,
        )

        if not posts:
            analysis_status.complete(message="No new posts to analyze")
            return

        # Fetch collections
        collections = client.fetch_collections()

        # Analyze posts
        total_posts = len(posts)
        analyzed_count = 0
        total_tokens = 0

        for i, post_data in enumerate(posts, 1):
            progress = 10 + int((i / total_posts) * 80)
            analysis_status.update_progress(
                progress, f"Analyzing post {i}/{total_posts}..."
            )

            # Save post
            post = Post.query.get(post_data["id"])
            if not post:
                post = Post(
                    id=post_data["id"],
                    shortcode=post_data.get("shortcode"),
                    username=post_data.get("username", "unknown"),
                    timestamp=datetime.fromisoformat(post_data["timestamp"])
                    if post_data.get("timestamp")
                    else None,
                    thumbnail_url=post_data.get("thumbnail_url"),
                    media_url=post_data.get("media_url"),
                    caption=post_data.get("caption"),
                    media_type=post_data.get("media_type", 1),
                    is_video=post_data.get("is_video", False),
                    likes=post_data.get("likes", 0),
                    comments=post_data.get("comments", 0),
                    collections=[
                        collections.get(str(cid), str(cid))
                        for cid in post_data.get("saved_collection_ids", [])
                    ],
                )
                db.session.add(post)
                db.session.commit()

            # Skip if already analyzed
            if post.analysis:
                continue

            # Analyze
            result, tokens = analyzer.analyze_post(post_data)
            total_tokens += tokens

            if result:
                save_analysis_for_post(post, result, tokens)
                db.session.commit()
                analyzed_count += 1

        # Rebuild RAG index
        analysis_status.update_progress(90, "Indexing for search...")
        analyzed_posts = Post.query.filter(Post.analysis != None).all()
        rag_service.index_posts(analyzed_posts)

        analysis_status.complete(
            results={
                "analyzed_count": analyzed_count,
                "total_tokens": total_tokens,
                "results": build_analysis_results(),
            },
            message=f"Analyzed {analyzed_count} posts",
        )

    except Exception as e:
        logger.error(f"Analysis job error: {e}")
        analysis_status.fail(str(e))


@app.route("/api/analyze", methods=["POST"])
@limiter.limit("5 per minute")
def start_analysis():
    """Start analysis job."""
    status = analysis_status.get_status()
    if status.running:
        return jsonify({"error": "Analysis already running"}), 400

    try:
        data = AnalysisRequest(**request.get_json()).model_dump()
        set_auth(data)

        thread = threading.Thread(target=run_analysis_job, args=(data,))
        thread.daemon = True
        thread.start()

        return jsonify({"status": "started", "message": "Analysis started"})

    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route("/api/fetch-posts", methods=["POST"])
@limiter.limit("10 per minute")
def fetch_posts():
    """Fetch posts from Instagram and sync them to the database."""
    status = analysis_status.get_status()
    if status.running:
        return jsonify({"error": "Task already running"}), 400

    try:
        data = AnalysisRequest(**request.get_json()).model_dump()
        analysis_status.start("Fetching posts...")

        client = InstagramClient(
            sessionid=data.get("sessionid"),
            raw_cookie=data.get("raw_cookie"),
            user_agent=data.get("user_agent"),
        )
        is_valid, msg = client.validate_session()
        if not is_valid:
            raise InstagramAuthError(msg)

        existing_ids = {row[0] for row in db.session.query(Post.id).all()}
        collections = client.fetch_collections()
        posts = client.fetch_saved_posts(limit=data.get("max_posts"), existing_ids=existing_ids)

        new_count = 0
        for post_data in posts:
            post = Post.query.get(post_data["id"])
            if post is None:
                post = Post(
                    id=post_data["id"],
                    shortcode=post_data.get("shortcode"),
                    username=post_data.get("username", "unknown"),
                    timestamp=parse_iso_datetime(post_data.get("timestamp")),
                    thumbnail_url=post_data.get("thumbnail_url"),
                    media_url=post_data.get("media_url"),
                    caption=post_data.get("caption"),
                    media_type=post_data.get("media_type", 1),
                    is_video=post_data.get("is_video", False),
                    likes=post_data.get("likes", 0),
                    comments=post_data.get("comments", 0),
                    collections=[
                        collections.get(str(cid), str(cid))
                        for cid in post_data.get("saved_collection_ids", [])
                    ],
                )
                db.session.add(post)
                new_count += 1
            else:
                post.shortcode = post_data.get("shortcode") or post.shortcode
                post.username = post_data.get("username") or post.username
                post.timestamp = parse_iso_datetime(post_data.get("timestamp")) or post.timestamp
                post.thumbnail_url = post_data.get("thumbnail_url") or post.thumbnail_url
                post.media_url = post_data.get("media_url") or post.media_url
                post.caption = post_data.get("caption") or post.caption
                post.media_type = post_data.get("media_type", post.media_type)
                post.is_video = bool(post_data.get("is_video", post.is_video))
                post.likes = post_data.get("likes", post.likes) or 0
                post.comments = post_data.get("comments", post.comments) or 0
                post.collections = [
                    collections.get(str(cid), str(cid))
                    for cid in post_data.get("saved_collection_ids", [])
                ]

        db.session.commit()
        analysis_status.complete(
            results={"count": len(posts), "new": new_count},
            message=f"Sync complete! Found {len(posts)} posts ({new_count} new).",
        )
        return jsonify({"status": "completed", "count": len(posts), "new": new_count})
    except Exception as e:
        db.session.rollback()
        analysis_status.fail(str(e))
        return jsonify({"error": str(e)}), 500


@app.route("/api/analyze-batch", methods=["POST"])
@limiter.limit("5 per minute")
def analyze_batch():
    """Analyze existing unanalyzed posts."""
    status = analysis_status.get_status()
    if status.running:
        return jsonify({"error": "Task already running"}), 400

    try:
        data = BatchAnalysisRequest(**(request.get_json() or {})).model_dump()
        analyzer = build_ai_analyzer(data)
        if not analyzer.is_available():
            raise ValueError("No analysis model credentials are configured")
        deployment_ok, deployment_message = analyzer.validate_analysis_backend()
        if not deployment_ok:
            raise ValueError(f"Analysis backend error: {deployment_message}")

        analysis_status.start("Analyzing posts...")
        query = Post.query.filter(Post.analysis == None)
        if data.get("batch_size"):
            query = query.limit(data["batch_size"])
        posts = query.all()

        analyzed_count = 0
        total_tokens = 0
        failed_count = 0
        total_posts = len(posts)
        for index, post in enumerate(posts, start=1):
            analysis_status.update_progress(
                int((index / max(total_posts, 1)) * 100),
                f"Analyzing {index}/{total_posts}...",
            )
            result, tokens = analyzer.analyze_post(post.to_dict())
            total_tokens += tokens
            if not result:
                failed_count += 1
                continue

            save_analysis_for_post(post, result, tokens)
            analyzed_count += 1

        db.session.commit()
        rag_service.index_posts(Post.query.filter(Post.analysis != None).all())
        if total_posts > 0 and analyzed_count == 0:
            raise ValueError(
                "Analysis failed for all posts. Check your Gemini/Azure analysis configuration in Settings."
            )
        analysis_status.complete(
            results={
                "analyzed_count": analyzed_count,
                "failed_count": failed_count,
                "total_tokens": total_tokens,
                "results": build_analysis_results(),
            },
            message=f"Analysis complete! Processed {analyzed_count} posts.",
        )
        return jsonify({"status": "completed", "analyzed_count": analyzed_count})
    except Exception as e:
        db.session.rollback()
        analysis_status.fail(str(e))
        return jsonify({"error": str(e)}), 500


@app.route("/api/profile", methods=["GET"])
def get_profile():
    """Return the persisted psychometric profile snapshot."""
    try:
        workspace_profile = get_workspace_profile()
        if not workspace_profile.psychometric_profile:
            workspace_profile = refresh_workspace_profile_snapshot()
        return jsonify(workspace_profile.psychometric_profile)
    except ValueError as e:
        return jsonify({"error": str(e)}), 404
    except Exception as e:
        logger.error(f"Profile error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/profile/refresh", methods=["POST"])
def refresh_profile():
    """Regenerate and persist the psychometric profile snapshot."""
    try:
        data = request.get_json() or {}
        workspace_profile = refresh_workspace_profile_snapshot(data)
        return jsonify(workspace_profile.to_dict())
    except ValueError as e:
        return jsonify({"error": str(e)}), 404
    except Exception as e:
        logger.error(f"Profile refresh error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/profile-settings", methods=["GET"])
def get_profile_settings():
    """Return manual goals, priorities, and the saved profile snapshot."""
    try:
        workspace_profile = get_workspace_profile()
        db.session.commit()
        return jsonify(workspace_profile.to_dict())
    except Exception as e:
        db.session.rollback()
        return jsonify({"error": str(e)}), 500


@app.route("/api/profile-settings", methods=["PUT"])
def update_profile_settings():
    """Persist manual goals and priorities."""
    try:
        data = ProfileSettingsRequest(**(request.get_json() or {})).model_dump()
        workspace_profile = get_workspace_profile()
        workspace_profile.manual_goals = data["manual_goals"]
        workspace_profile.priorities = data["priorities"]
        workspace_profile.constraints = data["constraints"]
        workspace_profile.focus_areas = data["focus_areas"]
        db.session.add(workspace_profile)
        db.session.commit()
        return jsonify(workspace_profile.to_dict())
    except Exception as e:
        db.session.rollback()
        return jsonify({"error": str(e)}), 400


@app.route("/api/results", methods=["GET"])
def get_results():
    """Return full analysis results."""
    results = build_analysis_results()
    if not results["analyzed_posts"]:
        return jsonify({"error": "No results available"}), 404
    return jsonify(results)


@app.route("/api/summary", methods=["GET"])
def get_summary():
    """Return summary only."""
    results = build_analysis_results()
    if not results["analyzed_posts"]:
        return jsonify({"error": "No results available"}), 404
    return jsonify(results["summary"])


@app.route("/api/results/download", methods=["GET"])
def download_results():
    """Download analysis results as JSON."""
    results = build_analysis_results()
    if not results["analyzed_posts"]:
        return jsonify({"error": "No results available"}), 404
    output_path = config.app.output_dir / "analyzed_results.json"
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2, ensure_ascii=False)
    return send_file(
        output_path,
        mimetype="application/json",
        as_attachment=True,
        download_name=f"instagram_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
    )


@app.route("/api/results/csv", methods=["GET"])
def export_results_csv():
    """Export analyzed posts as CSV."""
    results = build_analysis_results()
    if not results["analyzed_posts"]:
        return jsonify({"error": "No results available"}), 404

    rows = []
    for item in results["analyzed_posts"]:
        analysis = item.get("analysis") or {}
        sentiment = analysis.get("sentiment") or {}
        rows.append(
            {
                "id": item.get("id"),
                "username": item.get("username"),
                "caption": item.get("caption"),
                "category": analysis.get("category"),
                "sentiment_label": sentiment.get("label"),
                "sentiment_score": sentiment.get("score"),
                "topics": ", ".join(analysis.get("topics") or []),
                "action_items": " | ".join(analysis.get("action_items") or []),
                "likes": item.get("likes"),
                "comments": item.get("comments"),
                "timestamp": item.get("timestamp"),
            }
        )

    csv_buffer = StringIO()
    pd.DataFrame(rows).to_csv(csv_buffer, index=False)
    return Response(
        csv_buffer.getvalue(),
        mimetype="text/csv",
        headers={
            "Content-Disposition": f"attachment; filename=analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        },
    )


@app.route("/api/export", methods=["GET"])
def export_markdown():
    """Export analysis and tasks as Markdown."""
    results = build_analysis_results()
    if not results["analyzed_posts"]:
        return jsonify({"error": "No analyzed posts found"}), 404

    lines = [
        "# Instagram Saved Content Analysis",
        "",
        f"Generated on: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
        "",
        "## Overview",
        f"- Total analyzed posts: {results['summary']['total_posts']}",
        f"- Average sentiment: {results['summary']['avg_sentiment']}",
        f"- Top topics: {', '.join(results['summary']['top_topics']) or 'N/A'}",
        "",
        "## Tasks",
    ]

    tasks = ActionTask.query.order_by(ActionTask.created_at.desc()).all()
    if tasks:
        for task in tasks:
            lines.append(f"- [{'x' if task.status == 'done' else ' '}] {task.title} ({task.status})")
    else:
        lines.append("- No tasks yet")

    lines.extend(["", "## Posts"])
    for post in results["analyzed_posts"]:
        analysis = post.get("analysis") or {}
        lines.extend(
            [
                f"### @{post.get('username') or 'unknown'}",
                f"- Category: {analysis.get('category') or 'Other'}",
                f"- Sentiment: {(analysis.get('sentiment') or {}).get('label', 'Neutral')}",
                f"- Topics: {', '.join(analysis.get('topics') or []) or 'N/A'}",
                f"- Caption: {post.get('caption') or ''}",
                "",
            ]
        )

    output_path = config.app.output_dir / f"instagram_export_{datetime.now().strftime('%Y%m%d')}.md"
    with open(output_path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))
    return send_file(
        output_path,
        mimetype="text/markdown",
        as_attachment=True,
        download_name=output_path.name,
    )


@app.route("/api/media-proxy", methods=["GET"])
def media_proxy():
    """Proxy Instagram media for more reliable rendering in the UI."""
    raw_url = (request.args.get("url") or "").strip()
    shortcode = (request.args.get("shortcode") or "").strip()
    candidates: List[str] = []

    if raw_url:
        from urllib.parse import urlparse

        parsed = urlparse(raw_url)
        if parsed.scheme != "https" or not is_allowed_media_host(parsed.hostname):
            return jsonify({"error": "Invalid media URL host"}), 400
        candidates.append(raw_url)
    if shortcode:
        candidates.append(f"https://www.instagram.com/p/{shortcode}/")
    if not candidates:
        return jsonify({"error": "url or shortcode is required"}), 400

    headers = {
        "User-Agent": config.instagram.user_agent,
        "Accept": "image/avif,image/webp,image/apng,image/*,*/*;q=0.8",
        "Referer": "https://www.instagram.com/",
    }
    if config.instagram.raw_cookie:
        headers["Cookie"] = config.instagram.raw_cookie
    elif config.instagram.sessionid:
        headers["Cookie"] = f"sessionid={config.instagram.sessionid}"

    seen = set()
    while candidates:
        candidate = candidates.pop(0)
        if candidate in seen:
            continue
        seen.add(candidate)
        try:
            response = requests.get(candidate, headers=headers, timeout=20, allow_redirects=True)
        except Exception:
            continue
        if response.status_code != 200:
            continue
        content_type = (response.headers.get("Content-Type") or "").lower()
        if content_type.startswith("image/"):
            proxy_response = Response(response.content, status=200, content_type=response.headers.get("Content-Type", "image/jpeg"))
            proxy_response.headers["Cache-Control"] = response.headers.get("Cache-Control", "public, max-age=3600")
            return proxy_response
        if "text/html" in content_type:
            og_image_url = extract_og_image_url(response.text)
            if og_image_url:
                candidates.append(og_image_url)
    return jsonify({"error": "Media not available"}), 404


@app.route("/api/chat", methods=["POST"])
def chat():
    """Chat with indexed content using RAG context."""
    try:
        data = ChatRequest(**request.get_json()).model_dump()
        analyzer = build_ai_analyzer()
        if not analyzer.is_available():
            return jsonify({"error": "Azure OpenAI credentials are not configured"}), 500
        deployment_ok, deployment_message = analyzer.validate_chat_deployment()
        if not deployment_ok:
            return jsonify({"error": f"Azure deployment error: {deployment_message}"}), 400

        conversation_id = data.get("conversation_id") or str(uuid.uuid4())
        conversation = db.session.get(Conversation, conversation_id)
        if conversation is None:
            conversation = Conversation(id=conversation_id)
            db.session.add(conversation)
            db.session.commit()

        db.session.add(Message(conversation_id=conversation_id, role="user", content=data["query"]))
        db.session.commit()

        relevant_posts = rag_service.search(data["query"], k=5)
        if not relevant_posts:
            relevant_posts = Post.query.limit(5).all()

        context = "\n\n".join(rag_service._build_context(post) for post in relevant_posts)
        response = analyzer.client.chat.completions.create(
            model=analyzer.model,
            messages=[
                {
                    "role": "system",
                    "content": "You are Oracle, an AI assistant helping the user explore saved Instagram content. Use the provided context and be concise.\n\nContext:\n" + context,
                },
                {"role": "user", "content": data["query"]},
            ],
            max_completion_tokens=500,
        )
        answer = response.choices[0].message.content or ""
        db.session.add(Message(conversation_id=conversation_id, role="assistant", content=answer))
        db.session.commit()
        return jsonify({"answer": answer, "conversation_id": conversation_id})
    except Exception as e:
        db.session.rollback()
        return jsonify({"error": str(e)}), 500


@app.route("/api/tasks", methods=["GET"])
def list_tasks():
    """List tasks with filters and pagination."""
    try:
        filters = TasksFilterRequest(
            page=request.args.get("page", 1, type=int),
            per_page=request.args.get("per_page", 25, type=int),
            status=request.args.get("status", "all"),
            due=request.args.get("due", "all"),
        )
        now = datetime.now(timezone.utc)
        query = ActionTask.query
        if filters.status != "all":
            query = query.filter(ActionTask.status == normalize_status(filters.status))
        if filters.due == "today":
            start = datetime(now.year, now.month, now.day, tzinfo=timezone.utc)
            end = start + timedelta(days=1)
            query = query.filter(ActionTask.due_date >= start, ActionTask.due_date < end)
        elif filters.due == "overdue":
            query = query.filter(
                ActionTask.due_date != None,
                ActionTask.due_date < now,
                ActionTask.status.in_(["pending", "in_progress"]),
            )
        elif filters.due == "week":
            end = now + timedelta(days=7)
            query = query.filter(
                ActionTask.due_date != None,
                ActionTask.due_date >= now,
                ActionTask.due_date <= end,
            )

        pagination = query.order_by(ActionTask.created_at.desc()).paginate(
            page=filters.page,
            per_page=filters.per_page,
            error_out=False,
        )
        return jsonify(
            {
                "tasks": [task.to_dict() for task in pagination.items],
                "total": pagination.total,
                "pages": pagination.pages,
                "current_page": pagination.page,
                "has_next": pagination.has_next,
                "has_prev": pagination.has_prev,
            }
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/tasks", methods=["POST"])
def create_task():
    """Create a task."""
    try:
        data = TaskCreateRequest(**request.get_json()).model_dump()
        source_key = task_source_key(data.get("post_id"), data["title"])
        existing = ActionTask.query.filter_by(source_key=source_key).first()
        if existing:
            return jsonify({"task": existing.to_dict(), "created": False})

        task = ActionTask(
            post_id=data.get("post_id"),
            title=data["title"],
            notes=data.get("notes"),
            next_step=data.get("next_step"),
            status=normalize_status(data.get("status")),
            priority=normalize_priority(data.get("priority")),
            effort=normalize_effort(data.get("effort")),
            impact=normalize_impact(data.get("impact")),
            horizon=normalize_horizon(data.get("horizon")),
            due_date=parse_iso_datetime(data.get("due_date")),
            scheduled_for=parse_iso_datetime(data.get("scheduled_for")),
            source=data.get("source") or "manual",
            source_key=source_key,
            evidence_text=data.get("evidence_text"),
        )
        if task.status == "done":
            task.completed_at = datetime.now(timezone.utc)
        db.session.add(task)
        db.session.commit()
        return jsonify({"task": task.to_dict(), "created": True}), 201
    except Exception as e:
        db.session.rollback()
        return jsonify({"error": str(e)}), 500


@app.route("/api/tasks/<int:task_id>", methods=["PATCH"])
def update_task(task_id: int):
    """Update a task."""
    task = db.session.get(ActionTask, task_id)
    if task is None:
        return jsonify({"error": "Task not found"}), 404
    try:
        data = TaskUpdateRequest(**request.get_json()).model_dump(exclude_unset=True)
        if "title" in data:
            task.title = data["title"]
        if "notes" in data:
            task.notes = data["notes"]
        if "next_step" in data:
            task.next_step = data["next_step"]
        if "priority" in data:
            task.priority = normalize_priority(data["priority"])
        if "effort" in data:
            task.effort = normalize_effort(data["effort"])
        if "impact" in data:
            task.impact = normalize_impact(data["impact"])
        if "horizon" in data:
            task.horizon = normalize_horizon(data["horizon"])
        if "due_date" in data:
            task.due_date = parse_iso_datetime(data["due_date"])
        if "scheduled_for" in data:
            task.scheduled_for = parse_iso_datetime(data["scheduled_for"])
        if "evidence_text" in data:
            task.evidence_text = data["evidence_text"]
        if "source" in data and data["source"]:
            task.source = data["source"]
        if "status" in data:
            new_status = normalize_status(data["status"])
            task.completed_at = datetime.now(timezone.utc) if new_status == "done" else None
            task.status = new_status
        task.source_key = task_source_key(task.post_id, task.title)
        db.session.commit()
        return jsonify({"task": task.to_dict()})
    except Exception as e:
        db.session.rollback()
        return jsonify({"error": str(e)}), 500


@app.route("/api/tasks/stats", methods=["GET"])
def task_stats():
    """Return task summary metrics."""
    now = datetime.now(timezone.utc)
    today_start = datetime(now.year, now.month, now.day, tzinfo=timezone.utc)
    tomorrow = today_start + timedelta(days=1)
    completed_7d = ActionTask.query.filter(
        ActionTask.completed_at != None,
        ActionTask.completed_at >= now - timedelta(days=7),
    ).count()
    created_7d = ActionTask.query.filter(
        ActionTask.created_at >= now - timedelta(days=7),
    ).count()
    completion_rate_7d = round((completed_7d / created_7d) * 100, 1) if created_7d else 0.0
    return jsonify(
        {
            "total": ActionTask.query.count(),
            "open": ActionTask.query.filter(ActionTask.status.in_(["pending", "in_progress"])).count(),
            "done": ActionTask.query.filter(ActionTask.status == "done").count(),
            "overdue": ActionTask.query.filter(
                ActionTask.due_date != None,
                ActionTask.due_date < now,
                ActionTask.status.in_(["pending", "in_progress"]),
            ).count(),
            "due_today": ActionTask.query.filter(
                ActionTask.due_date != None,
                ActionTask.due_date >= today_start,
                ActionTask.due_date < tomorrow,
                ActionTask.status.in_(["pending", "in_progress"]),
            ).count(),
            "completed_7d": completed_7d,
            "created_7d": created_7d,
            "completion_rate_7d": completion_rate_7d,
        }
    )


@app.route("/api/tasks/today-plan", methods=["GET"])
def today_plan():
    """Return top tasks for today."""
    max_items = min(request.args.get("max_items", 3, type=int), 10)
    tasks = ActionTask.query.filter(
        ActionTask.status.in_(["pending", "in_progress"])
    ).all()
    tasks = sorted(
        tasks,
        key=lambda task: (
            {"today": 0, "this_week": 1, "later": 2}.get(task.horizon or "later", 3),
            -int(task.priority or 0),
            coerce_utc_datetime(task.due_date)
            or datetime.max.replace(tzinfo=timezone.utc),
            coerce_utc_datetime(task.created_at)
            or datetime.max.replace(tzinfo=timezone.utc),
        ),
    )[:max_items]
    return jsonify(
        {
            "plan": [task.to_dict() for task in tasks],
            "count": len(tasks),
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }
    )


@app.route("/api/tasks/bootstrap", methods=["POST"])
def bootstrap_tasks():
    """Create goal-aware tasks from recent analyzed content."""
    status = task_job_status.get_status()
    if status.running:
        return jsonify({"error": "Task generation already running"}), 400
    try:
        data = TaskBootstrapRequest(**(request.get_json() or {})).model_dump()
        task_job_status.start("Generating tasks...")
        task_job_status.update_progress(15, "Loading goals and recent content...")
        workspace_profile = get_workspace_profile()
        posts = get_recent_analyzed_posts(limit=30)
        suggestions = planner_service.generate_tasks(
            workspace_profile,
            posts,
            limit=data["limit"],
        )

        task_job_status.update_progress(60, "Saving prioritized tasks...")
        created = 0
        for suggestion in suggestions:
            source_key = task_source_key(suggestion.get("post_id"), suggestion["title"])
            if ActionTask.query.filter_by(source_key=source_key).first():
                continue
            db.session.add(
                ActionTask(
                    post_id=suggestion.get("post_id"),
                    title=suggestion["title"],
                    notes=suggestion.get("notes"),
                    next_step=suggestion.get("next_step"),
                    status="pending",
                    priority=normalize_priority(suggestion.get("priority")),
                    effort=normalize_effort(suggestion.get("effort")),
                    impact=normalize_impact(suggestion.get("impact")),
                    horizon=normalize_horizon(suggestion.get("horizon")),
                    due_date=suggested_due_date_for_horizon(
                        suggestion.get("horizon"), data["due_days"]
                    ),
                    source="goal_planner",
                    source_key=source_key,
                    evidence_text=suggestion.get("evidence_text"),
                )
            )
            created += 1
        db.session.commit()
        task_job_status.complete(
            results={"created_count": created, "suggestion_count": len(suggestions)},
            message=f"Task generation complete! Created {created} goal-aware tasks.",
        )
        return jsonify({"status": "completed", "created_count": created})
    except Exception as e:
        db.session.rollback()
        task_job_status.fail(str(e))
        return jsonify({"error": str(e)}), 500


@app.route("/api/admin/normalize-analysis", methods=["POST"])
def normalize_analysis():
    """Normalize legacy JSON list fields."""
    updated = 0
    try:
        for row in Analysis.query.all():
            before = json.dumps(row.to_dict(), sort_keys=True, default=str)
            row.topics = row.topics
            row.learning_points = row.learning_points
            row.action_items = row.action_items
            after = json.dumps(row.to_dict(), sort_keys=True, default=str)
            if before != after:
                updated += 1
        db.session.commit()
        return jsonify({"updated_rows": updated, "total_rows": Analysis.query.count()})
    except Exception as e:
        db.session.rollback()
        return jsonify({"error": str(e)}), 500


@app.route("/api/rebuild-rag", methods=["POST"])
def rebuild_rag():
    """Rebuild the RAG index from analyzed posts."""
    posts = Post.query.join(Analysis).all()
    if not posts:
        return jsonify({"error": "No analyzed posts found"}), 404
    rag_service._reset_index()
    indexed = rag_service.index_posts(posts)
    return jsonify({"success": True, "indexed_count": indexed})
