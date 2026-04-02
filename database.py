"""Database models with proper relationships and JSON handling."""

import json
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import Index, event
from sqlalchemy.orm import backref, relationship
from sqlalchemy.types import TypeDecorator, Text

db = SQLAlchemy()


class JSONList(TypeDecorator):
    """Custom type for storing lists as JSON."""

    impl = Text
    cache_ok = True

    def process_bind_param(self, value: Optional[Any], dialect) -> Optional[str]:
        """Convert Python list to JSON string."""
        if value is None:
            return None
        if isinstance(value, (list, tuple)):
            return json.dumps(
                [str(item).strip() for item in value if str(item).strip()],
                ensure_ascii=False,
            )
        if isinstance(value, str):
            # Try to parse as JSON first
            try:
                parsed = json.loads(value)
                if isinstance(parsed, list):
                    return json.dumps(
                        [str(item).strip() for item in parsed if str(item).strip()],
                        ensure_ascii=False,
                    )
            except json.JSONDecodeError:
                # Treat as single item list
                return json.dumps([value.strip()], ensure_ascii=False)
        return json.dumps([str(value)], ensure_ascii=False)

    def process_result_value(self, value: Optional[str], dialect) -> List[str]:
        """Convert JSON string to Python list."""
        if value is None:
            return []
        try:
            parsed = json.loads(value)
            if isinstance(parsed, list):
                return [str(item).strip() for item in parsed if str(item).strip()]
            return [str(parsed).strip()] if str(parsed).strip() else []
        except json.JSONDecodeError:
            return [value.strip()] if value.strip() else []


class JSONBlob(TypeDecorator):
    """Custom type for storing JSON-compatible data."""

    impl = Text
    cache_ok = True

    def process_bind_param(self, value: Optional[Any], dialect) -> Optional[str]:
        """Convert Python data to JSON string."""
        if value is None:
            return None
        return json.dumps(value, ensure_ascii=False)

    def process_result_value(self, value: Optional[str], dialect) -> Optional[Any]:
        """Convert JSON string back to Python data."""
        if value is None:
            return None
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return None


class Post(db.Model):
    """Instagram post model."""

    __tablename__ = "posts"

    id = db.Column(db.String(50), primary_key=True)
    shortcode = db.Column(db.String(50), nullable=True, index=True)
    username = db.Column(db.String(50), nullable=True, index=True)
    timestamp = db.Column(db.DateTime(timezone=True), nullable=True, index=True)
    thumbnail_url = db.Column(db.String(500), nullable=True)
    media_url = db.Column(db.String(500), nullable=True)
    caption = db.Column(db.Text, nullable=True)
    media_type = db.Column(db.Integer, default=1)
    is_video = db.Column(db.Boolean, default=False)
    likes = db.Column(db.Integer, default=0)
    comments = db.Column(db.Integer, default=0)
    collections = db.Column(JSONList, default=list)
    created_at = db.Column(
        db.DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )
    updated_at = db.Column(
        db.DateTime(timezone=True), onupdate=lambda: datetime.now(timezone.utc)
    )

    # Relationships
    analysis = relationship(
        "Analysis",
        backref=backref("post", uselist=False),
        uselist=False,
        cascade="all, delete-orphan",
    )
    tasks = relationship(
        "ActionTask",
        backref="post",
        lazy="dynamic",
        cascade="all, delete-orphan",
    )
    verification = relationship(
        "PostVerification",
        backref=backref("post", uselist=False),
        uselist=False,
        cascade="all, delete-orphan",
    )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "shortcode": self.shortcode,
            "username": self.username,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "thumbnail_url": self.thumbnail_url,
            "media_url": self.media_url,
            "caption": self.caption,
            "media_type": self.media_type,
            "is_video": self.is_video,
            "video_url": self.media_url if self.is_video else None,
            "likes": self.likes,
            "comments": self.comments,
            "collections": self.collections or [],
            "analysis": self.analysis.to_dict() if self.analysis else None,
            "verification": self.verification.to_dict(include_raw_report=False)
            if self.verification
            else None,
        }


class Analysis(db.Model):
    """Post analysis model."""

    __tablename__ = "analysis"

    id = db.Column(db.Integer, primary_key=True)
    post_id = db.Column(
        db.String(50), db.ForeignKey("posts.id"), nullable=False, unique=True
    )

    category = db.Column(db.String(50), nullable=True)
    sentiment_score = db.Column(db.Float, default=0.0)
    sentiment_label = db.Column(db.String(20), default="Neutral")
    credibility_score = db.Column(db.Integer, nullable=True)

    # JSON fields using custom type
    topics = db.Column(JSONList, default=list)
    learning_points = db.Column(JSONList, default=list)
    action_items = db.Column(JSONList, default=list)

    # Raw analysis storage
    raw_analysis = db.Column(db.Text, nullable=True)

    # Enhanced extraction fields
    ocr_text = db.Column(db.Text, nullable=True)
    video_transcript = db.Column(db.Text, nullable=True)
    visual_description = db.Column(db.Text, nullable=True)
    video_summary = db.Column(db.Text, nullable=True)

    created_at = db.Column(
        db.DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )
    updated_at = db.Column(
        db.DateTime(timezone=True), onupdate=lambda: datetime.now(timezone.utc)
    )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "category": self.category,
            "sentiment": {
                "score": self.sentiment_score,
                "label": self.sentiment_label,
            },
            "credibility_score": self.credibility_score,
            "topics": self.topics or [],
            "learning_points": self.learning_points or [],
            "action_items": self.action_items or [],
            "ocr_text": self.ocr_text,
            "video_transcript": self.video_transcript,
            "visual_description": self.visual_description,
            "video_summary": self.video_summary,
        }


class ActionTask(db.Model):
    """Action task model."""

    __tablename__ = "action_tasks"

    id = db.Column(db.Integer, primary_key=True)
    post_id = db.Column(
        db.String(50), db.ForeignKey("posts.id"), nullable=True, index=True
    )
    title = db.Column(db.String(500), nullable=False)
    notes = db.Column(db.Text, nullable=True)
    next_step = db.Column(db.Text, nullable=True)
    status = db.Column(db.String(20), nullable=False, default="pending", index=True)
    priority = db.Column(db.Integer, nullable=False, default=2)
    effort = db.Column(db.String(20), nullable=True)
    impact = db.Column(db.String(20), nullable=True)
    horizon = db.Column(db.String(20), nullable=True, index=True)
    due_date = db.Column(db.DateTime(timezone=True), nullable=True, index=True)
    scheduled_for = db.Column(db.DateTime(timezone=True), nullable=True, index=True)
    completed_at = db.Column(db.DateTime(timezone=True), nullable=True, index=True)
    source = db.Column(db.String(50), nullable=False, default="manual")
    source_key = db.Column(db.String(255), unique=True, nullable=True, index=True)
    evidence_text = db.Column(db.Text, nullable=True)
    created_at = db.Column(
        db.DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        index=True,
    )
    updated_at = db.Column(
        db.DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
    )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "post_id": self.post_id,
            "title": self.title,
            "notes": self.notes,
            "next_step": self.next_step,
            "status": self.status,
            "priority": self.priority,
            "effort": self.effort,
            "impact": self.impact,
            "horizon": self.horizon,
            "due_date": self.due_date.isoformat() if self.due_date else None,
            "scheduled_for": self.scheduled_for.isoformat()
            if self.scheduled_for
            else None,
            "completed_at": self.completed_at.isoformat()
            if self.completed_at
            else None,
            "source": self.source,
            "source_key": self.source_key,
            "evidence_text": self.evidence_text,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }


class WorkspaceProfile(db.Model):
    """Singleton profile with manual goals and persisted psychometric snapshot."""

    __tablename__ = "workspace_profile"

    id = db.Column(db.Integer, primary_key=True, default=1)
    manual_goals = db.Column(JSONList, default=list)
    priorities = db.Column(JSONList, default=list)
    constraints = db.Column(JSONList, default=list)
    focus_areas = db.Column(JSONList, default=list)
    psychometric_profile = db.Column(JSONBlob, nullable=True)
    profile_refreshed_at = db.Column(db.DateTime(timezone=True), nullable=True)
    created_at = db.Column(
        db.DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )
    updated_at = db.Column(
        db.DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
    )

    @classmethod
    def get_singleton(cls) -> "WorkspaceProfile":
        """Load or create the singleton workspace profile."""
        profile = db.session.get(cls, 1)
        if profile is None:
            profile = cls(id=1)
            db.session.add(profile)
            db.session.flush()
        return profile

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "manual_goals": self.manual_goals or [],
            "priorities": self.priorities or [],
            "constraints": self.constraints or [],
            "focus_areas": self.focus_areas or [],
            "psychometric_profile": self.psychometric_profile or None,
            "profile_refreshed_at": self.profile_refreshed_at.isoformat()
            if self.profile_refreshed_at
            else None,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }


class PostVerification(db.Model):
    """Latest verification report for a post."""

    __tablename__ = "post_verification"

    id = db.Column(db.Integer, primary_key=True)
    post_id = db.Column(
        db.String(50), db.ForeignKey("posts.id"), nullable=False, unique=True, index=True
    )
    provider = db.Column(db.String(50), nullable=False)
    model = db.Column(db.String(100), nullable=True)
    status = db.Column(db.String(20), nullable=False, default="pending", index=True)
    verdict = db.Column(db.String(30), nullable=True)
    confidence = db.Column(db.Float, nullable=True)
    claims = db.Column(JSONBlob, nullable=True)
    source_links = db.Column(JSONBlob, nullable=True)
    evidence_summary = db.Column(db.Text, nullable=True)
    raw_report = db.Column(JSONBlob, nullable=True)
    last_error = db.Column(db.Text, nullable=True)
    created_at = db.Column(
        db.DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )
    updated_at = db.Column(
        db.DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
    )

    def to_dict(self, include_raw_report: bool = True) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "provider": self.provider,
            "model": self.model,
            "status": self.status,
            "verdict": self.verdict,
            "confidence": self.confidence,
            "claims": self.claims or [],
            "source_links": self.source_links or [],
            "evidence_summary": self.evidence_summary,
            "raw_report": self.raw_report if include_raw_report else None,
            "last_error": self.last_error,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }


class Conversation(db.Model):
    """Chat conversation model."""

    __tablename__ = "conversations"

    id = db.Column(db.String(36), primary_key=True)
    created_at = db.Column(
        db.DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )

    messages = relationship(
        "Message",
        backref="conversation",
        lazy="dynamic",
        cascade="all, delete-orphan",
    )


class Message(db.Model):
    """Chat message model."""

    __tablename__ = "messages"

    id = db.Column(db.Integer, primary_key=True)
    conversation_id = db.Column(
        db.String(36), db.ForeignKey("conversations.id"), nullable=False
    )
    role = db.Column(db.String(20), nullable=False)
    content = db.Column(db.Text, nullable=False)
    created_at = db.Column(
        db.DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )


# Indexes for performance
Index("idx_posts_timestamp", Post.timestamp)
Index("idx_analysis_category", Analysis.category)
Index("idx_analysis_sentiment", Analysis.sentiment_label)
Index("idx_tasks_status_priority", ActionTask.status, ActionTask.priority)
Index("idx_post_verification_status", PostVerification.status)


def get_db_uri() -> str:
    """Get database URI."""
    from config import config

    return config.database.url
