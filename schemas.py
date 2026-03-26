"""Input validation schemas using Pydantic."""

from typing import List, Optional

from pydantic import BaseModel, Field, field_validator


class InstagramAuthRequest(BaseModel):
    """Instagram authentication request."""

    sessionid: Optional[str] = Field(None, description="Instagram session ID")
    raw_cookie: Optional[str] = Field(None, description="Full cookie string")
    user_agent: Optional[str] = Field(None, description="Browser user agent")
    browser: str = Field("none", description="Browser to extract cookies from")

    @field_validator("sessionid", "raw_cookie", mode="after")
    @classmethod
    def normalize_credentials(cls, value):
        """Normalize optional credential values."""
        if value is None:
            return value
        text = str(value).strip()
        return text or None


class AnalysisRequest(BaseModel):
    """Analysis request parameters."""

    sessionid: str = Field(..., description="Instagram session ID")
    raw_cookie: Optional[str] = None
    user_agent: Optional[str] = None
    azure_endpoint: Optional[str] = None
    azure_key: Optional[str] = None
    model: str = Field("DeepSeek-V3.2", description="Azure model deployment name")
    max_posts: int = Field(200, ge=1, le=1000, description="Maximum posts to analyze")


class BatchAnalysisRequest(BaseModel):
    """Batch analysis request."""

    batch_size: Optional[int] = Field(None, ge=1, le=1000)
    azure_endpoint: Optional[str] = None
    azure_key: Optional[str] = None
    model: str = "DeepSeek-V3.2"


class TaskBootstrapRequest(BaseModel):
    """Task generation request."""

    limit: int = Field(50, ge=1, le=1000)
    due_days: int = Field(7, ge=1, le=90)


class TaskCreateRequest(BaseModel):
    """Task creation request."""

    title: str = Field(..., min_length=1, max_length=500)
    notes: Optional[str] = None
    post_id: Optional[str] = None
    status: str = Field("pending", pattern="^(pending|in_progress|done|archived)$")
    priority: int = Field(2, ge=1, le=3)
    due_date: Optional[str] = None
    scheduled_for: Optional[str] = None
    source: str = Field("manual", max_length=50)
    evidence_text: Optional[str] = None


class TaskUpdateRequest(BaseModel):
    """Task update request."""

    title: Optional[str] = Field(None, min_length=1, max_length=500)
    notes: Optional[str] = None
    status: Optional[str] = Field(None, pattern="^(pending|in_progress|done|archived)$")
    priority: Optional[int] = Field(None, ge=1, le=3)
    due_date: Optional[str] = None
    scheduled_for: Optional[str] = None
    evidence_text: Optional[str] = None
    source: Optional[str] = Field(None, max_length=50)


class ChatRequest(BaseModel):
    """Chat request."""

    query: str = Field(..., min_length=1, max_length=1000)
    conversation_id: Optional[str] = None


class PostsFilterRequest(BaseModel):
    """Posts filter request."""

    page: int = Field(1, ge=1)
    per_page: int = Field(50, ge=1, le=100)
    sort: str = Field(
        "newest", pattern="^(newest|oldest|sentiment_desc|sentiment_asc)$"
    )
    category: str = "all"
    collection: str = "all"
    sentiment: str = "all"


class TasksFilterRequest(BaseModel):
    """Tasks filter request."""

    page: int = Field(1, ge=1)
    per_page: int = Field(25, ge=1, le=100)
    status: str = "all"
    due: str = Field("all", pattern="^(all|today|overdue|week)$")
