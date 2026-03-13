"""Input validation schemas using Pydantic."""

from typing import List, Optional

from pydantic import BaseModel, Field, validator


class InstagramAuthRequest(BaseModel):
    """Instagram authentication request."""

    sessionid: Optional[str] = Field(None, description="Instagram session ID")
    raw_cookie: Optional[str] = Field(None, description="Full cookie string")
    user_agent: Optional[str] = Field(None, description="Browser user agent")
    browser: str = Field("none", description="Browser to extract cookies from")

    @validator("sessionid", "raw_cookie")
    def check_credentials(cls, v, values):
        """Ensure at least one credential is provided."""
        if not v and not values.get("sessionid") and not values.get("raw_cookie"):
            raise ValueError("Either sessionid or raw_cookie must be provided")
        return v


class AnalysisRequest(BaseModel):
    """Analysis request parameters."""

    sessionid: str = Field(..., description="Instagram session ID")
    raw_cookie: Optional[str] = None
    user_agent: Optional[str] = None
    azure_endpoint: Optional[str] = None
    azure_key: Optional[str] = None
    model: str = Field("DeepSeek-V3.2", description="Azure model deployment name")
    max_posts: int = Field(20, ge=1, le=1000, description="Maximum posts to analyze")


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
    status: str = Field("pending", regex="^(pending|in_progress|done|archived)$")
    priority: int = Field(2, ge=1, le=3)
    due_date: Optional[str] = None
    scheduled_for: Optional[str] = None
    source: str = Field("manual", max_length=50)
    evidence_text: Optional[str] = None


class TaskUpdateRequest(BaseModel):
    """Task update request."""

    title: Optional[str] = Field(None, min_length=1, max_length=500)
    notes: Optional[str] = None
    status: Optional[str] = Field(None, regex="^(pending|in_progress|done|archived)$")
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
    sort: str = Field("newest", regex="^(newest|oldest|sentiment_desc|sentiment_asc)$")
    category: str = "all"
    collection: str = "all"
    sentiment: str = "all"


class TasksFilterRequest(BaseModel):
    """Tasks filter request."""

    page: int = Field(1, ge=1)
    per_page: int = Field(25, ge=1, le=100)
    status: str = "all"
    due: str = Field("all", regex="^(all|today|overdue|week)$")
