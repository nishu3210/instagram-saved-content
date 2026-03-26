"""Test suite for Instagram Analyzer."""

import pytest
from datetime import datetime, timezone
from database import db, Post, Analysis, ActionTask, Conversation, Message


class TestPostModel:
    """Test Post model."""

    def test_post_creation(self, app):
        """Test creating a post."""
        with app.app_context():
            post = Post(
                id="test123",
                shortcode="ABC123",
                username="testuser",
                timestamp=datetime.now(timezone.utc),
                caption="Test caption",
                media_type=1,
                is_video=False,
                likes=100,
                comments=10,
            )
            db.session.add(post)
            db.session.commit()

            assert post.id == "test123"
            assert post.username == "testuser"
            assert post.likes == 100

    def test_post_to_dict(self, app):
        """Test post serialization."""
        with app.app_context():
            post = Post(
                id="test456",
                username="user2",
                caption="Test",
            )
            db.session.add(post)
            db.session.commit()

            data = post.to_dict()
            assert data["id"] == "test456"
            assert data["username"] == "user2"
            assert "analysis" in data


class TestAnalysisModel:
    """Test Analysis model."""

    def test_analysis_creation(self, app):
        """Test creating analysis."""
        with app.app_context():
            post = Post(id="test789", username="user3")
            db.session.add(post)
            db.session.commit()

            analysis = Analysis(
                post_id="test789",
                category="Technology",
                sentiment_score=0.8,
                sentiment_label="Positive",
                topics=["python", "coding"],
            )
            db.session.add(analysis)
            db.session.commit()

            assert analysis.category == "Technology"
            assert analysis.sentiment_score == 0.8


class TestActionTaskModel:
    """Test ActionTask model."""

    def test_task_creation(self, app):
        """Test creating a task."""
        with app.app_context():
            task = ActionTask(
                title="Test task",
                status="pending",
                priority=2,
            )
            db.session.add(task)
            db.session.commit()

            assert task.title == "Test task"
            assert task.status == "pending"


class TestInstagramClient:
    """Test Instagram client."""

    def test_client_initialization(self):
        """Test client initialization."""
        from services.instagram_client import InstagramClient

        client = InstagramClient(sessionid="test_session")
        assert client.sessionid == "test_session"
        assert "Cookie" in client._headers

    def test_headers_building(self):
        """Test header construction."""
        from services.instagram_client import InstagramClient

        client = InstagramClient(sessionid="test", user_agent="CustomAgent/1.0")
        assert client._headers["User-Agent"] == "CustomAgent/1.0"


class TestAIAnalyzer:
    """Test AI analyzer."""

    def test_analyzer_initialization(self):
        """Test analyzer initialization."""
        from services.ai_analyzer import AIAnalyzer

        analyzer = AIAnalyzer()
        # Should initialize even without credentials
        assert analyzer is not None

    def test_normalize_result(self):
        """Test result normalization."""
        from services.ai_analyzer import AIAnalyzer

        analyzer = AIAnalyzer()
        raw = {
            "category": "Technology",
            "sentiment": {"score": 0.8, "label": "positive"},
            "topics": ["python"],
        }
        normalized = analyzer._normalize_result(raw)

        assert normalized["category"] == "Technology"
        assert normalized["sentiment"]["label"] == "Positive"


class TestRAGService:
    """Test RAG service."""

    def test_service_initialization(self):
        """Test RAG service initialization."""
        from services.rag_service import RAGService

        service = RAGService()
        assert service.index is not None

    def test_build_context(self, app):
        """Test context building."""
        from services.rag_service import RAGService

        with app.app_context():
            service = RAGService()
            post = Post(id="ctx1", username="user", caption="Test")

            context = service._build_context(post)
            assert "user" in context
            assert "Test" in context


class TestConfig:
    """Test configuration."""

    def test_config_loading(self):
        """Test configuration loading."""
        from config import config

        assert config.instagram is not None
        assert config.azure is not None
        assert config.database is not None

    def test_azure_config_check(self):
        """Test Azure configuration check."""
        from config import AzureConfig

        # Unconfigured
        cfg = AzureConfig(api_key=None, endpoint=None)
        assert not cfg.is_configured()

        # Configured
        cfg2 = AzureConfig(api_key="test", endpoint="https://test.com")
        assert cfg2.is_configured()


class TestStatusManager:
    """Test status manager."""

    def test_thread_safety(self):
        """Test thread-safe operations."""
        from status_manager import ThreadSafeStatusManager

        manager = ThreadSafeStatusManager()

        # Start
        manager.start("Testing")
        status = manager.get_status()
        assert status.running
        assert status.message == "Testing"

        # Update
        manager.update_progress(50, "Halfway")
        status = manager.get_status()
        assert status.progress == 50

        # Complete
        manager.complete(results={"test": "data"})
        status = manager.get_status()
        assert not status.running
        assert status.results["test"] == "data"


class TestSchemas:
    """Test validation schemas."""

    def test_instagram_auth_request(self):
        """Test Instagram auth validation."""
        from schemas import InstagramAuthRequest

        # Valid
        req = InstagramAuthRequest(sessionid="test")
        assert req.sessionid == "test"

    def test_analysis_request(self):
        """Test analysis request validation."""
        from schemas import AnalysisRequest

        req = AnalysisRequest(sessionid="test", max_posts=10)
        assert req.sessionid == "test"
        assert req.max_posts == 10

        # Test max_posts limits
        with pytest.raises(ValueError):
            AnalysisRequest(sessionid="test", max_posts=2000)
