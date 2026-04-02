"""Test suite for Instagram Analyzer."""

import json
import pytest
from datetime import datetime, timezone
from database import (
    ActionTask,
    Analysis,
    Conversation,
    Message,
    Post,
    PostVerification,
    WorkspaceProfile,
    db,
)


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
                next_step="Do the smallest useful thing first.",
                effort="medium",
                impact="high",
                horizon="this_week",
            )
            db.session.add(task)
            db.session.commit()

            assert task.title == "Test task"
            assert task.status == "pending"
            assert task.next_step == "Do the smallest useful thing first."
            assert task.effort == "medium"
            assert task.impact == "high"
            assert task.horizon == "this_week"

    def test_task_to_dict_includes_structured_fields(self, app):
        """Task serialization should include planner-specific fields."""
        with app.app_context():
            task = ActionTask(
                title="Plan sprint",
                notes="Weekly planning",
                next_step="List the top three outcomes for the sprint.",
                effort="quick",
                impact="medium",
                horizon="today",
            )
            db.session.add(task)
            db.session.commit()

            data = task.to_dict()
            assert data["next_step"] == "List the top three outcomes for the sprint."
            assert data["effort"] == "quick"
            assert data["impact"] == "medium"
            assert data["horizon"] == "today"


class TestWorkspaceProfileModel:
    """Test workspace profile model."""

    def test_get_singleton(self, app):
        """Singleton helper should create exactly one row."""
        with app.app_context():
            profile = WorkspaceProfile.get_singleton()
            db.session.commit()

            again = WorkspaceProfile.get_singleton()
            db.session.commit()

            assert profile.id == 1
            assert again.id == 1
            assert WorkspaceProfile.query.count() == 1

    def test_profile_to_dict(self, app):
        """Profile serialization should preserve saved settings."""
        with app.app_context():
            profile = WorkspaceProfile(
                id=1,
                manual_goals=["Ship product"],
                priorities=["Health"],
                psychometric_profile={"archetype": "Builder"},
            )
            db.session.add(profile)
            db.session.commit()

            data = profile.to_dict()
            assert data["manual_goals"] == ["Ship product"]
            assert data["psychometric_profile"]["archetype"] == "Builder"


class TestPostVerificationModel:
    """Test verification model."""

    def test_verification_to_dict(self, app):
        """Verification serialization should include grounded report fields."""
        with app.app_context():
            post = Post(id="verify-1", username="tester")
            db.session.add(post)
            db.session.flush()

            verification = PostVerification(
                post_id=post.id,
                provider="openai",
                model="gpt-4.1",
                status="completed",
                verdict="supported",
                confidence=0.8,
                claims=[{"claim": "Test", "verdict": "supported"}],
                source_links=[{"title": "Source", "url": "https://example.com"}],
                raw_report={"verdict": "supported"},
            )
            db.session.add(verification)
            db.session.commit()

            data = verification.to_dict(include_raw_report=False)
            assert data["provider"] == "openai"
            assert data["verdict"] == "supported"
            assert data["raw_report"] is None


class TestVerificationService:
    """Test grounded verification providers."""

    def test_provider_list_includes_tavily_gemini(self):
        """The pluggable service should expose the Tavily + Gemini adapter."""
        from services.verification_service import VerificationService

        service = VerificationService()
        assert "tavily_gemini" in service.get_provider_names()

    def test_tavily_settings_require_search_and_llm_keys(self):
        """Tavily + Gemini verification needs both the search key and the LLM key."""
        from services.verification_service import VerificationSettings

        settings = VerificationSettings.from_overrides(
            {
                "provider": "tavily_gemini",
                "verification_model": "gemini-3.1-flash-lite-preview",
                "verification_api_key": "gemini-key",
                "tavily_api_key": "tavily-key",
                "max_claims": 3,
                "max_sources": 3,
            }
        )

        assert settings.is_configured()
        assert settings.provider == "tavily_gemini"
        assert settings.tavily_api_key == "tavily-key"

    def test_tavily_gemini_provider_builds_report(self, monkeypatch):
        """The Tavily + Gemini adapter should combine search results into a normalized report."""
        from services import verification_service as verification_module

        class FakeResponse:
            def __init__(self, text):
                self.text = text

        class FakeModels:
            def __init__(self):
                self.calls = []

            def generate_content(self, **kwargs):
                self.calls.append(kwargs)
                if len(self.calls) == 1:
                    return FakeResponse(
                        json.dumps(
                            {
                                "claims": [
                                    {
                                        "claim": "Walking 10 minutes after meals helps blood sugar",
                                        "query": "walking after meals blood sugar evidence",
                                        "why_check": "Health advice claim",
                                    }
                                ]
                            }
                        )
                    )
                return FakeResponse(
                    json.dumps(
                        {
                            "verdict": "supported",
                            "confidence": 0.82,
                            "evidence_summary": "Multiple credible sources supported the claim.",
                            "claims": [
                                {
                                    "claim": "Walking 10 minutes after meals helps blood sugar",
                                    "verdict": "supported",
                                    "confidence": 0.82,
                                    "rationale": "Evidence from reputable health sources aligned.",
                                    "sources": [
                                        {
                                            "title": "Example Study",
                                            "url": "https://example.com/study",
                                            "publisher": "example.com",
                                        }
                                    ],
                                }
                            ],
                            "source_links": [
                                {
                                    "title": "Example Study",
                                    "url": "https://example.com/study",
                                    "publisher": "example.com",
                                }
                            ],
                        }
                    )
                )

        fake_models = FakeModels()

        class FakeGeminiClient:
            def __init__(self, api_key):
                self.api_key = api_key
                self.models = fake_models

        class FakeSearchResponse:
            status_code = 200

            def json(self):
                return {
                    "results": [
                        {
                            "title": "Example Study",
                            "url": "https://example.com/study",
                            "content": "Short summary",
                            "raw_content": "Longer study excerpt",
                            "score": 0.91,
                        }
                    ]
                }

        monkeypatch.setattr(
            verification_module,
            "genai",
            type("FakeGenaiModule", (), {"Client": FakeGeminiClient}),
        )
        monkeypatch.setattr(
            verification_module.requests,
            "post",
            lambda *args, **kwargs: FakeSearchResponse(),
        )

        provider = verification_module.TavilyGeminiVerificationProvider()
        settings = verification_module.VerificationSettings.from_overrides(
            {
                "provider": "tavily_gemini",
                "verification_model": "gemini-3.1-flash-lite-preview",
                "verification_api_key": "gemini-key",
                "tavily_api_key": "tavily-key",
                "max_claims": 3,
                "max_sources": 3,
            }
        )

        report = provider.verify("Caption: health advice", settings)

        assert report["verdict"] == "supported"
        assert report["claims"][0]["claim"].startswith("Walking 10 minutes")
        assert report["source_links"][0]["url"] == "https://example.com/study"
        assert len(fake_models.calls) == 2
        assert fake_models.calls[0]["config"]["response_schema"]["type"] == "object"
        assert fake_models.calls[0]["config"]["response_mime_type"] == "application/json"

    def test_tavily_gemini_invalid_api_key_is_user_actionable(self, monkeypatch):
        """Gemini auth failures should return a clear message instead of bubbling as 500s."""
        from services import verification_service as verification_module

        class FakeModels:
            def generate_content(self, **kwargs):
                raise RuntimeError("400 INVALID_ARGUMENT. API_KEY_INVALID. API key not valid.")

        class FakeGeminiClient:
            def __init__(self, api_key):
                self.models = FakeModels()

        monkeypatch.setattr(
            verification_module,
            "genai",
            type("FakeGenaiModule", (), {"Client": FakeGeminiClient}),
        )

        provider = verification_module.TavilyGeminiVerificationProvider()
        settings = verification_module.VerificationSettings.from_overrides(
            {
                "provider": "tavily_gemini",
                "verification_model": "gemini-3.1-flash-lite-preview",
                "verification_api_key": "bad-key",
                "tavily_api_key": "tavily-key",
            }
        )

        with pytest.raises(verification_module.VerificationServiceError) as exc:
            provider.verify("Caption: health advice", settings)

        assert "Clear the 'LLM API Key' field" in str(exc.value)


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

        assert normalized["category"] == "AI & Tech"
        assert normalized["sentiment"]["label"] == "Positive"

    def test_analyze_post_prefers_gemini_when_available(self, monkeypatch):
        """Video analysis should use Gemini before the Whisper/Azure fallback path."""
        from services.ai_analyzer import AIAnalyzer

        analyzer = AIAnalyzer()
        analyzer.gemini_client = object()
        analyzer.client = object()

        expected = ({"category": "Education"}, 123)
        monkeypatch.setattr(analyzer, "_analyze_post_with_gemini", lambda post: expected)
        monkeypatch.setattr(
            analyzer,
            "extract_audio",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(
                AssertionError("Whisper fallback should not run when Gemini is configured")
            ),
        )

        result = analyzer.analyze_post(
            {
                "id": "video-1",
                "is_video": True,
                "video_url": "https://example.com/video.mp4",
            }
        )

        assert result == expected

    def test_validate_analysis_backend_prefers_gemini(self):
        """Backend validation should check Gemini first when it is configured."""
        from services.ai_analyzer import AIAnalyzer

        class FakeModels:
            def __init__(self):
                self.calls = []

            def generate_content(self, **kwargs):
                self.calls.append(kwargs)
                return object()

        fake_models = FakeModels()
        fake_client = type("FakeGeminiClient", (), {"models": fake_models})()

        analyzer = AIAnalyzer(gemini_model="gemini-3.1-flash-lite-preview")
        analyzer.gemini_client = fake_client
        analyzer.client = None

        ok, message = analyzer.validate_analysis_backend()

        assert ok is True
        assert "Gemini analysis model is valid" in message
        assert fake_models.calls == [
            {
                "model": "gemini-3.1-flash-lite-preview",
                "contents": "ping",
            }
        ]


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


class TestPlannerService:
    """Test goal-aware planning."""

    def _make_post(self, post_id: str, action_item: str, topic: str = "ai") -> Post:
        post = Post(
            id=post_id,
            username="planner",
            caption="Useful caption",
            timestamp=datetime.now(timezone.utc),
        )
        post.analysis = Analysis(
            post_id=post_id,
            category="Technology",
            sentiment_score=0.4,
            sentiment_label="Positive",
            topics=[topic],
            action_items=[action_item],
            learning_points=["Practice consistently"],
            raw_analysis='{"tokens_used": 10}',
        )
        return post

    def test_manual_goals_influence_tasks(self):
        """Manual goals should raise matching action items."""
        from services.planner_service import PlannerService

        planner = PlannerService()
        profile = WorkspaceProfile(
            manual_goals=["Build AI systems"],
            priorities=["Career"],
        )
        tasks = planner.generate_tasks(
            profile,
            [self._make_post("plan-1", "Build an AI prototype", "ai systems")],
            limit=5,
        )

        assert tasks
        assert tasks[0]["title"] == "Build an AI prototype"
        assert "Build AI systems" in tasks[0]["evidence_text"]

    def test_inferred_profile_signals_work_without_manual_goals(self):
        """Psychometric snapshot should still produce tasks when goals are empty."""
        from services.planner_service import PlannerService

        planner = PlannerService()
        profile = WorkspaceProfile(
            psychometric_profile={
                "growth_areas": ["Consistency"],
                "content_dna": {"primary_focus": "Learning"},
            }
        )
        tasks = planner.generate_tasks(
            profile,
            [self._make_post("plan-2", "Build a weekly learning routine", "learning")],
            limit=5,
        )

        assert tasks
        assert tasks[0]["title"] == "Build a weekly learning routine"

    def test_no_analyzed_content_returns_empty(self):
        """Planner should gracefully return no tasks without analyzed posts."""
        from services.planner_service import PlannerService

        planner = PlannerService()
        assert planner.generate_tasks(WorkspaceProfile(), [], limit=5) == []

    def test_duplicate_candidates_are_deduped(self):
        """Matching tasks from multiple posts should collapse to one suggestion."""
        from services.planner_service import PlannerService

        planner = PlannerService()
        profile = WorkspaceProfile(manual_goals=["Build AI systems"])
        posts = [
            self._make_post("plan-3", "Build an AI prototype"),
            self._make_post("plan-4", "Build an AI prototype"),
        ]
        tasks = planner.generate_tasks(profile, posts, limit=5)
        assert len(tasks) == 1

    def test_naive_sqlite_timestamps_do_not_break_recency_scoring(self):
        """Planner should tolerate naive timestamps returned by SQLite."""
        from services.planner_service import PlannerService

        planner = PlannerService()
        profile = WorkspaceProfile(manual_goals=["Build AI systems"])
        post = self._make_post("plan-5", "Build an AI prototype")
        post.timestamp = datetime.now().replace(microsecond=0)

        tasks = planner.generate_tasks(profile, [post], limit=5)

        assert tasks
        assert tasks[0]["title"] == "Build an AI prototype"


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

    def test_gemini_config_check(self):
        """Test Gemini configuration check."""
        from config import GeminiConfig

        cfg = GeminiConfig(api_key=None)
        assert not cfg.is_configured()

        cfg2 = GeminiConfig(api_key="test-key")
        assert cfg2.is_configured()

    def test_verification_config_check_for_tavily_gemini(self):
        """Tavily + Gemini verification config should require both keys."""
        from config import VerificationConfig

        cfg = VerificationConfig(
            provider="tavily_gemini",
            model="gemini-3.1-flash-lite-preview",
            api_key="gemini-key",
            tavily_api_key=None,
        )
        assert not cfg.is_configured()

        cfg2 = VerificationConfig(
            provider="tavily_gemini",
            model="gemini-3.1-flash-lite-preview",
            api_key="gemini-key",
            tavily_api_key="tavily-key",
        )
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
