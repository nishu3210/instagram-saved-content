"""Integration tests for Flask routes."""

from datetime import datetime, timezone

import app as app_module
from app import app
from database import ActionTask, Analysis, Post, PostVerification, WorkspaceProfile, db


def seed_analyzed_post():
    """Create a post with analysis for route tests."""
    post = Post(
        id="post-1",
        shortcode="ABC123",
        username="tester",
        caption="Useful caption",
        timestamp=datetime.now(timezone.utc),
        likes=12,
        comments=3,
        collections=["Ideas"],
    )
    db.session.add(post)
    db.session.flush()
    analysis = Analysis(
        post_id=post.id,
        category="Technology",
        sentiment_score=0.7,
        sentiment_label="Positive",
        topics=["python", "automation"],
        action_items=["Build a prototype"],
        raw_analysis='{"tokens_used": 42}',
    )
    db.session.add(analysis)
    db.session.commit()
    return post


def test_index_route_serves_ui(client):
    """Root route should serve the UI."""
    response = client.get("/")
    assert response.status_code == 200
    assert b"Instagram Intelligence" in response.data


def test_results_and_summary_routes(client, app):
    """Results endpoints should return analyzed content."""
    with app.app_context():
        seed_analyzed_post()

    results_response = client.get("/api/results")
    assert results_response.status_code == 200
    results = results_response.get_json()
    assert len(results["analyzed_posts"]) == 1
    assert results["summary"]["total_posts"] == 1

    summary_response = client.get("/api/summary")
    assert summary_response.status_code == 200
    summary = summary_response.get_json()
    assert summary["avg_sentiment"] == 0.7


def test_posts_filter_by_collection(client, app):
    """Posts endpoint should support collection filtering."""
    with app.app_context():
        seed_analyzed_post()

    response = client.get("/api/posts?collection=Ideas")
    assert response.status_code == 200
    payload = response.get_json()
    assert payload["total"] == 1
    assert payload["posts"][0]["collections"] == ["Ideas"]


def test_posts_filter_by_verification_status(client, app):
    """Posts endpoint should filter by verification status."""
    with app.app_context():
        verified = seed_analyzed_post()
        failed_post = Post(
            id="post-2",
            shortcode="XYZ999",
            username="failed",
            caption="Another useful caption",
            timestamp=datetime.now(timezone.utc),
        )
        unverified_post = Post(
            id="post-3",
            shortcode="LMN111",
            username="unverified",
            caption="Needs verification",
            timestamp=datetime.now(timezone.utc),
        )
        db.session.add_all([failed_post, unverified_post])
        db.session.flush()
        db.session.add(
            Analysis(
                post_id=failed_post.id,
                category="Business & Career",
                sentiment_score=0.2,
                sentiment_label="Neutral",
                topics=["sales"],
            )
        )
        db.session.add(
            Analysis(
                post_id=unverified_post.id,
                category="Productivity & Systems",
                sentiment_score=0.5,
                sentiment_label="Positive",
                topics=["planning"],
            )
        )
        db.session.add_all(
            [
                PostVerification(
                    post_id=verified.id,
                    provider="tavily_gemini",
                    model="gemini-3.1-flash-lite-preview",
                    status="completed",
                    verdict="supported",
                ),
                PostVerification(
                    post_id=failed_post.id,
                    provider="tavily_gemini",
                    model="gemini-3.1-flash-lite-preview",
                    status="failed",
                    last_error="boom",
                ),
            ]
        )
        db.session.commit()

    verified_response = client.get("/api/posts?verification=verified")
    assert verified_response.status_code == 200
    assert verified_response.get_json()["total"] == 1

    failed_response = client.get("/api/posts?verification=failed")
    assert failed_response.status_code == 200
    assert failed_response.get_json()["total"] == 1

    unverified_response = client.get("/api/posts?verification=unverified")
    assert unverified_response.status_code == 200
    assert unverified_response.get_json()["total"] == 1


def test_task_routes_and_stats(client, app):
    """Task creation, update, and stats should work together."""
    create_response = client.post(
        "/api/tasks",
        json={
            "title": "Ship MVP",
            "priority": 3,
            "source": "manual",
            "next_step": "Write the launch checklist.",
            "effort": "deep",
            "impact": "high",
            "horizon": "this_week",
        },
    )
    assert create_response.status_code == 201
    task = create_response.get_json()["task"]
    assert task["next_step"] == "Write the launch checklist."
    assert task["effort"] == "deep"
    assert task["impact"] == "high"
    assert task["horizon"] == "this_week"

    patch_response = client.patch(
        f"/api/tasks/{task['id']}",
        json={"status": "done"},
    )
    assert patch_response.status_code == 200
    assert patch_response.get_json()["task"]["status"] == "done"

    stats_response = client.get("/api/tasks/stats")
    assert stats_response.status_code == 200
    stats = stats_response.get_json()
    assert stats["done"] == 1
    assert "completion_rate_7d" in stats


def test_export_routes(client, app):
    """Export endpoints should return downloadable content."""
    with app.app_context():
        seed_analyzed_post()
        db.session.add(ActionTask(title="Follow up", status="pending", priority=2))
        db.session.commit()

    json_response = client.get("/api/results/download")
    assert json_response.status_code == 200
    assert json_response.mimetype == "application/json"

    csv_response = client.get("/api/results/csv")
    assert csv_response.status_code == 200
    assert csv_response.mimetype == "text/csv"

    markdown_response = client.get("/api/export")
    assert markdown_response.status_code == 200
    assert markdown_response.mimetype == "text/markdown"


def test_rebuild_rag_requires_analyzed_posts(client):
    """RAG rebuild should fail cleanly when there is no analyzed content."""
    response = client.post("/api/rebuild-rag")
    assert response.status_code == 404
    assert response.get_json()["error"] == "No analyzed posts found"


def test_profile_settings_crud(client, app):
    """Profile settings should round-trip through the singleton workspace profile."""
    response = client.get("/api/profile-settings")
    assert response.status_code == 200
    assert response.get_json()["manual_goals"] == []

    update = client.put(
        "/api/profile-settings",
        json={
            "manual_goals": ["Ship a product"],
            "priorities": ["Health"],
            "constraints": ["Low energy"],
            "focus_areas": ["AI"],
        },
    )
    assert update.status_code == 200
    assert update.get_json()["manual_goals"] == ["Ship a product"]

    fetch = client.get("/api/profile-settings")
    assert fetch.status_code == 200
    assert fetch.get_json()["focus_areas"] == ["AI"]


def test_profile_refresh_requires_analyzed_posts(client):
    """Refreshing the saved profile should fail without analysis."""
    response = client.post("/api/profile/refresh", json={})
    assert response.status_code == 404
    assert response.get_json()["error"] == "No analyzed posts found"


def test_profile_refresh_persists_snapshot(client, app, monkeypatch):
    """Refreshing the profile should store the generated psychometric snapshot."""
    with app.app_context():
        seed_analyzed_post()

    monkeypatch.setattr(
        app_module.AIAnalyzer,
        "generate_psychometric_profile",
        lambda self, posts: {
            "archetype": "Focused Builder",
            "one_liner": "Turns ideas into systems.",
            "traits": ["Curious", "Consistent"],
            "growth_areas": ["Patience"],
            "content_dna": {"primary_focus": "Learning"},
        },
    )

    response = client.post("/api/profile/refresh", json={})
    assert response.status_code == 200
    payload = response.get_json()
    assert payload["psychometric_profile"]["archetype"] == "Focused Builder"

    with app.app_context():
        profile = WorkspaceProfile.get_singleton()
        assert profile.psychometric_profile["archetype"] == "Focused Builder"


def test_verify_requires_provider_config(client, app, monkeypatch):
    """Manual verification should fail clearly when no provider key is configured."""
    from services.verification_service import VerificationSettings

    with app.app_context():
        seed_analyzed_post()

    monkeypatch.setattr(
        app_module.verification_service,
        "build_settings",
        lambda overrides=None: VerificationSettings(
            provider="tavily_gemini",
            model="gemini-3.1-flash-lite-preview",
            api_key="",
            base_url="https://api.openai.com/v1",
            tavily_api_key="",
            max_claims=5,
            max_sources=5,
        ),
    )

    response = client.post("/api/posts/post-1/verify", json={})
    assert response.status_code == 400
    assert "not configured" in response.get_json()["error"]


def test_verify_auto_hydrates_missing_analysis(client, app, monkeypatch):
    """Verification should analyze a post first when structured analysis is missing."""
    with app.app_context():
        post = Post(
            id="verify-auto",
            shortcode="XYZ123",
            username="tester",
            caption="Real caption for verification",
            timestamp=datetime.now(timezone.utc),
        )
        db.session.add(post)
        db.session.commit()

    def fake_analyze_existing_post(post, data=None):
        result = {
            "category": "Education",
            "sentiment": {"score": 0.4, "label": "Positive"},
            "credibility_score": 78,
            "topics": ["focus"],
            "learning_points": ["Try one method consistently"],
            "action_items": ["Test the focus method for a week"],
            "ocr_text": "",
            "video_transcript": "",
            "visual_description": "",
            "video_summary": "",
        }
        analysis = app_module.save_analysis_for_post(post, result, 12)
        db.session.commit()
        return analysis

    monkeypatch.setattr(app_module, "analyze_existing_post", fake_analyze_existing_post)
    monkeypatch.setattr(
        app_module.verification_service,
        "verify_post",
        lambda post, analysis, data=None: {
            "provider": "openai",
            "model": "gpt-4.1",
            "status": "completed",
            "verdict": "mixed",
            "confidence": 0.61,
            "claims": [{"claim": "Focus method works", "verdict": "mixed", "confidence": 0.61, "rationale": "Mixed evidence", "sources": []}],
            "source_links": [{"title": "Study", "url": "https://example.com", "publisher": "Example"}],
            "evidence_summary": "Some evidence supported it, some did not.",
            "raw_report": {"verdict": "mixed"},
        },
    )

    response = client.post(
        "/api/posts/verify-auto/verify",
        json={"provider": "openai", "verification_model": "gpt-4.1", "verification_api_key": "test-key"},
    )
    assert response.status_code == 200
    payload = response.get_json()
    assert payload["status"] == "completed"

    with app.app_context():
        assert Analysis.query.count() == 1
        assert PostVerification.query.count() == 1


def test_verify_rerun_replaces_existing_report(client, app, monkeypatch):
    """Rerunning verification should update the existing row instead of duplicating it."""
    with app.app_context():
        seed_analyzed_post()
        verification = PostVerification(
            post_id="post-1",
            provider="openai",
            model="old-model",
            status="completed",
            verdict="supported",
            confidence=0.9,
            claims=[{"claim": "Old"}],
            source_links=[],
            evidence_summary="Old summary",
            raw_report={"verdict": "supported"},
        )
        db.session.add(verification)
        db.session.commit()

    monkeypatch.setattr(
        app_module.verification_service,
        "verify_post",
        lambda post, analysis, data=None: {
            "provider": "openai",
            "model": "gpt-4.1",
            "status": "completed",
            "verdict": "disputed",
            "confidence": 0.33,
            "claims": [{"claim": "New"}],
            "source_links": [{"title": "Source", "url": "https://example.com", "publisher": "Example"}],
            "evidence_summary": "Updated summary",
            "raw_report": {"verdict": "disputed"},
        },
    )

    response = client.post(
        "/api/posts/post-1/verify",
        json={"provider": "openai", "verification_model": "gpt-4.1", "verification_api_key": "test-key"},
    )
    assert response.status_code == 200

    with app.app_context():
        assert PostVerification.query.count() == 1
        verification = PostVerification.query.filter_by(post_id="post-1").first()
        assert verification.verdict == "disputed"
        assert verification.model == "gpt-4.1"


def test_goal_aware_bootstrap_dedupes_tasks(client, app, monkeypatch):
    """Task bootstrap should create deduped goal-planner tasks."""
    with app.app_context():
        seed_analyzed_post()
        profile = WorkspaceProfile.get_singleton()
        profile.manual_goals = ["Ship a useful product"]
        db.session.commit()

    monkeypatch.setattr(
        app_module.planner_service,
        "generate_tasks",
        lambda profile, posts, limit: [
            {
                "title": "Build a prototype",
                "post_id": "post-1",
                "priority": 3,
                "next_step": "Block 30 minutes to sketch the prototype flow.",
                "evidence_text": "Aligned with your product goal.",
                "notes": "Source: @tester",
                "effort": "medium",
                "impact": "high",
                "horizon": "today",
            },
            {
                "title": "Build a prototype",
                "post_id": "post-1",
                "priority": 2,
                "next_step": "Duplicate",
                "evidence_text": "Duplicate suggestion",
                "notes": "Source: @tester",
                "effort": "quick",
                "impact": "medium",
                "horizon": "later",
            },
        ],
    )

    response = client.post("/api/tasks/bootstrap", json={"limit": 20, "due_days": 7})
    assert response.status_code == 200
    assert response.get_json()["created_count"] == 1

    with app.app_context():
        tasks = ActionTask.query.all()
        assert len(tasks) == 1
        assert tasks[0].source == "goal_planner"
        assert "Aligned with your product goal." in (tasks[0].evidence_text or "")
        assert tasks[0].next_step == "Block 30 minutes to sketch the prototype flow."
        assert tasks[0].effort == "medium"
        assert tasks[0].impact == "high"
        assert tasks[0].horizon == "today"


def test_stats_warn_on_stale_index_without_analysis(client, monkeypatch):
    """Stats should flag stale indexed artifacts without pretending analysis exists."""
    with app.app_context():
        db.session.add(Post(id="placeholder-1", username="ghost", caption="Post by ghost on 2025-01-01"))
        db.session.commit()

    monkeypatch.setattr(
        app_module.rag_service,
        "get_stats",
        lambda: {
            "total_indexed": 25,
            "index_vector_count": 25,
            "index_metadata_count": 12,
            "index_integrity_status": "warning",
            "index_integrity_warning": "FAISS vector count does not match indexed metadata rows.",
            "dimension": 3072,
            "model": "test",
        },
    )

    response = client.get("/api/stats")
    assert response.status_code == 200
    payload = response.get_json()
    assert payload["analyzed_count"] == 0
    assert payload["analysis_count"] == 0
    assert payload["stale_data_detected"] is True
    assert payload["stale_data_warning"]
    assert payload["index_vector_count"] == 25
    assert payload["index_metadata_count"] == 12
    assert payload["index_integrity_status"] == "warning"
    assert payload["index_integrity_warning"]


def test_stats_warn_on_index_metadata_analysis_mismatch(client, app, monkeypatch):
    """Stats should expose index integrity mismatches separately from DB counts."""
    with app.app_context():
        seed_analyzed_post()

    monkeypatch.setattr(
        app_module.rag_service,
        "get_stats",
        lambda: {
            "total_indexed": 10,
            "index_vector_count": 10,
            "index_metadata_count": 4,
            "index_integrity_status": "warning",
            "index_integrity_warning": "FAISS vector count does not match indexed metadata rows.",
            "dimension": 3072,
            "model": "test",
        },
    )

    response = client.get("/api/stats")
    assert response.status_code == 200
    payload = response.get_json()
    assert payload["analysis_count"] == 1
    assert payload["analyzed_count"] == 1
    assert payload["index_vector_count"] == 10
    assert payload["index_metadata_count"] == 4
    assert payload["index_integrity_status"] == "warning"
    assert "Indexed metadata count does not match analyzed post rows" in payload["index_integrity_warning"]


def test_category_refresh_updates_existing_analysis(client, app, monkeypatch):
    """Manual category refresh should update category and tags from stored text fields."""
    with app.app_context():
        seed_analyzed_post()

    monkeypatch.setattr(
        app_module.AIAnalyzer,
        "classify_saved_content",
        lambda self, post, analysis_fields: {
            "category": "Productivity & Systems",
            "topics": ["time blocking", "focus ritual", "weekly planning"],
        },
    )

    response = client.post("/api/admin/refresh-categories", json={})
    assert response.status_code == 200
    payload = response.get_json()
    assert payload["refreshed_count"] == 1

    with app.app_context():
        analysis = Analysis.query.filter_by(post_id="post-1").first()
        assert analysis.category == "Productivity & Systems"
        assert analysis.topics == ["time blocking", "focus ritual", "weekly planning"]
