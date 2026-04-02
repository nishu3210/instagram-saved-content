"""Tests for the richer goal-aware planner."""

from datetime import datetime, timedelta, timezone

from database import Analysis, Post, WorkspaceProfile, db
from services.planner_service import PlannerService


def _make_post(post_id: str, category: str, topics, action_items, days_ago: int = 1):
    """Create a simple analyzed post for planner tests."""
    timestamp = datetime.now(timezone.utc) - timedelta(days=days_ago)
    post = Post(
        id=post_id,
        username="tester",
        caption="Useful reel",
        timestamp=timestamp,
    )
    db.session.add(post)
    db.session.flush()
    db.session.add(
        Analysis(
            post_id=post.id,
            category=category,
            sentiment_score=0.6,
            sentiment_label="Positive",
            topics=topics,
            learning_points=["One useful insight"],
            action_items=action_items,
            video_summary="A concise summary of the saved reel.",
        )
    )
    db.session.commit()
    return post


def test_gemini_planner_path_returns_structured_tasks(app, monkeypatch):
    """Gemini planner output should be normalized into the richer task shape."""
    with app.app_context():
        post = _make_post(
            "planner-1",
            "AI & Tech",
            ["automation", "ai workflows"],
            ["Build an AI workflow prototype"],
        )
        profile = WorkspaceProfile.get_singleton()
        profile.manual_goals = ["Ship an AI product"]
        db.session.commit()

        service = PlannerService()
        monkeypatch.setattr(
            service,
            "_generate_tasks_with_gemini",
            lambda profile, posts, goal_signals, limit: [
                {
                    "title": "Prototype the AI workflow",
                    "post_id": post.id,
                    "priority": 3,
                    "next_step": "Draft the first workflow outline in a doc.",
                    "notes": "Focus on the highest-friction use case first.",
                    "evidence_text": "Matches your AI product goal and recent automation saves.",
                    "effort": "medium",
                    "impact": "high",
                    "horizon": "today",
                }
            ],
        )

        tasks = service.generate_tasks(profile, [post], limit=5)
        assert len(tasks) == 1
        assert tasks[0]["title"] == "Prototype the AI workflow"
        assert tasks[0]["next_step"] == "Draft the first workflow outline in a doc."
        assert tasks[0]["effort"] == "medium"
        assert tasks[0]["impact"] == "high"
        assert tasks[0]["horizon"] == "today"


def test_planner_falls_back_to_heuristics(app, monkeypatch):
    """When Gemini is unavailable, the planner should still produce structured tasks."""
    with app.app_context():
        post = _make_post(
            "planner-2",
            "Productivity & Systems",
            ["weekly planning", "time blocking"],
            ["Build a weekly review ritual"],
        )
        profile = WorkspaceProfile.get_singleton()
        profile.manual_goals = ["Get consistent with planning"]
        db.session.commit()

        service = PlannerService()
        monkeypatch.setattr(service, "_generate_tasks_with_gemini", lambda *args, **kwargs: [])

        tasks = service.generate_tasks(profile, [post], limit=5)
        assert len(tasks) == 1
        assert tasks[0]["title"] == "Build a weekly review ritual"
        assert tasks[0]["next_step"]
        assert tasks[0]["effort"] in {"quick", "medium", "deep"}
        assert tasks[0]["impact"] in {"low", "medium", "high"}
        assert tasks[0]["horizon"] in {"today", "this_week", "later"}


def test_planner_returns_empty_when_no_posts(app):
    """Planner should return no tasks when there is no analyzed content."""
    with app.app_context():
        service = PlannerService()
        profile = WorkspaceProfile.get_singleton()
        tasks = service.generate_tasks(profile, [], limit=5)
        assert tasks == []
