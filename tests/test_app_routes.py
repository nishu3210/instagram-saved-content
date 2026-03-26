"""Integration tests for Flask routes."""

from datetime import datetime, timezone

from app import app
from database import ActionTask, Analysis, Post, db


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


def test_task_routes_and_stats(client, app):
    """Task creation, update, and stats should work together."""
    create_response = client.post(
        "/api/tasks",
        json={"title": "Ship MVP", "priority": 3, "source": "manual"},
    )
    assert create_response.status_code == 201
    task = create_response.get_json()["task"]

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