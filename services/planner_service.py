"""Goal-aware task planning service."""

import json
import logging
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional

from config import config
from database import Post, WorkspaceProfile

try:
    from google import genai
except ImportError:  # pragma: no cover - optional in some test environments
    genai = None

logger = logging.getLogger(__name__)


@dataclass
class CandidateTask:
    """Internal task suggestion."""

    title: str
    post_id: Optional[str]
    priority: int
    score: float
    evidence_text: str
    notes: str
    next_step: str
    effort: str
    impact: str
    horizon: str


class PlannerService:
    """Generate actionable tasks from goals and recent content."""

    def __init__(self):
        self.gemini_model = config.gemini.model
        self.gemini_client = None
        if config.gemini.api_key and genai is not None:
            try:
                self.gemini_client = genai.Client(api_key=config.gemini.api_key)
            except Exception as exc:
                logger.warning(f"Failed to initialize Gemini planner client: {exc}")

    def _coerce_utc(self, value: Optional[datetime]) -> Optional[datetime]:
        """Treat naive datetimes from SQLite as UTC for stable planning logic."""
        if value is None:
            return None
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)

    def generate_tasks(
        self,
        profile: Optional[WorkspaceProfile],
        posts: List[Post],
        limit: int,
    ) -> List[Dict[str, Any]]:
        """Generate ranked task suggestions."""
        if not posts:
            return []

        goal_signals = self._collect_goal_signals(profile)
        gemini_tasks = self._generate_tasks_with_gemini(profile, posts, goal_signals, limit)
        if gemini_tasks:
            return gemini_tasks[:limit]

        logger.info("Falling back to heuristic planner")
        return self._generate_tasks_with_heuristics(profile, posts, goal_signals, limit)

    def _generate_tasks_with_gemini(
        self,
        profile: Optional[WorkspaceProfile],
        posts: List[Post],
        goal_signals: List[Dict[str, Any]],
        limit: int,
    ) -> List[Dict[str, Any]]:
        """Use Gemini to synthesize a richer cross-post plan."""
        if self.gemini_client is None or not self.gemini_model:
            return []

        prompt = self._build_gemini_prompt(profile, posts, goal_signals, limit)
        try:
            response = self.gemini_client.models.generate_content(
                model=self.gemini_model,
                contents=prompt,
                config={"response_mime_type": "application/json"},
            )
            payload = json.loads((getattr(response, "text", "") or "{}").strip())
            suggestions = payload.get("tasks") or []
        except Exception as exc:
            logger.warning(f"Gemini planner failed: {exc}")
            return []

        tasks: List[Dict[str, Any]] = []
        deduped: Dict[str, Dict[str, Any]] = {}
        post_lookup = {post.id: post for post in posts}

        for raw_task in suggestions:
            task = self._normalize_planner_task(raw_task, post_lookup)
            if task is None:
                continue
            key = self._normalize_key(task["title"])
            existing = deduped.get(key)
            if existing is None or task["priority"] > existing["priority"]:
                deduped[key] = task

        tasks = sorted(
            deduped.values(),
            key=lambda item: (
                self._horizon_rank(item["horizon"]),
                item["priority"],
                self._impact_rank(item["impact"]),
                item["title"].lower(),
            ),
        )
        return tasks[:limit]

    def _build_gemini_prompt(
        self,
        profile: Optional[WorkspaceProfile],
        posts: List[Post],
        goal_signals: List[Dict[str, Any]],
        limit: int,
    ) -> str:
        """Build the planning prompt with profile and recent content context."""
        profile_snapshot = profile.psychometric_profile if profile else {}
        goals = {
            "manual_goals": profile.manual_goals if profile else [],
            "priorities": profile.priorities if profile else [],
            "constraints": profile.constraints if profile else [],
            "focus_areas": profile.focus_areas if profile else [],
            "psychometric_profile": profile_snapshot or {},
            "goal_signals": goal_signals,
        }
        post_summaries = []
        for post in posts[:30]:
            analysis = post.analysis
            if analysis is None:
                continue
            post_summaries.append(
                {
                    "post_id": post.id,
                    "username": post.username,
                    "timestamp": self._coerce_utc(post.timestamp).isoformat()
                    if self._coerce_utc(post.timestamp)
                    else None,
                    "category": analysis.category,
                    "topics": (analysis.topics or [])[:6],
                    "learning_points": (analysis.learning_points or [])[:4],
                    "action_items": (analysis.action_items or [])[:4],
                    "video_summary": analysis.video_summary,
                    "caption": (post.caption or "")[:500],
                }
            )

        return f"""
You are planning a practical execution system from a person's saved Instagram content.
Generate at most {limit} genuinely useful tasks. Each task should be:
- aligned to the user's goals, priorities, and psychometric profile when available
- grounded in recent analyzed content
- concrete enough to act on immediately
- not a vague reminder or generic self-help advice

Rules:
- Prefer action-oriented tasks that can be started today or this week.
- Use action_items only as supporting evidence, not the sole source of ideas.
- Deduplicate overlapping ideas across multiple posts.
- Explain why each task matters based on goal alignment and source content.
- next_step must be a clear first move of 1 sentence.
- effort must be one of: quick, medium, deep.
- impact must be one of: low, medium, high.
- horizon must be one of: today, this_week, later.
- priority must be 1, 2, or 3 where 3 is highest.
- notes should be concise and operational, not fluffy.
- evidence_text should mention the goal alignment and the post or reasoning basis.

Return strict JSON only:
{{
  "tasks": [
    {{
      "title": "Task title",
      "post_id": "optional post id from the provided list",
      "priority": 3,
      "next_step": "Concrete first move",
      "notes": "Operational context",
      "evidence_text": "Why this was suggested",
      "effort": "quick",
      "impact": "high",
      "horizon": "today"
    }}
  ]
}}

Workspace profile:
{json.dumps(goals, ensure_ascii=False)}

Recent analyzed posts:
{json.dumps(post_summaries, ensure_ascii=False)}
""".strip()

    def _normalize_planner_task(
        self,
        raw_task: Dict[str, Any],
        post_lookup: Dict[str, Post],
    ) -> Optional[Dict[str, Any]]:
        """Validate and normalize Gemini planner output."""
        title = str(raw_task.get("title", "")).strip()
        next_step = str(raw_task.get("next_step", "")).strip()
        if not title or not next_step:
            return None

        post_id = raw_task.get("post_id")
        if post_id and post_id not in post_lookup:
            post_id = None

        priority = self._normalize_priority(raw_task.get("priority"))
        effort = self._normalize_effort(raw_task.get("effort"))
        impact = self._normalize_impact(raw_task.get("impact"))
        horizon = self._normalize_horizon(raw_task.get("horizon"))
        notes = str(raw_task.get("notes", "")).strip()
        evidence_text = str(raw_task.get("evidence_text", "")).strip()
        if not evidence_text:
            evidence_text = "Suggested from your recent saved content and workspace goals."

        return {
            "title": title[:500],
            "post_id": post_id,
            "priority": priority,
            "next_step": next_step[:500],
            "notes": notes[:1000],
            "evidence_text": evidence_text[:2000],
            "effort": effort,
            "impact": impact,
            "horizon": horizon,
        }

    def _generate_tasks_with_heuristics(
        self,
        profile: Optional[WorkspaceProfile],
        posts: List[Post],
        goal_signals: List[Dict[str, Any]],
        limit: int,
    ) -> List[Dict[str, Any]]:
        """Generate ranked task suggestions with a deterministic fallback."""
        candidates: Dict[str, CandidateTask] = {}

        for post in posts:
            for candidate in self._build_candidates_for_post(post, goal_signals):
                dedupe_key = self._normalize_key(candidate.title)
                existing = candidates.get(dedupe_key)
                if existing is None or candidate.score > existing.score:
                    candidates[dedupe_key] = candidate

        if not candidates and goal_signals:
            for signal in goal_signals[:limit]:
                title = f"Convert {signal['label']} into one concrete weekly action"
                candidates[self._normalize_key(title)] = CandidateTask(
                    title=title,
                    post_id=None,
                    priority=2,
                    score=signal["weight"],
                    evidence_text=(
                        f"Created from your saved priority '{signal['label']}' because there "
                        "was not yet a concrete task tied to it."
                    ),
                    notes="Source: workspace goals",
                    next_step=f"Write one measurable weekly milestone for {signal['label']}.",
                    effort="quick",
                    impact="medium",
                    horizon="this_week",
                )

        ranked = sorted(
            candidates.values(),
            key=lambda item: (
                self._horizon_rank(item.horizon),
                item.priority,
                item.score,
                item.title.lower(),
            ),
        )
        return [
            {
                "title": item.title,
                "post_id": item.post_id,
                "priority": item.priority,
                "evidence_text": item.evidence_text,
                "notes": item.notes,
                "next_step": item.next_step,
                "effort": item.effort,
                "impact": item.impact,
                "horizon": item.horizon,
            }
            for item in ranked[:limit]
        ]

    def _build_candidates_for_post(
        self,
        post: Post,
        goal_signals: List[Dict[str, Any]],
    ) -> List[CandidateTask]:
        """Generate candidate tasks from one analyzed post."""
        if not post.analysis:
            return []

        analysis = post.analysis
        recent_bonus = self._recency_bonus(post.timestamp)
        supporting_topics = [topic for topic in (analysis.topics or []) if topic][:4]
        candidates: List[CandidateTask] = []

        for action_item in analysis.action_items or []:
            score, matches = self._score_text(action_item, analysis, goal_signals)
            task_title = action_item[:500]
            candidates.append(
                CandidateTask(
                    title=task_title,
                    post_id=post.id,
                    priority=self._priority_from_score(score + recent_bonus),
                    score=score + recent_bonus + 4.0,
                    evidence_text=self._build_evidence_text(
                        post,
                        action_item,
                        supporting_topics,
                        matches,
                        "action item",
                    ),
                    notes=f"Source: @{post.username or 'unknown'}",
                    next_step=self._build_next_step(action_item),
                    effort=self._effort_from_text(action_item),
                    impact=self._impact_from_score(score + recent_bonus + 1.0),
                    horizon=self._horizon_from_score(score + recent_bonus + 1.0),
                )
            )

        if not candidates:
            for learning_point in (analysis.learning_points or [])[:2]:
                if not learning_point.strip():
                    continue
                title = f"Practice: {learning_point[:140]}"
                score, matches = self._score_text(learning_point, analysis, goal_signals)
                candidates.append(
                    CandidateTask(
                        title=title[:500],
                        post_id=post.id,
                        priority=self._priority_from_score(score + recent_bonus),
                        score=score + recent_bonus + 2.0,
                        evidence_text=self._build_evidence_text(
                            post,
                            learning_point,
                            supporting_topics,
                            matches,
                            "learning point",
                        ),
                        notes=f"Source: @{post.username or 'unknown'}",
                        next_step=self._build_next_step(learning_point),
                        effort="medium",
                        impact=self._impact_from_score(score + recent_bonus),
                        horizon=self._horizon_from_score(score + recent_bonus),
                    )
                )

        if not candidates and supporting_topics:
            seed_text = supporting_topics[0]
            title = f"Turn {seed_text} into a concrete next step"
            score, matches = self._score_text(seed_text, analysis, goal_signals)
            candidates.append(
                CandidateTask(
                    title=title[:500],
                    post_id=post.id,
                    priority=self._priority_from_score(score + recent_bonus),
                    score=score + recent_bonus + 1.0,
                    evidence_text=self._build_evidence_text(
                        post,
                        seed_text,
                        supporting_topics,
                        matches,
                        "topic",
                    ),
                    notes=f"Source: @{post.username or 'unknown'}",
                    next_step=f"Turn {seed_text} into a single measurable experiment this week.",
                    effort="quick",
                    impact=self._impact_from_score(score),
                    horizon=self._horizon_from_score(score),
                )
            )

        return candidates

    def _collect_goal_signals(
        self, profile: Optional[WorkspaceProfile]
    ) -> List[Dict[str, Any]]:
        """Collect weighted planning signals from manual goals and profile."""
        signals: List[Dict[str, Any]] = []
        if profile is None:
            return signals

        for label in profile.manual_goals or []:
            signals.append({"label": label, "weight": 5.0, "source": "manual_goal"})
        for label in profile.priorities or []:
            signals.append({"label": label, "weight": 4.5, "source": "priority"})
        for label in profile.focus_areas or []:
            signals.append({"label": label, "weight": 4.0, "source": "focus_area"})
        for label in profile.constraints or []:
            signals.append({"label": label, "weight": 1.5, "source": "constraint"})

        snapshot = profile.psychometric_profile or {}
        for label in snapshot.get("growth_areas") or []:
            signals.append({"label": label, "weight": 3.5, "source": "growth_area"})
        content_dna = snapshot.get("content_dna") or {}
        for key in ("primary_focus", "secondary_focus"):
            label = content_dna.get(key)
            if label:
                signals.append({"label": label, "weight": 3.0, "source": "content_dna"})

        deduped: Dict[str, Dict[str, Any]] = {}
        for signal in signals:
            key = self._normalize_key(signal["label"])
            if key not in deduped or signal["weight"] > deduped[key]["weight"]:
                deduped[key] = signal
        return list(deduped.values())

    def _score_text(
        self,
        text: str,
        analysis,
        goal_signals: List[Dict[str, Any]],
    ) -> tuple[float, List[str]]:
        """Score a text candidate against goal signals and analysis context."""
        corpus = " ".join(
            [
                text,
                analysis.category or "",
                " ".join(analysis.topics or []),
                " ".join(analysis.learning_points or []),
                analysis.video_summary or "",
            ]
        )
        corpus_tokens = set(self._tokenize(corpus))
        score = 0.0
        matches: List[str] = []
        for signal in goal_signals:
            signal_tokens = set(self._tokenize(signal["label"]))
            if signal_tokens and corpus_tokens.intersection(signal_tokens):
                score += signal["weight"]
                matches.append(signal["label"])
        if analysis.category:
            score += 0.5
        if analysis.video_summary:
            score += 0.5
        return score, matches[:3]

    def _build_evidence_text(
        self,
        post: Post,
        seed_text: str,
        topics: List[str],
        matches: List[str],
        origin: str,
    ) -> str:
        """Build user-facing evidence text."""
        timestamp = self._coerce_utc(post.timestamp)
        when = timestamp.strftime("%Y-%m-%d") if timestamp else "unknown date"
        reasons = []
        if matches:
            reasons.append(f"aligns with {', '.join(matches)}")
        if topics:
            reasons.append(f"topics: {', '.join(topics[:3])}")
        reasons.append(f"derived from {origin} on @{post.username or 'unknown'}")
        return (
            f"Suggested from '{seed_text[:180]}' because it {'; '.join(reasons)} "
            f"from a recent saved post dated {when}."
        )

    def _build_next_step(self, seed_text: str) -> str:
        """Turn a task seed into a usable first step."""
        cleaned = seed_text.strip().rstrip(".")
        if not cleaned:
            return "Define one concrete first step and schedule it."
        if len(cleaned.split()) <= 3:
            return f"Write a 15-minute starter step for {cleaned.lower()} and schedule it."
        return f"Start by scoping the first 15-minute move for: {cleaned[:180]}."

    def _priority_from_score(self, score: float) -> int:
        """Map score to task priority."""
        if score >= 8:
            return 3
        if score >= 4:
            return 2
        return 1

    def _impact_from_score(self, score: float) -> str:
        """Map score to an impact bucket."""
        if score >= 7:
            return "high"
        if score >= 3:
            return "medium"
        return "low"

    def _horizon_from_score(self, score: float) -> str:
        """Map score to a delivery horizon."""
        if score >= 7:
            return "today"
        if score >= 3:
            return "this_week"
        return "later"

    def _effort_from_text(self, value: str) -> str:
        """Estimate effort based on language cues."""
        lowered = value.lower()
        if any(word in lowered for word in ("build", "launch", "design", "write", "create", "system")):
            return "deep"
        if any(word in lowered for word in ("review", "outline", "plan", "test", "practice")):
            return "medium"
        return "quick"

    def _normalize_priority(self, value: Any) -> int:
        """Normalize priority input."""
        try:
            priority = int(value)
        except (TypeError, ValueError):
            return 2
        return max(1, min(3, priority))

    def _normalize_effort(self, value: Any) -> str:
        """Normalize effort input."""
        effort = str(value or "medium").strip().lower()
        return effort if effort in {"quick", "medium", "deep"} else "medium"

    def _normalize_impact(self, value: Any) -> str:
        """Normalize impact input."""
        impact = str(value or "medium").strip().lower()
        return impact if impact in {"low", "medium", "high"} else "medium"

    def _normalize_horizon(self, value: Any) -> str:
        """Normalize horizon input."""
        horizon = str(value or "this_week").strip().lower()
        return horizon if horizon in {"today", "this_week", "later"} else "this_week"

    def _horizon_rank(self, horizon: str) -> int:
        """Sort today first, then this week, then later."""
        return {"today": 0, "this_week": 1, "later": 2}.get(horizon, 3)

    def _impact_rank(self, impact: str) -> int:
        """Sort high impact first."""
        return {"high": 0, "medium": 1, "low": 2}.get(impact, 3)

    def _recency_bonus(self, timestamp: Optional[datetime]) -> float:
        """Favor recent saves."""
        normalized_timestamp = self._coerce_utc(timestamp)
        if normalized_timestamp is None:
            return 0.0
        now = datetime.now(timezone.utc)
        delta_days = max((now - normalized_timestamp).days, 0)
        if delta_days <= 7:
            return 2.0
        if delta_days <= 30:
            return 1.0
        return 0.0

    def _tokenize(self, value: str) -> Iterable[str]:
        """Tokenize text for lightweight matching."""
        return [
            token
            for token in re.split(r"[^a-zA-Z0-9]+", value.lower())
            if len(token) > 2
        ]

    def _normalize_key(self, value: str) -> str:
        """Normalize text for dedupe."""
        return " ".join(self._tokenize(value))[:200]
