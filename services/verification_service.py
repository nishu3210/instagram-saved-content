"""Grounded verification service with pluggable providers."""

import json
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

import requests

from config import config
from database import Analysis, Post

try:
    from google import genai
except ImportError:  # pragma: no cover - optional dependency in some test environments
    genai = None

logger = logging.getLogger(__name__)


class VerificationServiceError(Exception):
    """Raised when verification cannot be completed."""

    pass


@dataclass(frozen=True)
class VerificationSettings:
    """Verification runtime settings."""

    provider: str
    model: str
    api_key: str
    base_url: str
    tavily_api_key: str
    max_claims: int
    max_sources: int

    @classmethod
    def from_overrides(cls, overrides: Optional[Dict[str, Any]] = None) -> "VerificationSettings":
        """Build settings from config plus optional request overrides."""
        overrides = overrides or {}
        provider = (overrides.get("provider") or config.verification.provider or "").strip()
        default_model = (
            config.gemini.model if provider == "tavily_gemini" else config.verification.model
        )
        model = (
            overrides.get("verification_model")
            or overrides.get("model_override")
            or default_model
        )
        api_key = overrides.get("verification_api_key")
        if not api_key and provider == "tavily_gemini":
            api_key = config.gemini.api_key or config.verification.api_key
        elif not api_key:
            api_key = config.verification.api_key

        return cls(
            provider=provider,
            model=str(model or "").strip(),
            api_key=str(api_key or "").strip(),
            base_url=str(
                overrides.get("verification_base_url") or config.verification.base_url
            ).rstrip("/"),
            tavily_api_key=str(
                overrides.get("tavily_api_key") or config.verification.tavily_api_key or ""
            ).strip(),
            max_claims=max(
                1,
                min(
                    10,
                    int(overrides.get("max_claims") or config.verification.max_claims),
                ),
            ),
            max_sources=max(
                1,
                min(
                    10,
                    int(overrides.get("max_sources") or config.verification.max_sources),
                ),
            ),
        )

    def is_configured(self) -> bool:
        """Check if settings are usable."""
        if self.provider == "tavily_gemini":
            return bool(
                self.provider
                and self.model
                and self.api_key
                and self.tavily_api_key
            )
        return bool(self.provider and self.model and self.api_key)


class BaseVerificationProvider:
    """Provider contract for grounded verification."""

    name = "base"

    def verify(self, prompt: str, settings: VerificationSettings) -> Dict[str, Any]:
        """Verify claims with grounded web research."""
        raise NotImplementedError


class OpenAIGroundedVerificationProvider(BaseVerificationProvider):
    """OpenAI Responses API backed verification provider."""

    name = "openai"

    def verify(self, prompt: str, settings: VerificationSettings) -> Dict[str, Any]:
        """Call the OpenAI Responses API with web search enabled."""
        payload = {
            "model": settings.model,
            "tools": [{"type": "web_search_preview"}],
            "input": [
                {
                    "role": "system",
                    "content": [
                        {
                            "type": "input_text",
                            "text": (
                                "You are a careful fact-checking assistant. "
                                "Research the most important factual or advice-based claims, "
                                "cite high-quality web evidence, and respond with strict JSON only."
                            ),
                        }
                    ],
                },
                {
                    "role": "user",
                    "content": [{"type": "input_text", "text": prompt}],
                },
            ],
        }
        response = requests.post(
            f"{settings.base_url}/responses",
            headers={
                "Authorization": f"Bearer {settings.api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=90,
        )
        if response.status_code >= 400:
            raise VerificationServiceError(
                f"Verification provider error ({response.status_code}): {response.text[:300]}"
            )

        data = response.json()
        text = data.get("output_text") or self._extract_output_text(data)
        if not text:
            raise VerificationServiceError("Verification provider returned no text output")

        try:
            return json.loads(text.strip())
        except json.JSONDecodeError as exc:
            raise VerificationServiceError(
                f"Verification provider returned invalid JSON: {exc}"
            ) from exc

    def _extract_output_text(self, response_data: Dict[str, Any]) -> str:
        """Extract text from fallback response shapes."""
        output = response_data.get("output") or []
        chunks: List[str] = []
        for item in output:
            for content in item.get("content") or []:
                if content.get("type") in {"output_text", "text"} and content.get("text"):
                    chunks.append(str(content["text"]))
        return "\n".join(chunks)


class TavilyGeminiVerificationProvider(BaseVerificationProvider):
    """Tavily search plus Gemini synthesis verification provider."""

    name = "tavily_gemini"
    tavily_search_url = "https://api.tavily.com/search"

    def verify(self, prompt: str, settings: VerificationSettings) -> Dict[str, Any]:
        """Research claims with Tavily, then synthesize a report with Gemini."""
        if genai is None:
            raise VerificationServiceError(
                "Gemini verification requires the google-genai package."
            )

        try:
            client = genai.Client(api_key=settings.api_key)
        except Exception as exc:
            raise self._wrap_gemini_error(exc) from exc

        claims = self._extract_claims(client, prompt, settings)
        if not claims:
            return {
                "verdict": "insufficient_evidence",
                "confidence": 0.2,
                "evidence_summary": (
                    "No factual or advice-oriented claims were strong enough to justify external verification."
                ),
                "claims": [],
                "source_links": [],
            }

        research_packets = [
            self._search_claim(claim, settings) for claim in claims[: settings.max_claims]
        ]
        synthesis_prompt = self._build_synthesis_prompt(prompt, research_packets, settings)
        return self._generate_json(
            client,
            settings.model,
            synthesis_prompt,
            self._report_schema(),
            "verification report",
        )

    def _extract_claims(self, client, prompt: str, settings: VerificationSettings) -> List[Dict[str, str]]:
        """Extract the top check-worthy claims from the post context."""
        extraction_prompt = f"""
Review the Instagram post context below and extract at most {settings.max_claims} factual or advice-based claims worth verifying with web research.

Only include claims that could plausibly be checked against reputable sources.
Skip purely personal opinions, vague motivation, or aesthetic statements.
Prefer claims that are concrete, actionable, health/business/productivity related, or presented as true.

Return strict JSON only.

Post context:
{prompt}
""".strip()

        payload = self._generate_json(
            client,
            settings.model,
            extraction_prompt,
            self._claim_schema(settings.max_claims),
            "claim extraction",
        )
        claims = payload.get("claims") or []
        normalized: List[Dict[str, str]] = []
        for item in claims:
            if not isinstance(item, dict):
                continue
            claim = str(item.get("claim") or "").strip()
            query = str(item.get("query") or claim).strip()
            why_check = str(item.get("why_check") or "").strip()
            if claim and query:
                normalized.append(
                    {"claim": claim[:500], "query": query[:500], "why_check": why_check[:240]}
                )
        return normalized

    def _search_claim(self, claim: Dict[str, str], settings: VerificationSettings) -> Dict[str, Any]:
        """Research one extracted claim with Tavily."""
        response = requests.post(
            self.tavily_search_url,
            headers={
                "Authorization": f"Bearer {settings.tavily_api_key}",
                "Content-Type": "application/json",
            },
            json={
                "query": claim["query"],
                "topic": "general",
                "search_depth": "advanced",
                "max_results": max(3, settings.max_sources),
                "include_answer": False,
                "include_raw_content": True,
            },
            timeout=60,
        )
        if response.status_code >= 400:
            raise VerificationServiceError(
                f"Tavily search error ({response.status_code}): {response.text[:300]}"
            )

        payload = response.json()
        results: List[Dict[str, Any]] = []
        for item in payload.get("results") or []:
            if not isinstance(item, dict):
                continue
            url = str(item.get("url") or "").strip()
            if not url:
                continue
            results.append(
                {
                    "title": str(item.get("title") or url).strip()[:200],
                    "url": url[:1000],
                    "publisher": self._publisher_from_result(item),
                    "content": str(item.get("content") or "").strip()[:1200],
                    "raw_content": str(item.get("raw_content") or "").strip()[:3000],
                    "score": item.get("score"),
                }
            )

        return {
            "claim": claim["claim"],
            "query": claim["query"],
            "why_check": claim.get("why_check", ""),
            "results": results[: settings.max_sources],
        }

    def _build_synthesis_prompt(
        self,
        prompt: str,
        research_packets: List[Dict[str, Any]],
        settings: VerificationSettings,
    ) -> str:
        """Build the Gemini prompt that turns evidence into a stable report."""
        return f"""
You are a careful fact-checking assistant.

Use only the supplied Tavily research results to assess the claims extracted from this Instagram post.
If the evidence is weak, mixed, missing, anecdotal, or off-topic, prefer "insufficient_evidence" or "mixed".
Do not invent sources. Only cite URLs that appear in the research packets.
Keep at most {settings.max_sources} unique source links in the final response.
Respond with strict JSON only.

Original post context:
{prompt}

Research packets:
{json.dumps(research_packets, ensure_ascii=False)}
""".strip()

    def _generate_json(
        self,
        client,
        model: str,
        prompt: str,
        schema: Dict[str, Any],
        purpose: str,
    ) -> Dict[str, Any]:
        """Generate strict JSON with Gemini."""
        try:
            response = client.models.generate_content(
                model=model,
                contents=prompt,
                config={
                    "response_mime_type": "application/json",
                    "response_schema": schema,
                },
            )
        except Exception as exc:
            raise self._wrap_gemini_error(exc) from exc

        content = getattr(response, "text", "") or ""
        try:
            return json.loads(content.strip() or "{}")
        except json.JSONDecodeError as exc:
            raise VerificationServiceError(
                f"Gemini returned invalid JSON for {purpose}: {exc}"
            ) from exc

    def _publisher_from_result(self, item: Dict[str, Any]) -> str:
        """Infer a readable publisher from Tavily results."""
        for key in ("site_name", "source", "domain"):
            value = str(item.get(key) or "").strip()
            if value:
                return value[:120]

        url = str(item.get("url") or "").strip()
        hostname = urlparse(url).hostname or ""
        return hostname.replace("www.", "")[:120]

    def _claim_schema(self, max_claims: int) -> Dict[str, Any]:
        """Return the JSON schema for claim extraction."""
        return {
            "type": "object",
            "properties": {
                "claims": {
                    "type": "array",
                    "maxItems": max_claims,
                    "items": {
                        "type": "object",
                        "properties": {
                            "claim": {"type": "string"},
                            "query": {"type": "string"},
                            "why_check": {"type": "string"},
                        },
                        "required": ["claim", "query", "why_check"],
                    },
                }
            },
            "required": ["claims"],
        }

    def _report_schema(self) -> Dict[str, Any]:
        """Return the JSON schema for the final verification report."""
        source_schema = {
            "type": "object",
            "properties": {
                "title": {"type": "string"},
                "url": {"type": "string"},
                "publisher": {"type": "string"},
            },
            "required": ["title", "url", "publisher"],
        }
        claim_schema = {
            "type": "object",
            "properties": {
                "claim": {"type": "string"},
                "verdict": {
                    "type": "string",
                    "enum": [
                        "supported",
                        "mixed",
                        "disputed",
                        "insufficient_evidence",
                    ],
                },
                "confidence": {"type": "number"},
                "rationale": {"type": "string"},
                "sources": {"type": "array", "items": source_schema},
            },
            "required": ["claim", "verdict", "confidence", "rationale", "sources"],
        }
        return {
            "type": "object",
            "properties": {
                "verdict": {
                    "type": "string",
                    "enum": [
                        "supported",
                        "mixed",
                        "disputed",
                        "insufficient_evidence",
                    ],
                },
                "confidence": {"type": "number"},
                "evidence_summary": {"type": "string"},
                "claims": {"type": "array", "items": claim_schema},
                "source_links": {"type": "array", "items": source_schema},
            },
            "required": [
                "verdict",
                "confidence",
                "evidence_summary",
                "claims",
                "source_links",
            ],
        }

    def _wrap_gemini_error(self, exc: Exception) -> VerificationServiceError:
        """Convert Gemini SDK failures into user-actionable verification errors."""
        message = str(exc)
        if "API key not valid" in message or "API_KEY_INVALID" in message:
            return VerificationServiceError(
                "Gemini verification API key is invalid. Clear the 'LLM API Key' field "
                "in Settings to use the server GEMINI_API_KEY, or paste a valid Gemini API key."
            )
        return VerificationServiceError(f"Gemini verification failed: {message}")


class VerificationService:
    """High-level verification orchestrator."""

    def __init__(self):
        self._providers = {
            OpenAIGroundedVerificationProvider.name: OpenAIGroundedVerificationProvider(),
            TavilyGeminiVerificationProvider.name: TavilyGeminiVerificationProvider(),
        }

    def get_provider_names(self) -> List[str]:
        """Return supported provider names."""
        return sorted(self._providers.keys())

    def build_settings(self, overrides: Optional[Dict[str, Any]] = None) -> VerificationSettings:
        """Resolve runtime settings."""
        return VerificationSettings.from_overrides(overrides)

    def verify_post(
        self,
        post: Post,
        analysis: Analysis,
        overrides: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Run grounded verification for a post."""
        settings = self.build_settings(overrides)
        if not settings.is_configured():
            raise VerificationServiceError(
                "Verification provider is not configured. Add a provider, model, and API key in Settings."
            )

        provider = self._providers.get(settings.provider)
        if provider is None:
            raise VerificationServiceError(
                f"Unsupported verification provider: {settings.provider}"
            )

        prompt = self._build_prompt(post, analysis, settings)
        raw_report = provider.verify(prompt, settings)
        normalized = self._normalize_report(raw_report, settings)
        normalized["provider"] = settings.provider
        normalized["model"] = settings.model
        normalized["raw_report"] = raw_report
        return normalized

    def _build_prompt(
        self, post: Post, analysis: Analysis, settings: VerificationSettings
    ) -> str:
        """Build a grounded verification prompt."""
        return f"""
Verify the most important factual or advice-oriented claims in this Instagram post.

Rules:
- Research at most {settings.max_claims} claims.
- Use grounded web research.
- Prefer reputable and directly relevant sources.
- Keep at most {settings.max_sources} unique source links in the final summary.
- If content is motivational/opinion-heavy and not meaningfully verifiable, say so and mark verdict as "insufficient_evidence".
- Respond with valid JSON only in this shape:
{{
  "verdict": "supported|mixed|disputed|insufficient_evidence",
  "confidence": 0.0,
  "evidence_summary": "short summary",
  "claims": [
    {{
      "claim": "text",
      "verdict": "supported|mixed|disputed|insufficient_evidence",
      "confidence": 0.0,
      "rationale": "short explanation",
      "sources": [
        {{"title": "source title", "url": "https://...", "publisher": "site"}}
      ]
    }}
  ],
  "source_links": [
    {{"title": "source title", "url": "https://...", "publisher": "site"}}
  ]
}}

Post details:
- Username: {post.username or "unknown"}
- Caption: {post.caption or ""}
- Category: {analysis.category or "Other"}
- Topics: {", ".join(analysis.topics or [])}
- Learning points: {" | ".join(analysis.learning_points or [])}
- Action items: {" | ".join(analysis.action_items or [])}
- OCR text: {analysis.ocr_text or ""}
- Visual description: {analysis.visual_description or ""}
- Video transcript: {analysis.video_transcript or ""}
- Video summary: {analysis.video_summary or ""}
""".strip()

    def _normalize_report(
        self, report: Dict[str, Any], settings: VerificationSettings
    ) -> Dict[str, Any]:
        """Normalize provider output into a stable shape."""
        claims: List[Dict[str, Any]] = []
        seen_sources = set()
        source_links: List[Dict[str, str]] = []

        for item in report.get("claims") or []:
            if not isinstance(item, dict):
                continue
            claim_sources = self._normalize_sources(item.get("sources"))
            for source in claim_sources:
                key = source.get("url") or json.dumps(source, sort_keys=True)
                if key and key not in seen_sources and len(source_links) < settings.max_sources:
                    seen_sources.add(key)
                    source_links.append(source)
            claims.append(
                {
                    "claim": str(item.get("claim") or "").strip(),
                    "verdict": self._normalize_verdict(item.get("verdict")),
                    "confidence": self._normalize_confidence(item.get("confidence")),
                    "rationale": str(item.get("rationale") or "").strip(),
                    "sources": claim_sources[: settings.max_sources],
                }
            )
            if len(claims) >= settings.max_claims:
                break

        extra_sources = self._normalize_sources(report.get("source_links"))
        for source in extra_sources:
            key = source.get("url") or json.dumps(source, sort_keys=True)
            if key and key not in seen_sources and len(source_links) < settings.max_sources:
                seen_sources.add(key)
                source_links.append(source)

        return {
            "status": "completed",
            "verdict": self._normalize_verdict(report.get("verdict")),
            "confidence": self._normalize_confidence(report.get("confidence")),
            "evidence_summary": str(report.get("evidence_summary") or "").strip(),
            "claims": claims,
            "source_links": source_links,
        }

    def _normalize_sources(self, sources: Any) -> List[Dict[str, str]]:
        """Normalize source metadata."""
        normalized: List[Dict[str, str]] = []
        if not isinstance(sources, list):
            return normalized
        for source in sources:
            if isinstance(source, str):
                url = source.strip()
                if url:
                    normalized.append({"title": url, "url": url, "publisher": ""})
                continue
            if not isinstance(source, dict):
                continue
            url = str(source.get("url") or "").strip()
            title = str(source.get("title") or url or "Source").strip()
            publisher = str(source.get("publisher") or "").strip()
            if url:
                normalized.append(
                    {"title": title[:200], "url": url[:1000], "publisher": publisher[:120]}
                )
        return normalized

    def _normalize_verdict(self, value: Any) -> str:
        """Normalize verdict values."""
        verdict = str(value or "insufficient_evidence").strip().lower()
        allowed = {"supported", "mixed", "disputed", "insufficient_evidence"}
        return verdict if verdict in allowed else "insufficient_evidence"

    def _normalize_confidence(self, value: Any) -> float:
        """Normalize confidence to 0..1."""
        try:
            confidence = float(value)
        except (TypeError, ValueError):
            confidence = 0.0
        return max(0.0, min(1.0, confidence))
