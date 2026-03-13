"""RAG (Retrieval Augmented Generation) service for semantic search."""

import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional

import faiss  # type: ignore
import numpy as np
from openai import AzureOpenAI

from config import config
from database import Post

logger = logging.getLogger(__name__)


class RAGService:
    """RAG service for semantic search over posts."""

    def __init__(self):
        self.embedding_model = config.embedding.model
        self.dimension = config.embedding.dimension
        self.index_file = config.app.output_dir / "rag_index.bin"
        self.metadata_file = config.app.output_dir / "indexed_posts.json"
        self.index: Optional[faiss.Index] = None
        self.posts_metadata: List[Dict] = []
        self.client: Optional[AzureOpenAI] = None

        self._load_or_create_index()
        self._init_client()

    def _load_or_create_index(self) -> None:
        """Load existing index or create new one."""
        if self.index_file.exists() and self.metadata_file.exists():
            try:
                self.index = faiss.read_index(str(self.index_file))
                with open(self.metadata_file, "r") as f:
                    self.posts_metadata = json.load(f)

                # Check dimension compatibility
                if self.index.d != self.dimension:
                    logger.warning(
                        f"Dimension mismatch: index={self.index.d}, config={self.dimension}. Rebuilding..."
                    )
                    self._reset_index()
                else:
                    logger.info(f"Loaded index with {self.index.ntotal} vectors")

            except Exception as e:
                logger.error(f"Failed to load index: {e}")
                self._reset_index()
        else:
            self._reset_index()

    def _reset_index(self) -> None:
        """Reset to empty index."""
        self.index = faiss.IndexFlatL2(self.dimension)
        self.posts_metadata = []
        self._cleanup_index_files()

    def _cleanup_index_files(self) -> None:
        """Remove index files."""
        try:
            if self.index_file.exists():
                self.index_file.unlink()
            if self.metadata_file.exists():
                self.metadata_file.unlink()
        except OSError:
            pass

    def _init_client(self) -> None:
        """Initialize Azure OpenAI client."""
        if config.azure.is_configured():
            try:
                self.client = AzureOpenAI(
                    azure_endpoint=config.azure.endpoint,
                    api_key=config.azure.api_key,
                    api_version=config.azure.api_version,
                )
            except Exception as e:
                logger.error(f"Failed to initialize client: {e}")

    def _get_embedding(self, text: str) -> List[float]:
        """Generate embedding for text."""
        if not self.client:
            return [0.0] * self.dimension

        try:
            response = self.client.embeddings.create(
                input=text[:8000],  # Limit text length
                model=self.embedding_model,
            )
            embedding = response.data[0].embedding

            # Verify dimension
            if len(embedding) != self.dimension:
                logger.error(
                    f"Embedding dimension mismatch: {len(embedding)} != {self.dimension}"
                )
                return [0.0] * self.dimension

            return embedding

        except Exception as e:
            logger.error(f"Failed to generate embedding: {e}")
            return [0.0] * self.dimension

    def _save_index(self) -> None:
        """Save index to disk."""
        try:
            faiss.write_index(self.index, str(self.index_file))
            with open(self.metadata_file, "w") as f:
                json.dump(self.posts_metadata, f)
        except Exception as e:
            logger.error(f"Failed to save index: {e}")

    def index_posts(self, posts: List[Post]) -> int:
        """Index posts for search. Returns number of new posts indexed."""
        if not posts or not self.index:
            return 0

        # Get existing IDs
        existing_ids = {p["id"] for p in self.posts_metadata}

        new_embeddings = []
        new_metadata = []

        for post in posts:
            if post.id in existing_ids:
                continue

            # Build context string
            context = self._build_context(post)
            embedding = self._get_embedding(context)

            # Skip if embedding failed
            if all(v == 0.0 for v in embedding):
                continue

            new_embeddings.append(embedding)
            new_metadata.append(
                {
                    "id": post.id,
                    "context_preview": context[:100],
                }
            )

        if new_embeddings:
            # Add to FAISS index
            xb = np.array(new_embeddings).astype("float32")
            self.index.add(xb)
            self.posts_metadata.extend(new_metadata)
            self._save_index()

            logger.info(f"Indexed {len(new_embeddings)} new posts")
            return len(new_embeddings)

        return 0

    def _build_context(self, post: Post) -> str:
        """Build rich context string from post."""
        parts = [
            f"Post by {post.username or 'unknown'}",
            f"Caption: {post.caption or ''}",
        ]

        if post.analysis:
            analysis = post.analysis
            parts.extend(
                [
                    f"Category: {analysis.category}",
                    f"Topics: {', '.join(analysis.topics or [])}",
                    f"Sentiment: {analysis.sentiment_label}",
                ]
            )

            if analysis.action_items:
                parts.append(f"Actions: {', '.join(analysis.action_items)}")

            if analysis.ocr_text:
                parts.append(f"OCR: {analysis.ocr_text[:500]}")

            if analysis.video_transcript:
                parts.append(f"Transcript: {analysis.video_transcript[:1000]}")

        return " | ".join(parts)

    def search(self, query: str, k: int = 5) -> List[Post]:
        """Search for similar posts."""
        if not self.index or self.index.ntotal == 0:
            return []

        query_emb = self._get_embedding(query)
        xq = np.array([query_emb]).astype("float32")

        # Search
        D, I = self.index.search(xq, min(k, self.index.ntotal))

        results = []
        for idx in I[0]:
            if idx == -1 or idx >= len(self.posts_metadata):
                continue

            post_id = self.posts_metadata[idx]["id"]
            post = Post.query.get(post_id)
            if post:
                results.append(post)

        return results

    def get_stats(self) -> Dict:
        """Get RAG statistics."""
        return {
            "total_indexed": self.index.ntotal if self.index else 0,
            "dimension": self.dimension,
            "model": self.embedding_model,
        }
