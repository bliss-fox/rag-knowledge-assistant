"""Cross-Encoder based Reranker implementation.

This module implements reranking using Cross-Encoder models that directly score
(query, passage) pairs. Supports both local models via sentence-transformers
and API-based endpoints.
"""

from __future__ import annotations

import logging
import os
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeoutError
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.core.settings import resolve_path
from src.libs.reranker.base_reranker import BaseReranker

logger = logging.getLogger(__name__)


class CrossEncoderRerankError(RuntimeError):
    """Raised when Cross-Encoder reranking fails."""


class CrossEncoderReranker(BaseReranker):
    """Cross-Encoder based reranker for scoring query-passage pairs.
    
    This implementation uses Cross-Encoder models (e.g., ms-marco-MiniLM)
    that directly encode and score (query, passage) pairs, providing more
    accurate relevance scores than bi-encoder approaches at the cost of
    higher computational requirements.
    
    Design Principles Applied:
    - Pluggable: Can be swapped with other reranker implementations via factory.
    - Config-Driven: Model name and parameters come from settings.yaml.
    - Observable: Supports TraceContext for monitoring (Stage F integration).
    - Fallback-Aware: Provides timeout/failure signals for upstream fallback.
    - Deterministic Testing: Supports mock scorer injection for testing.
    """
    
    def __init__(
        self,
        settings: Any,
        model: Optional[Any] = None,
        timeout: float = 10.0,
        **kwargs: Any
    ) -> None:
        """Initialize the Cross-Encoder Reranker.
        
        Args:
            settings: Application settings containing rerank configuration.
            model: Optional pre-initialized CrossEncoder model. If None, creates
                from settings.rerank.model. Used for testing to inject mock models.
            timeout: Maximum time (seconds) to wait for reranking. Default 10s.
                Used to enable fallback strategies when reranking takes too long.
            **kwargs: Additional provider-specific parameters.
        """
        self.settings = settings
        self.timeout = timeout
        self.kwargs = kwargs
        self.device = self._resolve_device(
            kwargs.get("device", getattr(settings.rerank, "device", "auto"))
        )
        cache_dir = kwargs.get(
            "cache_dir", getattr(settings.rerank, "cache_dir", "./data/models/huggingface")
        )
        self.cache_dir = self._resolve_cache_dir(cache_dir)
        
        # Initialize or inject model
        if model is not None:
            self.model = model
        else:
            try:
                model_name = self._get_model_name_from_settings(settings)
                self.model = self._load_cross_encoder_model(model_name)
            except Exception as e:
                raise CrossEncoderRerankError(
                    f"Failed to initialize Cross-Encoder model: {e}"
                ) from e
    
    def _get_model_name_from_settings(self, settings: Any) -> str:
        """Extract model name from settings.
        
        Args:
            settings: Application settings.
        
        Returns:
            Model name string.
        
        Raises:
            AttributeError: If rerank.model is not configured.
        """
        try:
            model_name = settings.rerank.model
            if not model_name or not isinstance(model_name, str):
                raise ValueError("Model name must be a non-empty string")
            return model_name
        except AttributeError as e:
            raise AttributeError(
                "Missing configuration: settings.rerank.model. "
                "Please specify 'rerank.model' in settings.yaml"
            ) from e
    
    def _load_cross_encoder_model(self, model_name: str) -> Any:
        """Load the Cross-Encoder model.
        
        Args:
            model_name: Name or path of the Cross-Encoder model.
        
        Returns:
            Initialized CrossEncoder instance.
        
        Raises:
            ImportError: If sentence-transformers is not installed.
            RuntimeError: If model loading fails.
        """
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as e:
            raise ImportError(
                "sentence-transformers is required for Cross-Encoder reranking. "
                "Install it with: pip install sentence-transformers"
            ) from e
        
        try:
            logger.info(f"Loading Cross-Encoder model: {model_name}")
            logger.info("Cross-Encoder device: %s", self.device)
            if self.cache_dir is not None:
                self.cache_dir.mkdir(parents=True, exist_ok=True)
            model = CrossEncoder(
                model_name, device=self.device,
                cache_folder=str(self.cache_dir) if self.cache_dir is not None else None,
            )
            logger.info(f"Cross-Encoder model loaded successfully: {model_name}")
            return model
        except Exception as e:
            raise RuntimeError(
                f"Failed to load Cross-Encoder model '{model_name}': {e}"
            ) from e
    
    def rerank(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        trace: Optional[Any] = None,
        **kwargs: Any,
    ) -> List[Dict[str, Any]]:
        """Rerank candidates using Cross-Encoder scoring.
        
        Args:
            query: The user query string.
            candidates: List of candidate records to rerank. Each must contain
                either 'text' or 'content' field for scoring.
            trace: Optional TraceContext for observability (Stage F integration).
            **kwargs: Additional parameters (top_k to limit output, etc.).
        
        Returns:
            Reranked list of candidates ordered by relevance score (descending).
            Each candidate includes a 'rerank_score' field with the model's score.
        
        Raises:
            ValueError: If query or candidates are invalid.
            CrossEncoderRerankError: If scoring fails or times out.
        """
        # Validate inputs
        self.validate_query(query)
        self.validate_candidates(candidates)
        
        # Extract top_k parameter
        top_k = kwargs.get("top_k", len(candidates))
        if not isinstance(top_k, int) or top_k < 1:
            raise ValueError(f"top_k must be a positive integer, got {top_k}")
        
        try:
            # Prepare (query, passage) pairs for scoring
            pairs = self._prepare_pairs(query, candidates)
            
            # Score pairs using the model
            scores = self._score_pairs(pairs, trace=trace)
            
            # Attach scores to candidates and sort
            reranked = self._attach_scores_and_sort(candidates, scores, top_k)
            
            if trace:
                self._log_trace(trace, query, len(candidates), len(reranked))
            
            return reranked
            
        except Exception as e:
            logger.error(f"Cross-Encoder reranking failed: {e}", exc_info=True)
            # Signal failure for upstream fallback logic
            raise CrossEncoderRerankError(
                f"Cross-Encoder reranking failed: {e}"
            ) from e
    
    def _prepare_pairs(
        self,
        query: str,
        candidates: List[Dict[str, Any]]
    ) -> List[tuple[str, str]]:
        """Prepare (query, passage) pairs for scoring.
        
        Args:
            query: The user query.
            candidates: List of candidate records.
        
        Returns:
            List of (query, passage_text) tuples.
        """
        pairs = []
        for candidate in candidates:
            # Extract text from candidate (support both 'text' and 'content' keys)
            text = candidate.get("text") or candidate.get("content", "")
            if not isinstance(text, str):
                text = str(text)
            pairs.append((query, text))
        return pairs
    
    def _score_pairs(
        self,
        pairs: List[tuple[str, str]],
        trace: Optional[Any] = None
    ) -> List[float]:
        """Score (query, passage) pairs using the Cross-Encoder model.
        
        Args:
            pairs: List of (query, passage) tuples.
            trace: Optional TraceContext for observability.
        
        Returns:
            List of relevance scores (one per pair).
        
        Raises:
            CrossEncoderRerankError: If scoring fails or times out.
        """
        try:
            # Use model.predict() to score all pairs in batch
            executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="cross-encoder")
            try:
                future = executor.submit(self.model.predict, pairs)
                try:
                    scores = future.result(timeout=self.timeout)
                except FutureTimeoutError as exc:
                    future.cancel()
                    raise CrossEncoderRerankError(
                        f"Cross-Encoder scoring exceeded {self.timeout:.1f}s timeout"
                    ) from exc
            finally:
                executor.shutdown(wait=False, cancel_futures=True)
            
            # Convert numpy array to list if needed
            if hasattr(scores, 'tolist'):
                scores = scores.tolist()
            
            return scores
            
        except Exception as e:
            raise CrossEncoderRerankError(
                f"Failed to score pairs with Cross-Encoder: {e}"
            ) from e

    def warmup(self) -> None:
        """Materialize model weights and kernels before serving requests."""
        self._score_pairs([("warmup", "warmup")])

    @staticmethod
    def _resolve_device(configured: str) -> str:
        if configured != "auto":
            return configured
        try:
            import torch

            return "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            return "cpu"

    @staticmethod
    def _resolve_cache_dir(configured: str | None) -> Path | None:
        if not configured:
            return None
        if configured == "user":
            local_app_data = os.environ.get("LOCALAPPDATA")
            if local_app_data:
                return (Path(local_app_data) / "ModularRAG" / "models" / "huggingface").resolve()
            return (Path.home() / ".modular-rag" / "models" / "huggingface").resolve()
        return resolve_path(str(configured))
    
    def _attach_scores_and_sort(
        self,
        candidates: List[Dict[str, Any]],
        scores: List[float],
        top_k: int
    ) -> List[Dict[str, Any]]:
        """Attach scores to candidates and sort by relevance.
        
        Args:
            candidates: Original candidate list.
            scores: Relevance scores from the model.
            top_k: Number of top candidates to return.
        
        Returns:
            Sorted list of top_k candidates with 'rerank_score' field added.
        """
        # Attach scores to candidates
        scored_candidates = []
        for candidate, score in zip(candidates, scores):
            # Create a copy to avoid modifying original
            candidate_copy = candidate.copy()
            candidate_copy["rerank_score"] = float(score)
            scored_candidates.append(candidate_copy)
        
        # Sort by score (descending) and take top_k
        sorted_candidates = sorted(
            scored_candidates,
            key=lambda x: x["rerank_score"],
            reverse=True
        )
        
        return sorted_candidates[:top_k]
    
    def _log_trace(
        self,
        trace: Any,
        query: str,
        input_count: int,
        output_count: int
    ) -> None:
        """Log reranking operation to trace context.
        
        Args:
            trace: TraceContext instance.
            query: The query string.
            input_count: Number of input candidates.
            output_count: Number of output candidates.
        """
        # Placeholder for Stage F integration
        # Future: trace.log_rerank_step(...)
        logger.debug(
            f"Cross-Encoder rerank: query='{query[:50]}...', "
            f"input={input_count}, output={output_count}"
        )
