"""MiniLM embedding provider implementation."""

import logging
import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from academic_agent.embeddings.protocols import EmbeddingProvider
from academic_agent.embeddings.tokenizer import HuggingFaceTokenCounter

logger = logging.getLogger(__name__)

DEFAULT_MINILM_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


class MiniLMProvider:
    """Embedding provider for all-MiniLM-L6-v2.

    A fast, lightweight 384-dimensional symmetric embedding model.
    Encodes queries and passages symmetrically without prefixes.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MINILM_MODEL,
        device: str = "auto",
    ):
        self._name = model_name
        self._dim = 384
        self._max_seq_length = 256
        self._device = self._resolve_device(device)

        logger.info("Initializing MiniLMProvider on device: %s", self._device)
        try:
            self._model = SentenceTransformer(self._name, device=self._device)
        except Exception as e:
            if self._device == "cuda":
                logger.warning("Failed to initialize MiniLM on CUDA (%s), falling back to CPU", e)
                self._device = "cpu"
                self._model = SentenceTransformer(self._name, device="cpu")
            else:
                raise

        self._token_counter = HuggingFaceTokenCounter(self._name)

    @staticmethod
    def _resolve_device(device_str: str) -> str:
        if device_str == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        if device_str == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA requested but not available. Falling back to CPU.")
            return "cpu"
        return device_str

    @property
    def name(self) -> str:
        return self._name

    @property
    def dimension(self) -> int:
        return self._dim

    @property
    def max_sequence_length(self) -> int:
        return self._max_seq_length

    @property
    def device(self) -> str:
        return self._device

    def count_tokens(self, text: str) -> int:
        return self._token_counter.count_tokens(text)

    def to(self, device: str) -> None:
        target = self._resolve_device(device)
        self._model.to(target)
        self._device = target

    def _encode_with_fallback(
        self,
        texts: list[str],
        batch_size: int,
        normalize: bool,
    ) -> np.ndarray:
        try:
            embeddings = self._model.encode(
                texts,
                batch_size=batch_size,
                normalize_embeddings=normalize,
                show_progress_bar=False,
                convert_to_numpy=True,
            )
            return embeddings.astype(np.float32)
        except torch.cuda.OutOfMemoryError as oom:
            logger.warning("CUDA OOM encountered during MiniLM encoding. Falling back to CPU: %s", oom)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            self.to("cpu")
            embeddings = self._model.encode(
                texts,
                batch_size=batch_size,
                normalize_embeddings=normalize,
                show_progress_bar=False,
                convert_to_numpy=True,
            )
            return embeddings.astype(np.float32)

    def encode_queries(
        self,
        queries: list[str],
        batch_size: int = 32,
        normalize: bool = True,
    ) -> np.ndarray:
        """Encode search queries symmetrically without instructions."""
        return self._encode_with_fallback(queries, batch_size=batch_size, normalize=normalize)

    def encode_documents(
        self,
        documents: list[str],
        batch_size: int = 32,
        normalize: bool = True,
    ) -> np.ndarray:
        """Encode passage/chunk documents."""
        return self._encode_with_fallback(documents, batch_size=batch_size, normalize=normalize)
