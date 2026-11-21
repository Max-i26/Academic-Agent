"""BGE embedding provider implementation."""

import logging
import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from academic_agent.embeddings.protocols import EmbeddingProvider
from academic_agent.embeddings.tokenizer import HuggingFaceTokenCounter

logger = logging.getLogger(__name__)

DEFAULT_BGE_MODEL = "BAAI/bge-small-en-v1.5"
DEFAULT_QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "


class BGEProvider:
    """Embedding provider for BAAI/bge-small-en-v1.5.

    A 384-dimensional asymmetric embedding model with a 512-token sequence limit.
    Requires an asymmetric query prefix for queries, but plain text for documents.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_BGE_MODEL,
        device: str = "auto",
        query_instruction: str = DEFAULT_QUERY_INSTRUCTION,
    ):
        self._name = model_name
        self._dim = 384
        self._max_seq_length = 512
        self._query_instruction = query_instruction
        self._device = self._resolve_device(device)

        logger.info("Initializing BGEProvider on device: %s", self._device)
        try:
            self._model = SentenceTransformer(self._name, device=self._device)
        except Exception as e:
            if self._device == "cuda":
                logger.warning("Failed to initialize BGE on CUDA (%s), falling back to CPU", e)
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

    @property
    def query_instruction(self) -> str:
        return self._query_instruction

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
            logger.warning("CUDA OOM encountered during BGE encoding. Falling back to CPU: %s", oom)
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
        """Encode search queries by prepending the asymmetric query instruction."""
        formatted_queries = [self._query_instruction + q for q in queries]
        return self._encode_with_fallback(formatted_queries, batch_size=batch_size, normalize=normalize)

    def encode_documents(
        self,
        documents: list[str],
        batch_size: int = 32,
        normalize: bool = True,
    ) -> np.ndarray:
        """Encode passage/chunk documents without query instruction."""
        return self._encode_with_fallback(documents, batch_size=batch_size, normalize=normalize)
