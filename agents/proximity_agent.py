"""Pha 1 — ProximityAgent: đo độ tương đồng giữa các giả thuyết, dựng đồ thị
proximity, gắn cờ trùng lặp để các pha sau (Ranking, Evolution) không lãng
phí tài nguyên so sánh/cải tiến các giả thuyết gần như giống hệt nhau.

Cách tiếp cận: encode nội dung (content + rationale) mỗi giả thuyết thành
vector bằng sentence-transformers (local), tính cosine similarity cho mọi
cặp — thay vì gọi LLM từng cặp như trước. Nhanh, ít chi phí, ổn định hơn.
"""
from __future__ import annotations

import asyncio
import itertools
from typing import List, Optional, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer

from agents.base_agent import BaseAgent
from config import EmbeddingConfig
from models.hypothesis import Hypothesis, HypothesisStatus


class ProximityAgent(BaseAgent):
    name = "proximity_agent"

    def __init__(self, llm, memory, embedding: Optional[EmbeddingConfig] = None):
        # embedding ở đây là EmbeddingConfig (không phải LLMClient). Gọi
        # super().__init__ để set llm/memory; self.embedding sẽ lưu config embedding.
        super().__init__(llm, memory)
        self._emb_config: EmbeddingConfig = embedding or EmbeddingConfig()
        # Lazy-load: chỉ nạp SentenceTransformer khi run() lần đầu, tránh load
        # torch/sentence-transformers ngay khi merely import module này.
        self._model: Optional[SentenceTransformer] = None
        # Cache vector theo hypothesis id: hypothesis cũ xuất hiện lại ở iteration
        # sau (content không đổi) thì tái dùng vector, chỉ encode hypothesis mới.
        self._emb_cache: dict[str, np.ndarray] = {}

    @property
    def model(self) -> SentenceTransformer:
        """Nạp model sentence-transformer 1 lần (lazy), tái dùng cho mọi lần gọi."""
        if self._model is None:
            self._model = SentenceTransformer(
                self._emb_config.model_name,
                device=self._emb_config.device,
            )
        return self._model

    def _encode_hypotheses(self, active: List[Hypothesis]) -> np.ndarray:
        """Encode content+rationale của các giả thuyết active thành ma trận (n, d),
        đã L2-normalize. Có cache theo id để không re-encode hypothesis đã tính.
        """
        n = len(active)
        if n == 0:
            return np.zeros((0, 0), dtype=np.float32)

        # Tách hypothesis chưa có trong cache -> cần encode mới.
        texts_to_encode: List[str] = []
        ids_to_encode: List[str] = []
        for h in active:
            if h.id not in self._emb_cache:
                texts_to_encode.append(f"{h.content}\n{h.rationale}")
                ids_to_encode.append(h.id)

        # Encode batch các hypothesis mới; trả về mảng (m, d) đã normalize L2.
        if texts_to_encode:
            new_vecs = self.model.encode(
                texts_to_encode,
                batch_size=self._emb_config.batch_size,
                normalize_embeddings=True,
                convert_to_numpy=True,
            )
            for hid, vec in zip(ids_to_encode, new_vecs):
                self._emb_cache[hid] = vec.astype(np.float32)

        # Lắp ma trận (n, d) theo thứ tự `active`, lấy vector từ cache.
        dim = next(iter(self._emb_cache.values())).shape[0]
        emb = np.zeros((n, dim), dtype=np.float32)
        for idx, h in enumerate(active):
            emb[idx] = self._emb_cache[h.id]
        return emb

    async def run(self, duplicate_threshold: float = 0.85) -> List[Tuple[str, str, float]]:
        active = self.memory.get_active_hypotheses()
        pairs = list(itertools.combinations(active, 2))     # chỉnh hợp C(n,2)
        if not pairs:
            return []

        # Encode (đồng bộ) bọc trong to_thread để không block event loop orchestrator.
        # Tính luôn ma trận similarity trong cùng thread: emb @ emb.T = cosine (đã normalize).
        def _compute():
            emb = self._encode_hypotheses(active)
            return emb @ emb.T

        sim_matrix = await asyncio.to_thread(_compute)

        results: List[Tuple[str, str, float]] = []
        n = len(active)
        for i, j in itertools.combinations(range(n), 2):
            ha, hb = active[i], active[j]
            # Cosine ∈ [-1, 1]; với text embedding thường dương. Clip về [0, 1] cho
            # an toàn với duplicate_threshold (chỉ quan tâm tương đồng dương mạnh).
            sim = float(max(0.0, sim_matrix[i, j]))
            results.append((ha.id, hb.id, sim))
            self.memory.set_proximity(ha.id, hb.id, sim)
            if sim >= duplicate_threshold:
                # Giữ lại giả thuyết có elo cao hơn (hoặc bằng), đánh dấu bản còn lại
                # là trùng lặp để loại khỏi tournament/evolution.
                loser = hb if ha.elo_rating >= hb.elo_rating else ha
                if loser.status == HypothesisStatus.ACTIVE:
                    self.memory.mark_status(loser.id, HypothesisStatus.DUPLICATE)

        return results
