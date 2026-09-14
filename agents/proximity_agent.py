"""Pha 1 — ProximityAgent: đo độ tương đồng giữa các giả thuyết, dựng đồ thị
proximity, gắn cờ trùng lặp để các pha sau (Ranking, Evolution) không lãng
phí tài nguyên so sánh/cải tiến các giả thuyết gần như giống hệt nhau.

Cách tiếp cận: embedding + LLM xác nhận
    1. Encode nội dung (content + rationale) mỗi giả thuyết thành vector bằng
       sentence-transformers (chạy local, model đa ngôn ngữ vì giả thuyết viết
       tiếng Việt), tính cosine cho MỌI cặp mới -> dựng proximity graph. Không tốn
       lượt gọi API; vector được cache theo id, cặp đã tính không tính lại.
    2. Embedding gần như không phân biệt được ý NGƯỢC NGHĨA ("tăng cường" vs "ức chế"
       cùng một cơ chế cho vector gần như giống nhau), nên KHÔNG đánh dấu trùng chỉ
       dựa trên cosine. Cặp có cosine >= duplicate_threshold chỉ là "nghi trùng" và
       được hỏi LLM: có thật là cùng một ý tưởng không? (tối đa max_llm_checks cặp/lượt,
       cặp chưa kịp hỏi sẽ được hỏi ở lượt sau).
    3. LLM xác nhận trùng -> đánh dấu DUPLICATE bản có Elo thấp hơn.
"""
from __future__ import annotations

import asyncio
import itertools
import logging
from typing import Callable, Dict, List, Optional, Set, Tuple

from agents.base_agent import BaseAgent, as_bool
from config import EmbeddingConfig
from models.hypothesis import Hypothesis, HypothesisStatus

logger = logging.getLogger("proximity_agent")

# Hàm encode: danh sách text -> danh sách vector đã chuẩn hoá L2.
Encoder = Callable[[List[str]], List[List[float]]]

SYSTEM_PROMPT = """Bạn là ProximityAgent trong hệ thống multi-agent hỗ trợ
nghiên cứu khoa học. Nhiệm vụ: xác định hai giả thuyết có phải là CÙNG MỘT Ý
TƯỞNG (trùng lặp) hay không, xét theo mục tiêu nghiên cứu. Chỉ coi là trùng khi
cùng cơ chế đề xuất, cùng chiều tác động (tăng/giảm, kích hoạt/ức chế) và cùng
đối tượng nghiên cứu — khác cách diễn đạt không quan trọng. Hai giả thuyết ngược
chiều tác động, khác cơ chế hoặc khác đối tượng thì KHÔNG trùng. Trả lời bằng JSON."""

JSON_SCHEMA_HINT = """Schema JSON trả về:
{
  "duplicate": true | false,
  "reason": "<giải thích ngắn 1 câu>"
}"""


class SentenceTransformerEncoder:
    """Encode text bằng sentence-transformers (chạy local). Nạp model lười (lazy) ở lần
    gọi đầu tiên để import module này không kéo theo torch."""

    def __init__(self, config: EmbeddingConfig):
        self.config = config
        self._model = None

    def __call__(self, texts: List[str]) -> List[List[float]]:
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as e:
                raise RuntimeError(
                    "ProximityAgent cần sentence-transformers + torch. Cài theo hướng dẫn "
                    "trong requirements.txt rồi chạy lại."
                ) from e
            self._model = SentenceTransformer(self.config.model_name, device=self.config.device)
        vectors = self._model.encode(
            texts,
            batch_size=self.config.batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return [v.tolist() for v in vectors]


def _cosine(u: List[float], v: List[float]) -> float:
    # Vector đã chuẩn hoá L2 -> cosine chính là tích vô hướng.
    return sum(x * y for x, y in zip(u, v))


class ProximityAgent(BaseAgent):
    name = "proximity_agent"

    def __init__(self, llm, memory, embedding: Optional[EmbeddingConfig] = None,
                 encoder: Optional[Encoder] = None):
        # embedding ở đây là EmbeddingConfig (không phải LLMClient). encoder cho phép
        # thay backend embedding (vd: test dùng encoder giả, không cần torch).
        super().__init__(llm, memory, embedding)
        self.encoder: Encoder = encoder or SentenceTransformerEncoder(embedding or EmbeddingConfig())
        # Cache vector theo hypothesis id: content không đổi nên chỉ encode giả thuyết mới.
        self._emb_cache: Dict[str, List[float]] = {}
        # Cặp nghi trùng đã được LLM kết luận (trùng hoặc không) -> không hỏi lại.
        self._checked: Set[frozenset] = set()

    async def ensure_ready(self) -> None:
        """Nạp model embedding trước vòng lặp: thiếu thư viện/model thì báo lỗi ngay từ
        đầu, thay vì sau khi đã tốn lượt gọi LLM ở bước Generation."""
        await asyncio.to_thread(self.encoder, ["warmup"])

    def _encode_missing(self, hypotheses: List[Hypothesis]) -> None:
        missing = [h for h in hypotheses if h.id not in self._emb_cache]
        if not missing:
            return
        vectors = self.encoder([f"{h.content}\n{h.rationale}" for h in missing])
        for h, vec in zip(missing, vectors):
            self._emb_cache[h.id] = vec

    async def _confirm_duplicate(self, a: Hypothesis, b: Hypothesis) -> bool:
        user = (
            f"Mục tiêu nghiên cứu: {self.memory.research_goal}\n\n"
            f"Giả thuyết A: {a.content}\nCơ chế A: {a.rationale}\n\n"
            f"Giả thuyết B: {b.content}\nCơ chế B: {b.rationale}\n\n"
            f"{JSON_SCHEMA_HINT}{self.feedback_block()}"
        )
        data = await self.llm.complete_json(SYSTEM_PROMPT, user)
        # Thiếu trường -> coi là không trùng (không loại nhầm giả thuyết).
        return as_bool(data.get("duplicate"), default=False)

    async def run(
        self, duplicate_threshold: float = 0.80, max_llm_checks: int = 10
    ) -> List[Tuple[str, str, float]]:
        """Trả về các cặp MỚI được tính cosine trong lượt này: (id_a, id_b, similarity)."""
        active = self.memory.get_active_hypotheses()
        if len(active) < 2:
            return []

        # 1) Encode giả thuyết mới (đồng bộ, chạy trong thread để không block event loop).
        await asyncio.to_thread(self._encode_missing, active)

        # 2) Cosine cho các cặp chưa có trong graph.
        new_pairs: List[Tuple[str, str, float]] = []
        for a, b in itertools.combinations(active, 2):
            if self.memory.has_proximity(a.id, b.id):
                continue
            # Cosine ∈ [-1, 1]; clip về [0, 1] cho khớp thang của proximity graph.
            sim = max(0.0, _cosine(self._emb_cache[a.id], self._emb_cache[b.id]))
            self.memory.set_proximity(a.id, b.id, sim)
            new_pairs.append((a.id, b.id, sim))

        # 3) Cặp nghi trùng (kể cả cặp cũ chưa kịp hỏi ở lượt trước), cao nhất trước.
        candidates: List[Tuple[Hypothesis, Hypothesis, float]] = []
        for a, b in itertools.combinations(active, 2):
            sim = self.memory.proximity_graph.get(a.id, {}).get(b.id, 0.0)
            if sim >= duplicate_threshold and frozenset((a.id, b.id)) not in self._checked:
                candidates.append((a, b, sim))
        candidates.sort(key=lambda c: c[2], reverse=True)
        if len(candidates) > max_llm_checks:
            logger.info("Proximity: %d cặp nghi trùng, lượt này chỉ hỏi LLM %d cặp cao nhất",
                        len(candidates), max_llm_checks)
            candidates = candidates[:max_llm_checks]

        verdicts = await asyncio.gather(
            *[self._confirm_duplicate(a, b) for a, b, _ in candidates], return_exceptions=True
        )
        n_confirmed = 0
        for (a, b, sim), verdict in zip(candidates, verdicts):
            if isinstance(verdict, Exception):
                # Không chắc -> không loại; cặp chưa vào _checked nên lượt sau hỏi lại.
                logger.warning("Proximity: xác nhận cặp %s-%s lỗi, thử lại lượt sau: %s", a.id, b.id, verdict)
                continue
            self._checked.add(frozenset((a.id, b.id)))
            # Chỉ xử lý khi cả 2 còn active: nếu 1 bên đã bị đánh dấu trùng ở cặp trước
            # thì không kéo thêm bên kia xuống theo.
            if (verdict
                    and a.status == HypothesisStatus.ACTIVE
                    and b.status == HypothesisStatus.ACTIVE):
                # Giữ lại giả thuyết có elo cao hơn (hoặc bằng), đánh dấu bản còn lại
                # là trùng lặp để loại khỏi tournament/evolution.
                loser = b if a.elo_rating >= b.elo_rating else a
                self.memory.mark_status(loser.id, HypothesisStatus.DUPLICATE)
                n_confirmed += 1

        if candidates:
            logger.info("Proximity: hỏi LLM %d cặp nghi trùng (cosine >= %.2f), xác nhận trùng %d cặp",
                        len(candidates), duplicate_threshold, n_confirmed)
        return new_pairs
