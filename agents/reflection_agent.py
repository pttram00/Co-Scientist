"""Pha 2 — ReflectionAgent: phản biện giả thuyết theo 2 bước (Methods, tr. 33):

1. Initial review (không dùng tool): chấm nhanh correctness / novelty / feasibility
   và đánh giá an toàn sơ bộ -> loại sớm giả thuyết hỏng (ARCHIVED) hoặc không an
   toàn (UNSAFE). Đồng thời đề xuất query để tra cứu ở bước sau.
2. Full review (có tra cứu tài liệu qua Retriever): chấm lại 3 tiêu chí dựa trên
   bài báo thật, tóm tắt phần "đã biết" trước khi đánh giá novelty, và mô phỏng cơ
   chế từng bước để tìm kịch bản thất bại.

Mỗi giả thuyết chỉ được review một lần: chỉ xử lý giả thuyết active CHƯA có full
review. Giả thuyết có full review mới được vào tournament (RankingAgent).
"""
from __future__ import annotations

import asyncio
import logging
from typing import List, Tuple

from agents.base_agent import BaseAgent, as_bool, truncate
from config import RetrieverConfig
from models.hypothesis import Hypothesis, HypothesisStatus, Review
from models.paper import Paper
from retrieval.retriever import Retriever

logger = logging.getLogger("reflection_agent")

INITIAL_SYSTEM_PROMPT = """Bạn là ReflectionAgent — nhà phản biện khoa học trong hệ
thống multi-agent hỗ trợ nghiên cứu. Đây là bước INITIAL REVIEW: đánh giá nhanh,
CHƯA tra cứu tài liệu. Nhiệm vụ:
- Chấm sơ bộ (0-10): correctness (đúng đắn/hợp lý), novelty (tính mới),
  feasibility (khả năng kiểm chứng).
- Quyết định "pass": chỉ đánh trượt khi giả thuyết có lỗi khoa học rõ ràng, lạc
  khỏi mục tiêu nghiên cứu, vi phạm ràng buộc, hoặc hiển nhiên đã được biết. Khi
  còn phân vân, cho qua để bước full review (có tra cứu tài liệu) quyết định.
- Đánh giá an toàn sơ bộ: "safe" = false nếu giả thuyết hoặc thí nghiệm đề xuất có
  thể tạo điều kiện cho nghiên cứu nguy hiểm, phi đạo đức hoặc gây hại (vd: tăng
  độc lực/khả năng lây của mầm bệnh, vũ khí sinh học/hoá học, thí nghiệm trên người
  không có đồng thuận).
- Đề xuất 1-2 query tìm bài báo (tiếng Anh, 3-8 từ) để kiểm tra novelty ở bước sau.
Trả lời bằng JSON."""

INITIAL_SCHEMA_HINT = """Schema JSON trả về:
{
  "correctness": <0-10>,
  "novelty": <0-10>,
  "feasibility": <0-10>,
  "pass": true | false,
  "safe": true | false,
  "safety_concerns": "<để trống nếu an toàn>",
  "search_queries": ["<query 1>", "<query 2>"],
  "comments": "<nhận xét ngắn; nêu lý do nếu đánh trượt>"
}"""

FULL_SYSTEM_PROMPT = """Bạn là ReflectionAgent — một nhà phản biện khoa học
nghiêm khắc nhưng công tâm. Đây là bước FULL REVIEW: bạn được cung cấp các bài
báo liên quan đã tra cứu. Đánh giá giả thuyết theo 3 tiêu chí (0-10):
- correctness: soi các giả định và lập luận nền tảng; chỉ ra mâu thuẫn với tài
  liệu nếu có.
- novelty: TRƯỚC TIÊN tóm tắt những khía cạnh của giả thuyết đã được biết trong
  tài liệu, sau đó đánh giá phần còn lại có thực sự mới không. Giả thuyết trùng
  với công trình đã công bố phải nhận điểm novelty thấp.
- feasibility: khả năng kiểm chứng bằng thực nghiệm/phân tích trong thực tế.
Hãy MÔ PHỎNG cơ chế đề xuất (hoặc thí nghiệm) từng bước để phát hiện kịch bản
thất bại trước khi chấm điểm. Không bịa nội dung bài báo. Trả lời bằng JSON."""

FULL_SCHEMA_HINT = """Schema JSON trả về:
{
  "known_aspects": "<những gì trong giả thuyết đã có trong tài liệu, dẫn tên bài nếu có>",
  "novelty_assessment": "<phần nào thực sự mới, phần nào không>",
  "simulation_notes": "<mô phỏng cơ chế từng bước, chỉ ra điểm yếu/kịch bản thất bại>",
  "correctness": <0-10>,
  "novelty": <0-10>,
  "feasibility": <0-10>,
  "comments": "<nhận xét tổng hợp, gợi ý cải thiện cụ thể>"
}"""


class ReflectionAgent(BaseAgent):
    name = "reflection_agent"

    def __init__(self, llm, memory, retriever: Retriever, config: RetrieverConfig):
        # Giống GenerationAgent: nhận thêm retriever để tra cứu tài liệu ở full review.
        super().__init__(llm, memory)
        self.retriever = retriever
        self.config = config

    async def run(self) -> List[Hypothesis]:
        """Review các giả thuyết active chưa có full review. Trả về những giả thuyết
        vừa có full review trong lượt này (đủ điều kiện vào tournament)."""
        pending = [h for h in self.memory.get_active_hypotheses() if not h.is_reviewed]
        if not pending:
            return []

        results = await asyncio.gather(
            *[self._review_pipeline(h) for h in pending], return_exceptions=True
        )
        reviewed: List[Hypothesis] = []
        for h, res in zip(pending, results):
            if isinstance(res, Exception):
                # Lỗi 1 giả thuyết không làm dừng cả pha; giả thuyết vẫn "chưa review"
                # nên sẽ được thử lại ở lượt Reflection sau.
                logger.warning("Reflection: review giả thuyết %s lỗi, thử lại lượt sau: %s", h.id, res)
                continue
            if res:
                reviewed.append(h)
        return reviewed

    # ------------------------------------------------------------ pipeline

    async def _review_pipeline(self, h: Hypothesis) -> bool:
        """Initial review (nếu chưa có) -> full review. True nếu đã có full review."""
        queries: List[str] = []
        if not h.reviews_of("initial"):
            review, data = await self._initial_review(h)
            h.reviews.append(review)

            # Thiếu trường "safe" -> coi là không an toàn (fail-closed).
            if not as_bool(data.get("safe"), default=False):
                concern = str(data.get("safety_concerns") or "không nêu lý do").strip()
                self.memory.mark_status(h.id, HypothesisStatus.UNSAFE)
                self.memory.add_safety_alert(f"[{h.id}] {concern}")
                logger.warning("Reflection: giả thuyết %s bị gắn cờ KHÔNG AN TOÀN: %s", h.id, concern)
                return False
            if not as_bool(data.get("pass"), default=True):
                self.memory.mark_status(h.id, HypothesisStatus.ARCHIVED)
                logger.info("Reflection: giả thuyết %s không qua initial review -> archived", h.id)
                return False
            queries = [q.strip() for q in (data.get("search_queries") or [])
                       if isinstance(q, str) and q.strip()]

        if not queries:
            # Không có query từ initial review (vd: lần trước full review lỗi, đang thử
            # lại) -> dùng phần đầu câu phát biểu giả thuyết làm query.
            queries = [" ".join(h.content.split()[:12])]
        h.reviews.append(await self._full_review(h, queries))
        return True

    async def _initial_review(self, h: Hypothesis) -> Tuple[Review, dict]:
        user = (
            f"{self._context_block()}\n\n{self._hypothesis_block(h)}\n\n"
            f"{INITIAL_SCHEMA_HINT}{self.feedback_block()}"
        )
        data = await self.llm.complete_json(INITIAL_SYSTEM_PROMPT, user)
        comments = str(data.get("comments", "")).strip()
        if data.get("safety_concerns"):
            comments = f"{comments}\nAn toàn: {data['safety_concerns']}".strip()
        review = Review(
            reviewer=self.name,
            review_type="initial",
            correctness=float(data["correctness"]),
            novelty=float(data["novelty"]),
            feasibility=float(data["feasibility"]),
            comments=comments,
        )
        return review, data

    async def _full_review(self, h: Hypothesis, queries: List[str]) -> Review:
        # Retriever không raise: nguồn lỗi thì bỏ, toàn bộ lỗi thì trả [] -> vẫn review
        # nhưng prompt ghi rõ là chưa kiểm chứng được bằng tài liệu.
        papers = await self.retriever.search(
            queries[: self.config.review_max_queries],
            k_per_source=self.config.review_k_per_source,
            pool_size=self.config.review_papers_per_review,
        )
        user = (
            f"{self._context_block()}\n\n{self._hypothesis_block(h)}\n\n"
            f"{self._literature_block(papers)}\n\n{FULL_SCHEMA_HINT}{self.feedback_block()}"
        )
        data = await self.llm.complete_json(FULL_SYSTEM_PROMPT, user)
        sections = [
            ("Đã biết", data.get("known_aspects")),
            ("Novelty", data.get("novelty_assessment")),
            ("Mô phỏng", data.get("simulation_notes")),
            ("Nhận xét", data.get("comments")),
        ]
        comments = "\n".join(f"{label}: {str(text).strip()}" for label, text in sections if text)
        return Review(
            reviewer=self.name,
            review_type="full",
            correctness=float(data["correctness"]),
            novelty=float(data["novelty"]),
            feasibility=float(data["feasibility"]),
            comments=comments,
            references=[f"{p.title} ({p.year or '?'})" for p in papers],
        )

    # ------------------------------------------------------------- helpers

    def _context_block(self) -> str:
        return (
            f"Mục tiêu nghiên cứu: {self.memory.research_goal}\n"
            f"Ràng buộc/bối cảnh: {self.memory.constraints or '(không có)'}"
        )

    @staticmethod
    def _hypothesis_block(h: Hypothesis) -> str:
        return (
            f"Giả thuyết: {h.content}\nCơ chế đề xuất: {h.rationale}\n"
            f"Thí nghiệm đề xuất: {h.suggested_experiment or '(chưa có)'}"
        )

    @staticmethod
    def _literature_block(papers: List[Paper]) -> str:
        if not papers:
            return (
                "Tài liệu tra cứu: (không tìm được bài báo nào — hãy đánh giá novelty "
                "thận trọng và nêu rõ là chưa kiểm chứng được bằng tài liệu)"
            )
        lines = []
        for p in papers:
            lines.append(f"- {p.title} (year={p.year or '?'}, citation={p.citations})")
            if p.abstract:
                lines.append(f"  {truncate(p.abstract, 700)}")
        return "Tài liệu tra cứu liên quan:\n" + "\n".join(lines)
