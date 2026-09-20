"""Pha 3 — MetaReviewAgent: tổng hợp mẫu hình phản biện lặp lại (từ review VÀ
từ lý do phân xử các trận đấu), phát phản hồi cải thiện cho các agent khác
(vòng lặp cải thiện liên tục), cập nhật research overview cho GenerationAgent,
theo dõi an toàn của hướng nghiên cứu, và khi kết thúc thì sinh báo cáo tổng
quan nghiên cứu (research overview).
"""
from __future__ import annotations

import logging
from typing import Dict, List

from agents.base_agent import BaseAgent
from models.hypothesis import Hypothesis

logger = logging.getLogger("meta_review_agent")

SYSTEM_PROMPT_FEEDBACK = """Bạn là MetaReviewAgent trong hệ thống multi-agent
hỗ trợ nghiên cứu khoa học. Nhiệm vụ: đọc các nhận xét phản biện (review
comments) và lý do phân xử các trận đấu trong tournament tích luỹ qua vòng lặp,
tìm ra CÁC MẪU HÌNH lặp lại (ví dụ: agent Generation hay đề xuất giả thuyết
thiếu tính khả thi, Reflection hay bỏ sót việc xét yếu tố chi phí thí nghiệm...),
rồi đề xuất phản hồi CỤ THỂ, NGẮN GỌN để cải thiện từng agent ở vòng lặp tiếp
theo. Đồng thời tóm tắt các hướng nghiên cứu đã/chưa được khám phá, và nêu mọi
lo ngại về an toàn/đạo đức của các hướng nghiên cứu hiện tại. Trả lời bằng JSON."""

JSON_SCHEMA_HINT_FEEDBACK = """Schema JSON trả về:
{
  "patterns": ["<mẫu hình 1>", "<mẫu hình 2>", ...],
  "feedback": {
    "generation_agent": "<phản hồi cụ thể, có thể để trống>",
    "proximity_agent": "<phản hồi cụ thể, có thể để trống>",
    "reflection_agent": "<phản hồi cụ thể, có thể để trống>",
    "ranking_agent": "<phản hồi cụ thể, có thể để trống>",
    "evolution_agent": "<phản hồi cụ thể, có thể để trống>"
  },
  "research_overview": "<5-10 dòng: các hướng nghiên cứu ĐÃ được khám phá và các hướng CHƯA được khám phá nên thử ở vòng sau>",
  "safety_concerns": ["<lo ngại an toàn/đạo đức về hướng nghiên cứu hiện tại; để mảng rỗng nếu không có>"]
}"""

SYSTEM_PROMPT_REPORT = """Bạn là MetaReviewAgent, nhiệm vụ cuối cùng: viết
một bản "Research Overview" súc tích, chuyên nghiệp, tổng hợp các giả thuyết
tốt nhất được hệ thống multi-agent tạo ra, kèm lý do và hướng kiểm chứng.

Báo cáo PHẢI có cấu trúc Markdown sau (giữ nguyên các tiêu đề):
  1. "## Tổng quan nghiên cứu" — bối cảnh + câu hỏi nghiên cứu.
  2. "## Tổng hợp bài báo nền tảng" — tóm tắt những bài báo thu thập làm
     grounding: mỗi bài 1-2 câu nêu ý chính, sau đó 1 đoạn tổng hợp xu hướng/
     lỗ hổng chung của literature. KHÔNG bịa nội dung ngoài abstract đã cho.
  3. "## Các giả thuyết nổi bật" — trình bày top giả thuyết kèm cơ chế, điểm
     mạnh/yếu và hướng kiểm chứng.
  4. "## Kết luận & hướng tiếp theo" — gợi ý nghiên cứu kế tiếp.
  5. "## Cảnh báo an toàn" — CHỈ thêm mục này nếu đầu vào có cảnh báo an toàn.

Viết bằng tiếng Việt, dùng Markdown."""


class MetaReviewAgent(BaseAgent):
    name = "meta_review_agent"

    def _collect_comments(self, limit: int = 40) -> str:
        """
        Thu thập các nhận xét phản biện gần đây nhất của tất cả các giả thuyết.
        Với đầu vào là limit - nó là số lượng nhận xét phản biện gần đây nhất muốn thu thập.
        """
        comments = []
        for h in self.memory.hypotheses.values():
            for r in h.reviews:
                comments.append(f"[{h.id}][{r.review_type}] {r.comments}")
        return "\n".join(comments[-limit:]) if comments else "(chưa có review nào)"

    def _collect_debates(self, limit: int = 20) -> str:
        """Lý do phân xử các trận gần nhất — nguồn thứ 2 để tìm mẫu hình (Methods, tr. 35-36)."""
        lines = [
            f"[{m.hypothesis_a_id} vs {m.hypothesis_b_id} -> thắng: {m.winner_id}] {m.rationale}"
            for m in self.memory.match_history[-limit:]
        ]
        return "\n".join(lines) if lines else "(chưa có trận đấu nào)"

    def _collect_papers(self, limit: int = 20) -> str:
        """Gom block danh sách paper (làm grounding) để chèn vào prompt báo cáo.

        Sắp xếp theo citation desc (giống retriever), giới hạn `limit` bài để
        vừa tiết kiệm token vừa đủ bối cảnh cho LLM tổng hợp. Mỗi paper kèm
        title, năm, citation và abstract (cắt tối đa ~400 ký tự để tránh dài).
        Trả "(chưa tra cứu paper nào)" nếu pool rỗng.
        """
        papers = self.memory.get_papers()  # đã dedup, có thể chưa sort
        papers.sort(key=lambda p: p.citations, reverse=True)
        if not papers:
            return "(chưa tra cứu paper nào — không có phần tổng hợp bài báo)"
        lines = []
        for p in papers[:limit]:
            abstract = (p.abstract or "").strip()
            if len(abstract) > 400:
                abstract = abstract[:400].rstrip() + "…"
            lines.append(
                f"- [{p.id}] (năm {p.year or '?'}, {p.citations} trích dẫn) {p.title}\n"
                f"  Tóm tắt: {abstract or '(không có abstract)'}"
            )
        return f"Tổng cộng {len(papers)} bài báo; sau đây là {min(len(papers), limit)} bài nhiều trích dẫn nhất:\n" + "\n".join(lines)

    async def run_feedback(self) -> Dict[str, str]:
        """
        Chạy cuối mỗi iteration: thu thập nhận xét phản biện + lý do phân xử các trận,
        đưa ra phản hồi cải thiện cho các agent khác, thêm các mẫu hình phản biện vào
        meta_review_notes, cập nhật research overview và ghi nhận cảnh báo an toàn.
        """
        user = (
            f"Mục tiêu nghiên cứu: {self.memory.research_goal}\n\n"
            f"Các nhận xét phản biện tích luỹ gần đây:\n{self._collect_comments()}\n\n"
            f"Lý do phân xử các trận đấu gần đây:\n{self._collect_debates()}\n\n"
            f"{JSON_SCHEMA_HINT_FEEDBACK}"
        )
        try:
            data = await self.llm.complete_json(SYSTEM_PROMPT_FEEDBACK, user)
        except Exception as e:
            # Thiếu feedback 1 vòng không làm hỏng kết quả -> bỏ qua thay vì dừng cả run.
            logger.warning("Meta-review: sinh feedback lỗi, bỏ qua vòng này: %s", e)
            return {}

        for pattern in data.get("patterns") or []:
            self.memory.add_meta_note(pattern)

        overview = data.get("research_overview")
        if isinstance(overview, str) and overview.strip():
            self.memory.research_overview = overview.strip()

        # Theo dõi an toàn liên tục (Supp. Note 7): ghi lại và cảnh báo người dùng qua log.
        for concern in data.get("safety_concerns") or []:
            if isinstance(concern, str) and concern.strip():
                self.memory.add_safety_alert(f"[meta-review] {concern.strip()}")
                logger.warning("Meta-review cảnh báo an toàn: %s", concern.strip())

        feedback = data.get("feedback") or {}
        if not isinstance(feedback, dict):
            feedback = {}
        for agent_name, note in feedback.items():
            if note:
                self.memory.add_agent_feedback(agent_name, note)
        return feedback

    async def run_final_report(self, top_k: int = 5) -> str:
        """
        Sinh báo cáo tổng quan nghiên cứu (research overview) dựa trên top-k giả thuyết tốt nhất.
        Chỉ lấy giả thuyết đã có full review và đã đấu ít nhất 1 trận.
        """
        top: List[Hypothesis] = self.memory.get_top_k(top_k, evaluated_only=True)
        lines = []
        for h in top:
            refs = h.reviews_of("full")[-1].references[:3] if h.reviews_of("full") else []
            lines.append(
                f"### {h.content}\n"
                f"- Elo: {h.elo_rating:.0f} ({h.matches_played} trận) | Correctness: {h.average_score('correctness'):.1f} "
                f"| Novelty: {h.average_score('novelty'):.1f} | Feasibility: {h.average_score('feasibility'):.1f}\n"
                f"- Cơ chế: {h.rationale}\n"
                f"- Thí nghiệm đề xuất: {h.suggested_experiment or '(chưa có)'}\n"
                + (f"- Tài liệu liên quan (từ full review): {'; '.join(refs)}\n" if refs else "")
            )
        top_summary = "\n".join(lines) if lines else "(không có giả thuyết nào đã được đánh giá đầy đủ)"
        safety_block = (
            "\n\nCảnh báo an toàn ghi nhận trong quá trình chạy:\n"
            + "\n".join(f"- {a}" for a in self.memory.safety_alerts[-10:])
            if self.memory.safety_alerts else ""
        )

        user = (
            f"Mục tiêu nghiên cứu: {self.memory.research_goal}\n\n"
            f"Bài báo thu thập làm grounding (dùng cho mục 'Tổng hợp bài báo nền tảng'):\n"
            f"{self._collect_papers()}\n\n"
            f"Top {top_k} giả thuyết sau {self.memory.iteration} vòng lặp:\n{top_summary}\n\n"
            f"Các mẫu hình phản biện quan sát được qua các vòng:\n"
            + "\n".join(f"- {n}" for n in self.memory.meta_review_notes[-10:])
            + safety_block
            + "\n\nHãy viết Research Overview hoàn chỉnh theo đúng cấu trúc tiêu đề đã nêu."
        )
        try:
            report = await self.llm.complete(SYSTEM_PROMPT_REPORT, user, max_tokens=3000)
        except Exception as e:
            # Không để mất toàn bộ kết quả chỉ vì lời gọi cuối cùng lỗi.
            logger.warning("Meta-review: viết báo cáo lỗi, dùng bản tổng hợp tự động: %s", e)
            report = ""
        if not report.strip():
            report = (
                "# Research Overview (bản tổng hợp tự động — LLM không viết được báo cáo)\n\n"
                f"Mục tiêu nghiên cứu: {self.memory.research_goal}\n\n{top_summary}{safety_block}\n"
            )
        self.memory.research_overview = report
        return report

    async def run(self, mode: str = "feedback", **kwargs):
        if mode == "feedback":
            return await self.run_feedback()
        elif mode == "final_report":
            return await self.run_final_report(**kwargs)
        raise ValueError(f"mode không hợp lệ: {mode}")
