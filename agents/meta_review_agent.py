"""Pha 3 — MetaReviewAgent: tổng hợp mẫu hình phản biện lặp lại, phát phản
hồi cải thiện cho các agent khác (vòng lặp cải thiện liên tục), và khi kết
thúc thì sinh báo cáo tổng quan nghiên cứu (research overview).
"""
from __future__ import annotations

from typing import Dict, List

from agents.base_agent import BaseAgent
from models.hypothesis import Hypothesis

SYSTEM_PROMPT_FEEDBACK = """Bạn là MetaReviewAgent trong hệ thống multi-agent
hỗ trợ nghiên cứu khoa học. Nhiệm vụ: đọc các nhận xét phản biện (review
comments) tích luỹ qua vòng lặp, tìm ra CÁC MẪU HÌNH lặp lại (ví dụ: agent
Generation hay đề xuất giả thuyết thiếu tính khả thi, Reflection hay bỏ sót
việc xét yếu tố chi phí thí nghiệm...), rồi đề xuất phản hồi CỤ THỂ, NGẮN
GỌN để cải thiện từng agent ở vòng lặp tiếp theo. Trả lời bằng JSON."""

JSON_SCHEMA_HINT_FEEDBACK = """Schema JSON trả về:
{
  "patterns": ["<mẫu hình 1>", "<mẫu hình 2>", ...],
  "feedback": {
    "generation_agent": "<phản hồi cụ thể, có thể để trống>",
    "proximity_agent": "<phản hồi cụ thể, có thể để trống>",
    "reflection_agent": "<phản hồi cụ thể, có thể để trống>",
    "ranking_agent": "<phản hồi cụ thể, có thể để trống>",
    "evolution_agent": "<phản hồi cụ thể, có thể để trống>"
  }
}"""

SYSTEM_PROMPT_REPORT = """Bạn là MetaReviewAgent, nhiệm vụ cuối cùng: viết
một bản "Research Overview" súc tích, chuyên nghiệp, tổng hợp các giả thuyết
tốt nhất được hệ thống multi-agent tạo ra, kèm lý do và hướng kiểm chứng.
Viết bằng tiếng Việt, dùng Markdown."""


class MetaReviewAgent(BaseAgent):
    name = "meta_review_agent"

    def _collect_comments(self, limit: int = 40) -> str:
        comments = []
        for h in self.memory.hypotheses.values():
            for r in h.reviews:
                comments.append(f"[{h.id}][{r.review_type}] {r.comments}")
        return "\n".join(comments[-limit:]) if comments else "(chưa có review nào)"

    async def run_feedback(self) -> Dict[str, str]:
        """Chạy sau Pha 2 / cuối mỗi iteration: sinh phản hồi cho các agent."""
        user = (
            f"Mục tiêu nghiên cứu: {self.memory.research_goal}\n\n"
            f"Các nhận xét phản biện tích luỹ gần đây:\n{self._collect_comments()}\n\n"
            f"{JSON_SCHEMA_HINT_FEEDBACK}"
        )
        data = await self.llm.complete_json(SYSTEM_PROMPT_FEEDBACK, user)
        for pattern in data.get("patterns", []):
            self.memory.add_meta_note(pattern)
        feedback = data.get("feedback", {})
        for agent_name, note in feedback.items():
            if note:
                self.memory.add_agent_feedback(agent_name, note)
        return feedback

    async def run_final_report(self, top_k: int = 5) -> str:
        top: List[Hypothesis] = self.memory.get_top_k(top_k)
        lines = []
        for h in top:
            lines.append(
                f"### {h.content}\n"
                f"- Elo: {h.elo_rating:.0f} | Correctness: {h.average_score('correctness'):.1f} "
                f"| Novelty: {h.average_score('novelty'):.1f} | Feasibility: {h.average_score('feasibility'):.1f}\n"
                f"- Cơ chế: {h.rationale}\n"
                f"- Thí nghiệm đề xuất: {h.suggested_experiment or '(chưa có)'}\n"
            )
        top_summary = "\n".join(lines) if lines else "(không có giả thuyết nào)"

        user = (
            f"Mục tiêu nghiên cứu: {self.memory.research_goal}\n\n"
            f"Top {top_k} giả thuyết sau {self.memory.iteration} vòng lặp:\n{top_summary}\n\n"
            f"Các mẫu hình phản biện quan sát được qua các vòng:\n"
            + "\n".join(f"- {n}" for n in self.memory.meta_review_notes[-10:])
            + "\n\nHãy viết Research Overview hoàn chỉnh."
        )
        report = await self.llm.complete(SYSTEM_PROMPT_REPORT, user, max_tokens=3000)
        return report

    async def run(self, mode: str = "feedback", **kwargs):
        if mode == "feedback":
            return await self.run_feedback()
        elif mode == "final_report":
            return await self.run_final_report(**kwargs)
        raise ValueError(f"mode không hợp lệ: {mode}")
