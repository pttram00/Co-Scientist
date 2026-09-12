"""Orchestrator: điều phối vòng lặp giữa 6 agent, dùng ContextMemory làm nguồn
sự thật duy nhất (single source of truth) cho toàn hệ thống.

Thiết kế thống nhất: chỉ còn MỘT Orchestrator nhận ContextMemory có sẵn (dù là
mới khởi tạo hay load từ state.json). Hai factory `for_new` / `for_resume` chỉ
khác nhau ở cách tạo memory — infra (llm + retriever + 6 agent) chung một đường
dẫn qua `_init_infra`.

Vì vậy chatbot có thể yêu cầu chạy lại THEO BƯỚC (mỗi bước = 1 agent) qua
`run_step(step, overwrite)`, thay vì chỉ chạy full iteration. `_clear_step`
quản lý "xoá output cũ của bước đó" khi user chọn overwrite; keep = append.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional

from agents.evolution_agent import EvolutionAgent
from agents.generation_agent import GenerationAgent
from agents.meta_review_agent import MetaReviewAgent
from agents.proximity_agent import ProximityAgent
from agents.ranking_agent import RankingAgent
from agents.reflection_agent import ReflectionAgent
from config import AppConfig
from llm.client import LLMClient
from memory.context_memory import ContextMemory
from models.hypothesis import HypothesisStatus
from retrieval.retriever import Retriever

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("orchestrator")


# Tên bước theo agent. Thứ tự này cũng là thứ tự chạy trong 1 iteration.
STEP_ORDER: List[str] = [
    "generation", "proximity", "reflection", "ranking", "evolution", "meta_review",
]


class Orchestrator:
    def __init__(self, memory: ContextMemory, config: Optional[AppConfig] = None):
        """Nhận memory có sẵn (tạo mới hoặc load). Infra chung đi qua _init_infra."""
        self.config = config or AppConfig()
        self.memory = memory
        self._init_infra()

    def _init_infra(self) -> None:
        """Khởi tạo llm client + retriever + 6 agent. Gọi chung cho new/resume."""
        self.llm = LLMClient(self.config.llm)
        self.retriever = Retriever(self.config.retriever)
        # 1 instance / agent, dùng chung memory + llm client trong suốt vòng đời.
        self.generation_agent = GenerationAgent(self.llm, self.memory, self.retriever, self.config.retriever)
        self.proximity_agent = ProximityAgent(self.llm, self.memory)
        self.reflection_agent = ReflectionAgent(self.llm, self.memory)
        self.ranking_agent = RankingAgent(self.llm, self.memory)
        self.evolution_agent = EvolutionAgent(self.llm, self.memory)
        self.meta_review_agent = MetaReviewAgent(self.llm, self.memory)
        # Map tên bước -> agent object, để run_step tra nhanh.
        self._agents: Dict = {
            "generation":      self.generation_agent,
            "proximity":       self.proximity_agent,
            "reflection":      self.reflection_agent,
            "ranking":         self.ranking_agent,
            "evolution":       self.evolution_agent,
            "meta_review":     self.meta_review_agent,
        }

    # -------------------------------------------------- factory
    @staticmethod
    def for_new(research_goal: str, constraints: str = "",
                config: Optional[AppConfig] = None) -> "Orchestrator":
        """Khởi tạo cho luồng "tạo hướng nghiên cứu mới"."""
        return Orchestrator(ContextMemory(research_goal=research_goal, constraints=constraints), config)

    @staticmethod
    def for_resume(state_path: str, config: Optional[AppConfig] = None) -> "Orchestrator":
        """Khởi tạo cho luồng "tiếp tục/nhận xét" — nạp state đã có."""
        return Orchestrator(ContextMemory.load(state_path), config)

    # -------------------------------------------------- params theo bước
    def _step_params(self, step: str) -> dict:
        """Tham số chạy agent của bước, lấy giá trị từ config (giống batch cũ)."""
        oc = self.config.orchestrator
        return {
            "generation":  {"n_hypotheses": oc.hypotheses_per_iteration},
            "proximity":   {"duplicate_threshold": oc.proximity_duplicate_threshold},
            "reflection":  {},
            "ranking":     {"n_matches": oc.matches_per_iteration},
            "evolution":   {"top_k": oc.top_k_for_evolution},
            "meta_review": {"mode": "feedback"},   # final_report có method riêng
        }[step]

    # -------------------------------------------------- clear output cũ
    def _clear_step(self, step: str) -> None:
        """Xoá output cũ của 1 bước (cho rerun overwrite). Mỗi agent ghi memory
        khác nhau nên xoá khác nhau. Bước giữ lại user_comment/agent_feedback của
        user để không mất nhận xét người dùng."""
        if step == "generation":
            # Xoá hypothesis do generation_agent sinh ở iteration hiện tại.
            cur = self.memory.iteration
            to_del = [
                hid for hid, h in self.memory.hypotheses.items()
                if h.source_agent == "generation_agent"
                # Không có cờ iteration trên hypothesis → xoá tất cả hypothesis gốc
                # do generation tạo (parent_ids rỗng). User comment theo hypothesis
                # cũng bị mất — đây là trade-off của overwrite generation.
                and not h.parent_ids
            ]
            for hid in to_del:
                del self.memory.hypotheses[hid]
        elif step == "proximity":
            # Xoá proximity graph + bỏ cờ DUPLICATE (để các hypothesis sống lại).
            self.memory.proximity_graph.clear()
            for h in self.memory.hypotheses.values():
                if h.status == HypothesisStatus.DUPLICATE:
                    h.status = HypothesisStatus.ACTIVE
        elif step == "reflection":
            # Xoá review do reflection thêm (initial/full/tournament), GIỮ user_comment.
            keep_types = {"user_comment"}
            for h in self.memory.hypotheses.values():
                h.reviews = [r for r in h.reviews if r.review_type in keep_types]
        elif step == "ranking":
            # Xoá match history + reset Elo về mặc định.
            self.memory.match_history.clear()
            for h in self.memory.hypotheses.values():
                h.elo_rating = 1200.0
                h.matches_played = 0
        elif step == "evolution":
            # Xoá hypothesis do evolution tạo (có parent_ids + source evolution).
            to_del = [
                hid for hid, h in self.memory.hypotheses.items()
                if h.source_agent == "evolution_agent" and h.parent_ids
            ]
            for hid in to_del:
                del self.memory.hypotheses[hid]
        elif step == "meta_review":
            # Xoá meta notes + agent feedback (giữ feedback của user? -> hiện giữ
            # hết vì user comment có thể nằm trong agent_feedback nếu user chê agent).
            self.memory.meta_review_notes.clear()
            # KHÔNG xoá agent_feedback để giữ nhận xét của user về agent.
        # else: bước không nhận diện -> không xoá.

    # -------------------------------------------------- chạy 1 bước
    async def run_step(self, step: str, overwrite: bool = True) -> str:
        """Chạy lại 1 agent (1 bước). overwrite=True xoá output cũ trước.
        Trả chuỗi log tiếng Việt cho chatbot hiển thị. Kết quả ghi vào memory."""
        if step not in self._agents:
            return f"Bước '{step}' không hợp lệ. Các bước: {', '.join(STEP_ORDER)}"

        if overwrite:
            self._clear_step(step)

        agent = self._agents[step]
        params = self._step_params(step)
        logger.info("=== Run step: %s (overwrite=%s) ===", step, overwrite)
        try:
            if step == "meta_review":
                # MetaReview có 2 mode: "feedback" (per-iteration) chạy qua run_step.
                # "final_report" chạy riêng qua run_final_report().
                result = await agent.run(**params)
                n = len([v for v in result.values() if v]) if isinstance(result, dict) else 0
                return f"✓ meta_review (feedback): sinh feedback cho {n} agent."
            result = await agent.run(**params)
            n = len(result) if hasattr(result, "__len__") else 0
            return f"✓ {step}: xong ({n} mục)."
        except Exception as e:
            return f"✗ {step} lỗi: {e}"

    # -------------------------------------------------- chạy full iteration
    async def run_full(self, n_iterations: Optional[int] = None,
                       state_path: str = "output/state.json",
                       report_path: str = "output/final_report.md") -> str:
        """Chạy n vòng lặp (mỗi vòng = 6 bước theo STEP_ORDER), tăng iteration,
        save sau mỗi vòng, cuối sinh final_report. Dùng cho luồng 'new'."""
        n = n_iterations if n_iterations is not None else self.config.orchestrator.n_iterations
        Path(state_path).parent.mkdir(parents=True, exist_ok=True)
        try:
            for i in range(n):
                self.memory.iteration = i + 1
                logger.info("=== Iteration %d/%d ===", i + 1, n)
                for step in STEP_ORDER:
                    await self.run_step(step, overwrite=False)  # full run: append, không xoá
                self.memory.save(state_path)

            logger.info("Sinh báo cáo tổng quan nghiên cứu cuối cùng...")
            report = await self.meta_review_agent.run(mode="final_report")
            Path(report_path).parent.mkdir(parents=True, exist_ok=True)
            Path(report_path).write_text(report, encoding="utf-8")
            logger.info("Đã lưu báo cáo tại %s", report_path)
            return report_path
        finally:
            await self.retriever.aclose()

    # -------------------------------------------------- tiện ích
    def save(self, state_path: str) -> None:
        """Wrapper để chatbot gọi 1 chỗ duy nhất khi cập nhật state."""
        self.memory.save(state_path)
