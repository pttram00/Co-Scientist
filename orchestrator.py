"""Orchestrator: điều phối vòng lặp 3 pha giữa các agent, dùng ContextMemory
làm nguồn sự thật duy nhất (single source of truth) cho toàn hệ thống.

Luồng chạy:
    0. SafetyAgent kiểm tra mục tiêu nghiên cứu -> không an toàn thì dừng.
    1..N. Mỗi iteration: Pha 1 (Generation + Proximity) -> Pha 2 (Reflection +
       Ranking) -> Pha 3 (Evolution + Meta-review). State được lưu sau MỖI pha.
    Pha kết thúc: review + xếp hạng các giả thuyết Evolution tạo ở vòng cuối.
    Báo cáo: MetaReviewAgent viết Research Overview từ giả thuyết đã đánh giá đầy đủ.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from agents.evolution_agent import EvolutionAgent
from agents.generation_agent import GenerationAgent
from agents.meta_review_agent import MetaReviewAgent
from agents.proximity_agent import ProximityAgent
from agents.ranking_agent import RankingAgent
from agents.reflection_agent import ReflectionAgent
from agents.safety_agent import SafetyAgent, UnsafeResearchGoalError
from config import AppConfig
from llm.client import LLMClient
from memory.context_memory import ContextMemory
from retrieval.retriever import Retriever

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("orchestrator")


class Orchestrator:
    def __init__(self, research_goal: str, constraints: str = "", config: Optional[AppConfig] = None):
        self.config = config or AppConfig()
        self.memory = ContextMemory(research_goal=research_goal, constraints=constraints)
        self.llm = LLMClient(self.config.llm)
        self.retriever = Retriever(self.config.retriever)
        self.out_dir = Path(self.config.orchestrator.output_dir)

        # 1 instance / agent, dùng chung memory + llm client trong suốt vòng đời.
        # GenerationAgent và ReflectionAgent cần thêm retriever để tra cứu paper.
        self.safety_agent = SafetyAgent(self.llm, self.memory)
        self.generation_agent = GenerationAgent(self.llm, self.memory, self.retriever, self.config.retriever)
        self.proximity_agent = ProximityAgent(self.llm, self.memory)
        self.reflection_agent = ReflectionAgent(self.llm, self.memory, self.retriever, self.config.retriever)
        self.ranking_agent = RankingAgent(self.llm, self.memory)
        self.evolution_agent = EvolutionAgent(self.llm, self.memory)
        self.meta_review_agent = MetaReviewAgent(self.llm, self.memory)

    def _save_state(self) -> None:
        self.memory.save(str(self.out_dir / "state.json"))

    async def check_goal_safety(self):
        """Bước 0: mục tiêu nghiên cứu phải qua kiểm tra an toàn trước khi sinh giả thuyết."""
        safe, reason = await self.safety_agent.run()
        if not safe:
            self.memory.add_safety_alert(f"[goal] {reason}")
            self._save_state()
            raise UnsafeResearchGoalError(reason or "mục tiêu nghiên cứu bị đánh giá là không an toàn")
        logger.info("Safety: mục tiêu nghiên cứu đạt kiểm tra an toàn")

    async def _run_proximity(self):
        scored = await self.proximity_agent.run(
            duplicate_threshold=self.config.orchestrator.proximity_duplicate_threshold,
            max_pairs=self.config.orchestrator.proximity_max_pairs_per_iteration,
        )
        n_active = len(self.memory.get_active_hypotheses())
        logger.info("  Proximity: chấm %d cặp mới, %d giả thuyết còn active sau lọc trùng lặp",
                    len(scored), n_active)

    async def run_phase_1(self):
        """Pha 1: sinh giả thuyết + lọc trùng lặp."""
        logger.info("Pha 1 — Generation & Proximity (iteration %d)", self.memory.iteration)
        new_hyps = await self.generation_agent.run(
            n_hypotheses=self.config.orchestrator.hypotheses_per_iteration
        )
        logger.info("  Generation: sinh %d giả thuyết mới", len(new_hyps))
        await self._run_proximity()

    async def run_phase_2(self):
        """Pha 2: phản biện (giả thuyết chưa review) + xếp hạng tournament."""
        logger.info("Pha 2 — Reflection & Ranking (iteration %d)", self.memory.iteration)
        reviewed = await self.reflection_agent.run()
        logger.info("  Reflection: %d giả thuyết có full review mới", len(reviewed))

        matches = await self.ranking_agent.run(
            n_matches=self.config.orchestrator.matches_per_iteration
        )
        logger.info("  Ranking: đã chạy %d trận đấu", len(matches))

    async def run_phase_3(self):
        """Pha 3: tiến hoá giả thuyết top-rank + meta-review feedback."""
        logger.info("Pha 3 — Evolution & Meta-review (iteration %d)", self.memory.iteration)
        evolved = await self.evolution_agent.run(
            top_k=self.config.orchestrator.top_k_for_evolution
        )
        logger.info("  Evolution: tạo %d giả thuyết cải tiến", len(evolved))

        feedback = await self.meta_review_agent.run(mode="feedback")
        logger.info("  Meta-review: đã sinh feedback cho %d agent", len([v for v in feedback.values() if v]))

    async def run_closing_phase(self):
        """Pha kết thúc: giả thuyết Evolution tạo ở vòng cuối chưa được review/xếp hạng.
        Đánh giá chúng (proximity -> review -> tournament) trước khi viết báo cáo, để
        báo cáo chỉ gồm giả thuyết đã được đánh giá đầy đủ."""
        pending = [h for h in self.memory.get_active_hypotheses() if not h.is_reviewed]
        if not pending:
            return
        logger.info("Pha kết thúc — đánh giá %d giả thuyết chưa được review", len(pending))
        await self._run_proximity()
        await self.run_phase_2()

    async def run(self) -> str:
        """Chạy đủ n_iterations vòng lặp 3 pha, rồi sinh báo cáo cuối cùng."""
        self.out_dir.mkdir(parents=True, exist_ok=True)
        try:
            await self.check_goal_safety()
            for i in range(self.config.orchestrator.n_iterations):
                self.memory.iteration = i + 1
                logger.info("=== Iteration %d/%d ===", i + 1, self.config.orchestrator.n_iterations)
                # Lưu state sau mỗi pha: lỗi giữa chừng chỉ mất tối đa 1 pha.
                await self.run_phase_1()
                self._save_state()
                await self.run_phase_2()
                self._save_state()
                await self.run_phase_3()
                self._save_state()

            await self.run_closing_phase()
            self._save_state()

            logger.info("Sinh báo cáo tổng quan nghiên cứu cuối cùng...")
            report = await self.meta_review_agent.run(mode="final_report")
            report_path = self.out_dir / "final_report.md"
            report_path.write_text(report, encoding="utf-8")
            self._save_state()
            logger.info("Đã lưu báo cáo tại %s", report_path)
            return str(report_path)
        finally:
            # Đóng httpx client của retriever để không leak connection pool.
            await self.retriever.aclose()
