"""Orchestrator: điều phối vòng lặp 3 pha giữa 6 agent, dùng ContextMemory
làm nguồn sự thật duy nhất (single source of truth) cho toàn hệ thống.
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
from config import AppConfig
from llm.client import LLMClient
from memory.context_memory import ContextMemory

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("orchestrator")


class Orchestrator:
    def __init__(self, research_goal: str, constraints: str = "", config: Optional[AppConfig] = None):
        self.config = config or AppConfig()
        self.memory = ContextMemory(research_goal=research_goal, constraints=constraints)
        self.llm = LLMClient(self.config.llm)

        # 1 instance / agent, dùng chung memory + llm client trong suốt vòng đời
        self.generation_agent = GenerationAgent(self.llm, self.memory)
        self.proximity_agent = ProximityAgent(self.llm, self.memory)
        self.reflection_agent = ReflectionAgent(self.llm, self.memory)
        self.ranking_agent = RankingAgent(self.llm, self.memory)
        self.evolution_agent = EvolutionAgent(self.llm, self.memory)
        self.meta_review_agent = MetaReviewAgent(self.llm, self.memory)

    async def run_phase_1(self):
        """Pha 1: sinh giả thuyết + lọc trùng lặp."""
        logger.info("Pha 1 — Generation & Proximity (iteration %d)", self.memory.iteration)
        new_hyps = await self.generation_agent.run(
            n_hypotheses=self.config.orchestrator.hypotheses_per_iteration
        )
        logger.info("  Generation: sinh %d giả thuyết mới", len(new_hyps))

        await self.proximity_agent.run(
            duplicate_threshold=self.config.orchestrator.proximity_duplicate_threshold
        )
        n_active = len(self.memory.get_active_hypotheses())
        logger.info("  Proximity: %d giả thuyết còn active sau lọc trùng lặp", n_active)

    async def run_phase_2(self):
        """Pha 2: phản biện + xếp hạng tournament."""
        logger.info("Pha 2 — Reflection & Ranking (iteration %d)", self.memory.iteration)
        reviewed = await self.reflection_agent.run()
        logger.info("  Reflection: đã phản biện %d giả thuyết", len(reviewed))

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

    async def run(self) -> str:
        """Chạy đủ n_iterations vòng lặp 3 pha, rồi sinh báo cáo cuối cùng."""
        out_dir = Path(self.config.orchestrator.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        for i in range(self.config.orchestrator.n_iterations):
            self.memory.iteration = i + 1
            logger.info("=== Iteration %d/%d ===", i + 1, self.config.orchestrator.n_iterations)
            await self.run_phase_1()
            await self.run_phase_2()
            await self.run_phase_3()
            self.memory.save(str(out_dir / "state.json"))

        logger.info("Sinh báo cáo tổng quan nghiên cứu cuối cùng...")
        report = await self.meta_review_agent.run(mode="final_report")
        report_path = out_dir / "final_report.md"
        report_path.write_text(report, encoding="utf-8")
        logger.info("Đã lưu báo cáo tại %s", report_path)
        return str(report_path)
