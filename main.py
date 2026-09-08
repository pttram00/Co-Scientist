"""CLI entrypoint để chạy toàn bộ hệ thống multi-agent."""
from __future__ import annotations

import argparse
import asyncio

from config import AppConfig, LLMConfig, OrchestratorConfig
from orchestrator import Orchestrator


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Research Co-Scientist multi-agent framework")
    p.add_argument("--goal", required=True, help="Mục tiêu nghiên cứu")
    p.add_argument("--constraints", default="", help="Ràng buộc / bối cảnh bổ sung")
    p.add_argument("--iterations", type=int, default=3)
    p.add_argument("--hypotheses-per-iteration", type=int, default=6)
    p.add_argument("--matches-per-iteration", type=int, default=10)
    p.add_argument("--top-k-for-evolution", type=int, default=4)
    p.add_argument("--model", default="GLM-5.2")
    p.add_argument("--output-dir", default="output")
    return p.parse_args()


async def main():
    args = parse_args()
    config = AppConfig(
        llm=LLMConfig(model=args.model),
        orchestrator=OrchestratorConfig(
            n_iterations=args.iterations,
            hypotheses_per_iteration=args.hypotheses_per_iteration,
            matches_per_iteration=args.matches_per_iteration,
            top_k_for_evolution=args.top_k_for_evolution,
            output_dir=args.output_dir,
        ),
    )
    orchestrator = Orchestrator(research_goal=args.goal, constraints=args.constraints, config=config)
    report_path = await orchestrator.run()
    print(f"\nHoàn tất. Báo cáo tổng quan nghiên cứu: {report_path}")


if __name__ == "__main__":
    asyncio.run(main())
