"""CLI entrypoint để chạy toàn bộ hệ thống multi-agent."""
from __future__ import annotations

import argparse
import asyncio

from agents.safety_agent import UnsafeResearchGoalError
from config import AppConfig, EmbeddingConfig, LLMConfig, OrchestratorConfig
from orchestrator import Orchestrator


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Research Co-Scientist multi-agent framework")
    p.add_argument("--goal", required=True, help="Mục tiêu nghiên cứu")
    p.add_argument("--constraints", default="", help="Ràng buộc / bối cảnh bổ sung")
    p.add_argument("--iterations", type=int, default=3)
    p.add_argument("--hypotheses-per-iteration", type=int, default=6)
    p.add_argument("--matches-per-iteration", type=int, default=10)
    p.add_argument("--top-k-for-evolution", type=int, default=4)
    p.add_argument("--proximity-max-llm-checks", type=int, default=10,
                   help="Số cặp nghi trùng tối đa ProximityAgent hỏi LLM xác nhận mỗi lượt")
    p.add_argument("--embedding-model", default=EmbeddingConfig.model_name,
                   help="Tên model sentence-transformers dùng cho ProximityAgent")
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
            proximity_max_llm_checks_per_iteration=args.proximity_max_llm_checks,
            output_dir=args.output_dir,
        ),
        embedding=EmbeddingConfig(model_name=args.embedding_model),
    )
    orchestrator = Orchestrator(research_goal=args.goal, constraints=args.constraints, config=config)
    try:
        report_path = await orchestrator.run()
    except UnsafeResearchGoalError as e:
        print(f"\nMục tiêu nghiên cứu bị từ chối vì lý do an toàn: {e}")
        raise SystemExit(1)
    print(f"\nHoàn tất. Báo cáo tổng quan nghiên cứu: {report_path}")


if __name__ == "__main__":
    asyncio.run(main())
