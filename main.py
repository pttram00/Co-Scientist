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

def print_storage_status(self) -> None:
        """In ra toàn bộ dữ liệu đang được lưu trữ trong bộ nhớ của hệ thống."""
        print("="*60)
        print(" TỔNG QUAN BỘ NHỚ LƯU TRỮ ".center(60, "="))
        print("="*60)

        # 1. In Hypotheses (Các giả thuyết)
        print("\n[1] HYPOTHESES (GIẢ THUYẾT)")
        if not self.hypotheses:
            print("  (Trống)")
        else:
            for key, hypothesis in self.hypotheses.items():
                print(f"  - {key}: {hypothesis}")

        # 2. In Proximity Graph (Mức độ tương đồng)
        print("\n[2] PROXIMITY GRAPH (ĐỒ THỊ TƯƠNG ĐỒNG)")
        if not self.proximity_graph:
            print("  (Trống)")
        else:
            for source_hyp, target_list in self.proximity_graph.items():
                print(f"  - {source_hyp} tương đồng với:")
                for target_hyp, score in target_list:
                    print(f"      -> {target_hyp} (Điểm: {score:.4f})")

        # 3. In Match History (Lịch sử các trận đấu)
        print("\n[3] MATCH HISTORY (LỊCH SỬ ĐỐI ĐẦU)")
        if not self.match_history:
            print("  (Trống)")
        else:
            for i, match in enumerate(self.match_history, 1):
                print(f"  - Trận {i}: {match}")

        # 4. In Meta Review Notes (Nhận xét)
        print("\n[4] META REVIEW NOTES (NHẬN XÉT CỦA META REVIEWER)")
        if not self.meta_review_notes:
            print("  (Trống)")
        else:
            for i, note in enumerate(self.meta_review_notes, 1):
                print(f"  {i}. {note}")

        # 5. In Agent Feedback (Phản hồi từ Agent)
        print("\n[5] AGENT FEEDBACK (PHẢN HỒI CỦA AGENT)")
        if not self.agent_feedback:
            print("  (Trống)")
        else:
            for agent_name, feedbacks in self.agent_feedback.items():
                print(f"  - Agent [{agent_name}]:")
                for fb in feedbacks:
                    print(f"      + {fb}")
        
        print("\n" + "="*60)

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
    orchestrator = Orchestrator.for_new(research_goal=args.goal, constraints=args.constraints, config=config)
    report_path = await orchestrator.run_full(
        state_path=str(args.output_dir) + "/state.json",
        report_path=str(args.output_dir) + "/final_report.md",
    )
    print_storage_status(orchestrator.memory)  # In ra toàn bộ dữ liệu đang được lưu trữ trong bộ nhớ của hệ thống
    print(f"\nHoàn tất. Báo cáo tổng quan nghiên cứu: {report_path}")


if __name__ == "__main__":
    asyncio.run(main())
