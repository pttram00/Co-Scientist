"""CLI REPL cho chatbot RAG — phân luồng tự động bằng LLM intent classifier.

Cách chạy:
    python -m chatbot --state output/state.json

Vòng lặp: gõ câu tự nhiên → LLM classify_intent → điều hướng:
- "new":      tạo hướng nghiên cứu mới (hỏi goal/constraints → chạy full).
- "review":   nhận xét → gắn vào hypothesis (Review) hoặc agent (agent_feedback)
              tuỳ ngữ nghĩa; nếu kèm rerun_step → hỏi keep/overwrite → rerun.
- "rerun":    yêu cầu chạy lại 1 bước (agent) → hỏi keep/overwrite → rerun.
- "question": hỏi RAG → trả lời.
- "unknown":  hỏi lại user.

Lệnh g.rawQuery (không qua LLM): /list /show /ask /rebuild /help /quit
"""
from __future__ import annotations

import argparse
import asyncio
import sys

from chatbot.chat_agent import answer
from chatbot.indexer import build_index
from chatbot.intent import IntentResult, classify_intent, VALID_STEPS
from chatbot.user_comment import add_agent_feedback_comment, add_review_comment
from chatbot.vector_store import VectorStore
from config import AppConfig
from llm.client import LLMClient
from memory.context_memory import ContextMemory
from orchestrator import Orchestrator

HELP = """\
Lệnh g RawQuery (không qua LLM):
  /list            Liệt kê giả thuyết
  /show <id>       In chi tiết 1 giả thuyết
  /ask <câu>       Hỏi RAG
  /rebuild         Đánh lại index từ state.json
  /help            Hướng dẫn
  /quit            Thoát

Hoặc gõ câu tự nhiên:
  - "Tôi muốn nghiên cứu về enzymeX"        → đổi chủ đề (reset + chạy mới)
  - "đổi chủ đề sang proteinY"              → đổi chủ đề (reset + chạy mới)
  - "tạo lại giả thuyết dựa trên nhận xét"  → chạy tiếp để ra final mới (giữ data)
  - "Giả thuyết a1 cơ chế chưa chắc"        → nhận xét vào a1
  - "Reflection chấm sai, chạy lại"        → nhận xét + rerun Reflection
  - "có 2 ý tưởng giống nhau quá"          → nhận xét vào proximity (gợi ý rerun)
  - "Chạy lại Ranking"                      → rerun Ranking
  - "Báo cáo có bao nhiêu giả thuyết?"     → hỏi RAG
"""


def _print_list(memory: ContextMemory) -> None:
    if not memory.hypotheses:
        print("  (chưa có giả thuyết)")
        return
    for h in memory.hypotheses.values():
        short = h.content.replace("\n", " ")[:60]
        print(f"  [{h.id}] Elo={h.elo_rating:.0f} status={h.status.value} | {short}...")
    print(f"  ({len(memory.hypotheses)} giả thuyết, iteration hiện tại: {memory.iteration})")


def _print_show(memory: ContextMemory, hid: str) -> None:
    h = memory.hypotheses.get(hid)
    if not h:
        print(f"  Không có giả thuyết id '{hid}'.")
        return
    print(f"  id: {h.id}")
    print(f"  status: {h.status.value} | Elo: {h.elo_rating:.0f} | matches: {h.matches_played}")
    print(f"  Nội dung: {h.content}")
    print(f"  Cơ chế: {h.rationale}")
    print(f"  Thí nghiệm đề xuất: {h.suggested_experiment or '(chưa có)'}")
    print(f"  Điểm TB: correctness {h.average_score('correctness'):.1f}"
          f"/novelty {h.average_score('novelty'):.1f}"
          f"/feasibility {h.average_score('feasibility'):.1f}")
    print(f"  Reviews ({len(h.reviews)}):")
    for r in h.reviews:
        print(f"    - [{r.review_type}] {r.comments[:80]}")


def _load_memory_and_store(state_path: str, store_path: str, report_path: str,
                            model_name: str) -> tuple[ContextMemory, VectorStore]:
    memory = ContextMemory.load(state_path)
    store = VectorStore(model_name=model_name)
    try:
        store.load(store_path)
    except FileNotFoundError:
        print(f"[chatbot] Chưa có index, đang build từ {state_path} ...")
        store = build_index(memory, store_path=store_path, report_path=report_path,
                            model_name=model_name)
        print(f"[chatbot] Build xong: {store.size} chunk → {store_path}")
    return memory, store


def _rebuild_store(memory: ContextMemory, store_path: str, report_path: str,
                    model_name: str) -> VectorStore:
    return build_index(memory, store_path=store_path, report_path=report_path,
                       model_name=model_name)


def _ask_overwrite(step: str) -> bool:
    """Hỏi user keep (append) hay overwrite (xoá output cũ). Trả True=overwrite."""
    while True:
        ans = input(f"Bước '{step}' tạo output mới — keep (append) hay overwrite (xoá cũ)? [k/o]: ").strip().lower()
        if ans in ("o", "overwrite"):
            return True
        if ans in ("k", "keep"):
            return False
        print("  Gõ 'k' (keep) hoặc 'o' (overwrite).")


def _new_session_prompt(intent: IntentResult) -> tuple[str, str]:
    """Hỏi goal/constraints cho luồng 'new' (nếu LLM chưa trích ra từ câu)."""
    goal = intent.research_goal or input("Mục tiêu nghiên cứu: ").strip()
    constraints = input("Ràng buộc/bối cảnh (Enter để bỏ qua): ").strip()
    return goal, constraints


async def _handle_new_topic(intent: IntentResult, config: AppConfig,
                              state_path: str, report_path: str, store_path: str,
                              model_name: str) -> tuple[ContextMemory, VectorStore]:
    """Luồng 'new_topic': đổi chủ đề → reset memory → chạy full iteration → rebuild."""
    goal, constraints = _new_session_prompt(intent)
    if not goal:
        print("  Mục tiêu trống — huỷ tạo mới.")
        return None, None
    print(f"Đang chạy hệ thống cho mục tiêu: {goal} (reset dữ liệu cũ) ...")
    orch = Orchestrator.for_new(goal, constraints, config=config)
    await orch.run_full(config.orchestrator.n_iterations,
                        state_path=state_path, report_path=report_path)
    memory = ContextMemory.load(state_path)
    store = _rebuild_store(memory, store_path, report_path, model_name)
    print(f"✓ Hoàn thành. state.json đã lưu, index rebuild ({store.size} chunk).")
    return memory, store


async def _handle_resume_newfinal(intent: IntentResult, config: AppConfig,
                                    memory: ContextMemory,
                                    state_path: str, report_path: str,
                                    store_path: str, model_name: str) -> tuple[ContextMemory, VectorStore]:
    """Luồng 'resume_newfinal': tiếp tục trên state hiện có (KHÔNG reset) → chạy
    thêm iteration để ra final_report mới, tận dụng dữ liệu phiên trước. Dùng
    memory hiện tại của REPL, kèm start_iteration để chạy tiếp đúng số vòng."""
    ans = input(f"Chạy thêm bao nhiêu iteration? [mặc định 1]: ").strip()
    n = int(ans) if ans.isdigit() else 1
    start_it = memory.iteration
    print(f"Đang chạy thêm {n} iteration (từ iteration {start_it + 1}) để ra final mới ...")
    orch = Orchestrator(memory, config=config)   # giữ memory hiện tại, KHÔNG reset
    try:
        await orch.run_full(n_iterations=n, state_path=state_path,
                            report_path=report_path, start_iteration=start_it + 1)
    finally:
        pass  # run_self.aclose() trong run_full; memory đối tượng dùng chung.
    memory2 = ContextMemory.load(state_path)
    store = _rebuild_store(memory2, store_path, report_path, model_name)
    print(f"✓ Xong. iteration hiện tại: {memory2.iteration}, index rebuild ({store.size} chunk).")
    return memory2, store


async def _handle_review_or_rerun(intent: IntentResult, config: AppConfig,
                                   memory: ContextMemory, llm: LLMClient,
                                   state_path: str, report_path: str,
                                   store_path: str, model_name: str) -> VectorStore:
    """Luồng 'review' / 'rerun': gắn comment (nếu có) + rerun_step (nếu có).
    Khi user nhận xét về 1 agent (target_agent) mà không yêu cầu rerun_step,
    in gợi ý user có thể gõ /run <agent> để chạy lại — KHÔNG tự chạy."""
    # 1) Gắn comment tuỳ ngữ nghĩa.
    if intent.comment_text:
        if intent.target_agent:
            msg = await add_agent_feedback_comment(memory, intent.target_agent,
                                                     intent.comment_text, state_path)
            print(msg)
        elif intent.target_hypothesis_id:
            msg = await add_review_comment(memory, llm, intent.target_hypothesis_id,
                                             intent.comment_text, state_path)
            print(msg)
        else:
            print(f"  → Bạn nhận xét: {intent.comment_text} (chưa gắn vào đối tượng cụ thể).")

    # 2) Rerun bước (nếu có).
    if intent.rerun_step:
        overwrite = _ask_overwrite(intent.rerun_step)
        orch = Orchestrator(memory, config=config)   # dùng memory hiện tại
        msg = await orch.run_step(intent.rerun_step, overwrite=overwrite)
        print(msg)
        orch.save(state_path)
        store = _rebuild_store(memory, store_path, report_path, model_name)
        print(f"  (index rebuild: {store.size} chunk)")
        return store

    # 3) Gợi ý rerun nếu user nhận xét về agent mà không yêu cầu rerun.
    if intent.target_agent and not intent.rerun_step:
        print(f"  💡 Bạn có thể gõ \"/run {intent.target_agent}\" để chạy lại "
              f"agent này (sẽ hỏi keep/overwrite).")

    return None


async def _run_repl(state_path: str, store_path: str, report_path: str,
                     top_k: int, model_name: str, config: AppConfig) -> None:
    memory, store = _load_memory_and_store(state_path, store_path, report_path, model_name)
    llm = LLMClient(config.llm)

    print(f"[Co-Scientist Chatbot] {len(memory.hypotheses)} giả thuyết, "
          f"{len(memory.get_papers())} paper, iteration {memory.iteration}.")
    print("Gõ câu tự nhiên hoặc /help.\n")

    while True:
        try:
            line = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nTạm biệt.")
            break
        if not line:
            continue

        if line in ("/quit", "/exit"):
            print("Tạm biệt.")
            break
        if line == "/help":
            print(HELP)
            continue
        if line == "/list":
            _print_list(memory)
            continue
        if line.startswith("/show "):
            _print_show(memory, line[len("/show "):].strip())
            continue
        if line == "/rebuild":
            print("Đang build lại index...")
            try:
                store = _rebuild_store(memory, store_path, report_path, model_name)
                print(f"✓ Rebuild xong: {store.size} chunk.")
            except Exception as e:
                print(f"Lỗi rebuild index: {e}")
            continue
        if line.startswith("/ask "):
            query = line[len("/ask "):].strip()
            if not query:
                print("Thiếu câu hỏi.")
                continue
            print("(đang truy xuất + gọi GLM...)")
            try:
                reply = await answer(query, store, llm, top_k=top_k)
                print(reply)
            except Exception as e:
                print(f"Lỗi khi trả lời: {e}")
            continue

        # Câu tự nhiên → classify_intent.
        print("(đang phân loại ý định...)")
        try:
            intent = await classify_intent(line, llm, memory)
        except Exception as e:
            print(f"Lỗi classify: {e}")
            continue

        if intent.kind == "new_topic":
            try:
                new_mem, new_store = await _handle_new_topic(
                    intent, config, state_path, report_path, store_path, model_name)
                if new_mem is not None:
                    memory, store = new_mem, new_store
            except Exception as e:
                print(f"Lỗi khi tạo chủ đề mới: {e}")
            continue

        if intent.kind == "resume_newfinal":
            try:
                new_mem, new_store = await _handle_resume_newfinal(
                    intent, config, memory, state_path, report_path, store_path, model_name)
                if new_mem is not None:
                    memory, store = new_mem, new_store
            except Exception as e:
                print(f"Lỗi khi chạy tiếp để ra final mới: {e}")
            continue

        if intent.kind in ("review", "rerun"):
            try:
                new_store = await _handle_review_or_rerun(
                    intent, config, memory, llm, state_path, report_path, store_path, model_name)
                if new_store is not None:
                    store = new_store
            except Exception as e:
                print(f"Lỗi khi xử lý nhận xét/rerun: {e}")
            continue

        if intent.kind == "question":
            print("(đang truy xuất + gọi GLM...)")
            try:
                reply = await answer(line, store, llm, top_k=top_k)
                print(reply)
            except Exception as e:
                print(f"Lỗi khi trả lời: {e}")
            continue

        # unknown
        print("Tôi chưa hiểu rõ ý bạn. Bạn muốn:")
        print("  1) Tạo hướng nghiên cứu mới, hay")
        print("  2) Nhận xét/chỉnh sửa hướng có sẵn, hay")
        print("  3) Hỏi thông tin về kết quả đã có?")
        print("Hãy nói rõ hơn (vd: 'tôi muốn nghiên cứu...', 'giả thuyết a1 ...',")
        print("'chạy lại reflection', hoặc dùng /ask để hỏi RAG).")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Co-Scientist chatbot RAG (phân luồng tự động)")
    p.add_argument("--state", default="output/state.json", help="Path file state.json")
    p.add_argument("--model", default="GLM-5.2", help="Tên model LLM")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    config = AppConfig()
    config.llm.model = args.model
    cb = config.chatbot
    asyncio.run(_run_repl(
        state_path=args.state,
        store_path=cb.vector_store_path,
        report_path=cb.report_path,
        top_k=cb.top_k,
        model_name=cb.embedding_model,
        config=config,
    ))


if __name__ == "__main__":
    main()
