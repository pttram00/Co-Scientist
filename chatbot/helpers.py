"""Helper thuần cho chatbot: nạp/xây lại store, hỏi user, in giả thuyết.

Dùng chung cho REPL (__main__) và các handler trong chat_agent. Không chứa logic
nghiệp vụ — chỉ I/O và tiện ích.
"""
from __future__ import annotations

from memory.context_memory import ContextMemory

from chatbot.indexer import build_index
from chatbot.vector_store import VectorStore


def load_memory_and_store(state_path: str, store_path: str, report_path: str,
                          model_name: str) -> tuple[ContextMemory, VectorStore]:
    """Nạp ContextMemory + VectorStore. Nếu chưa có index thì build từ state."""
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


def rebuild_store(memory: ContextMemory, store_path: str, report_path: str,
                  model_name: str) -> VectorStore:
    """Đánh lại index từ memory hiện tại (dùng sau khi rerun/new_topic)."""
    return build_index(memory, store_path=store_path, report_path=report_path,
                       model_name=model_name)


def ask_overwrite(step: str) -> bool:
    """Hỏi user keep (append) hay overwrite (xoá output cũ). Trả True=overwrite."""
    while True:
        ans = input(f"Bước '{step}' tạo output mới — keep (append) hay overwrite (xoá cũ)? [k/o]: ").strip().lower()
        if ans in ("o", "overwrite"):
            return True
        if ans in ("k", "keep"):
            return False
        print("  Gõ 'k' (keep) hoặc 'o' (overwrite).")


def new_session_prompt(intent) -> tuple[str, str]:
    """Hỏi goal/constraints cho luồng new_topic (nếu LLM chưa trích ra từ câu)."""
    goal = intent.research_goal or input("Mục tiêu nghiên cứu: ").strip()
    constraints = input("Ràng buộc/bối cảnh (Enter để bỏ qua): ").strip()
    return goal, constraints


def print_list(memory: ContextMemory) -> None:
    """Lệnh /list — liệt kê giả thuyết."""
    if not memory.hypotheses:
        print("  (chưa có giả thuyết)")
        return
    for h in memory.hypotheses.values():
        short = h.content.replace("\n", " ")[:60]
        print(f"  [{h.id}] Elo={h.elo_rating:.0f} status={h.status.value} | {short}...")
    print(f"  ({len(memory.hypotheses)} giả thuyết, iteration hiện tại: {memory.iteration})")


def print_show(memory: ContextMemory, hid: str) -> None:
    """Lệnh /show <id> — in chi tiết 1 giả thuyết."""
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
