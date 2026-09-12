"""CLI REPL cho chatbot RAG.

Cách chạy:
    python -m chatbot --state output/state.json

Lệnh trong REPL:
    <id> <text>          Thêm nhận xét user vào giả thuyết (vd: <a1> cơ chế chưa chắc)
    /ask <câu hỏi>       Hỏi RAG (hoặc gõ câu hỏi thường không có prefix)
    /list                Liệt kê giả thuyết (id | elo | status | nội dung rút gọn)
    /show <id>           In chi tiết 1 giả thuyết
    /run <n>             Chạy thêm n iteration (resume từ state)
    /rebuild             Đánh lại index từ state.json
    /help                Hướng dẫn
    /quit                Thoát
"""
from __future__ import annotations

import argparse
import asyncio
import sys

from chatbot.chat_agent import answer
from chatbot.indexer import build_index
from chatbot.user_comment import add_user_comment, parse_comment
from chatbot.vector_store import VectorStore
from config import AppConfig
from llm.client import LLMClient
from memory.context_memory import ContextMemory
from orchestrator import Orchestrator

HELP = """\
Lệnh:
  <id> <text>   Thêm nhận xét vào giả thuyết (vd: <a1> cơ chế chưa thuyết phục)
  /ask <câu>    Hỏi RAG (hoặc gõ câu hỏi thường không có lệnh)
  /list         Liệt kê giả thuyết
  /show <id>    In chi tiết 1 giả thuyết
  /run <n>      Chạy thêm n iteration từ state hiện tại
  /rebuild      Đánh lại index từ state.json
  /help         Hướng dẫn
  /quit         Thoát
"""


def _print_list(memory: ContextMemory) -> None:
    if not memory.hypotheses:
        print("  (chưa có giả thuyết)")
        return
    for h in memory.hypotheses.values():
        status = h.status.value
        short = h.content.replace("\n", " ")[:60]
        print(f"  [{h.id}] Elo={h.elo_rating:.0f} status={status} | {short}...")
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
        print(f"    - [{r.review_type}] {r.comments}")


def _load_memory_and_store(state_path: str, store_path: str, report_path: str,
                            model_name: str) -> tuple[ContextMemory, VectorStore]:
    """Nạp memory + vector store (build lại nếu store chưa có / lạc state)."""
    memory = ContextMemory.load(state_path)
    store = VectorStore(model_name=model_name)
    # Build lại nếu store chưa tồn tại (lần đầu) — embedding tốn vài giây.
    try:
        store.load(store_path)
    except FileNotFoundError:
        print(f"[chatbot] Chưa có index, đang build từ {state_path} ...")
        store = build_index(memory, store_path=store_path, report_path=report_path,
                            model_name=model_name)
        print(f"[chatbot] Build xong: {store.size} chunk → {store_path}")
    return memory, store


async def _run_repl(state_path: str, store_path: str, report_path: str,
                     top_k: int, model_name: str, config: AppConfig) -> None:
    memory, store = _load_memory_and_store(state_path, store_path, report_path, model_name)
    llm = LLMClient(config.llm)

    print(f"[Co-Scientist Chatbot] {len(memory.hypotheses)} giả thuyết, "
          f"{len(memory.get_papers())} paper, iteration {memory.iteration}.")
    print("Gõ /help để xem lệnh.\n")

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
            store = build_index(memory, store_path=store_path, report_path=report_path,
                                model_name=model_name)
            print(f"✓ Rebuild xong: {store.size} chunk.")
            continue
        if line.startswith("/run "):
            arg = line[len("/run "):].strip()
            n = int(arg) if arg.isdigit() else 1
            print(f"Đang chạy thêm {n} iteration (resume)...")
            orch = Orchestrator.from_memory(memory, config=config)
            await orch.run_iterations(n, state_path)
            # Memory đã thay đổi qua cùng object — rebuild index.
            memory = ContextMemory.load(state_path)
            store = build_index(memory, store_path=store_path, report_path=report_path,
                                model_name=model_name)
            print(f"✓ Xong. state.json đã cập nhật, index rebuild ({store.size} chunk).")
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

        # Comment của user: "<id> <text>"
        if parse_comment(line):
            try:
                msg = await add_user_comment(memory, llm, line, state_path)
                print(msg)
                # Comment mới có thể đáng được index — rebuild nhẹ (corpus nhỏ).
                store = build_index(memory, store_path=store_path, report_path=report_path,
                                    model_name=model_name)
            except Exception as e:
                print(f"Lỗi khi ghi nhận xét: {e}")
            continue

        # Mặc định: coi như câu hỏi RAG.
        print("(đang truy xuất + gọi GLM...)")
        try:
            reply = await answer(line, store, llm, top_k=top_k)
            print(reply)
        except Exception as e:
            print(f"Lỗi khi trả lời: {e}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Co-Scientist chatbot RAG")
    p.add_argument("--state", default="output/state.json", help="Path file state.json")
    p.add_argument("--model", default="GLM-5.2", help="Tên model LLM (indent)")
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
