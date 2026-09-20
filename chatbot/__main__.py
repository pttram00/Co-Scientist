"""CLI REPL cho chatbot RAG — phân luồng tự động bằng LLM intent classifier.

Cách chạy:
    python -m chatbot --state output/state.json

Vòng lặp: gõ câu tự nhiên → LLM classify_intent → điều hướng (xử lý nằm trong
chat_agent, helper trong helpers):
- new_topic:      tạo hướng nghiên cứu mới (hỏi goal/constraints → chạy full).
- resume_newfinal: chạy thêm iteration để ra final mới (giữ data, không reset).
- review/rerun:   nhận xét → gắn vào hypothesis/agent; nếu kèm rerun_step → rerun.
- question:       hỏi RAG → trả lời (ReAct loop trong chat_agent.run_rag_agent).
- unknown:        hỏi lại user.

Lệnh slash (không qua LLM): /list /show /ask /rebuild /help /quit
"""
from __future__ import annotations

import argparse
import asyncio

from chatbot.chat_agent import answer, handle_review_or_rerun, handle_resume_newfinal, handle_new_topic
from chatbot.intent import classify_intent
from chatbot import helpers
from config import AppConfig
from llm.client import LLMClient

HELP = """\
Lệnh slash (không qua LLM):
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
  - "Reflection chấm sai, chạy lại"         → nhận xét + rerun Reflection
  - "có 2 ý tưởng giống nhau quá"           → nhận xét vào proximity (gợi ý rerun)
  - "Chạy lại Ranking"                      → rerun Ranking
  - "Báo cáo có bao nhiêu giả thuyết?"      → hỏi RAG
"""


async def _run_repl(state_path: str, store_path: str, report_path: str,
                    top_k: int, model_name: str, config: AppConfig) -> None:
    memory, store = helpers.load_memory_and_store(state_path, store_path, report_path, model_name)
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
            helpers.print_list(memory)
            continue
        if line.startswith("/show "):
            helpers.print_show(memory, line[len("/show "):].strip())
            continue
        if line == "/rebuild":
            print("Đang build lại index...")
            try:
                store = helpers.rebuild_store(memory, store_path, report_path, model_name)
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
                new_mem, new_store = await handle_new_topic(
                    intent, config, state_path, report_path, store_path, model_name)
                if new_mem is not None:
                    memory, store = new_mem, new_store
            except Exception as e:
                print(f"Lỗi khi tạo chủ đề mới: {e}")
            continue

        if intent.kind == "resume_newfinal":
            try:
                new_mem, new_store = await handle_resume_newfinal(
                    intent, config, memory, state_path, report_path, store_path, model_name)
                if new_mem is not None:
                    memory, store = new_mem, new_store
            except Exception as e:
                print(f"Lỗi khi chạy tiếp để ra final mới: {e}")
            continue

        if intent.kind in ("review", "rerun"):
            try:
                new_store = await handle_review_or_rerun(
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
    p.add_argument("--model", default=None, help="Tên model LLM (bỏ qua để dùng MODEL_DEFAULT trong .env)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    config = AppConfig()
    # Nếu user truyền --model則 ghi đè; còn không thì giữ model từ MODEL_DEFAULT trong .env
    # (qua LLMConfig) — tránh gửi model sai ("GLM-5.2") → 403 Forbidden trên proxy boltz.
    if args.model:
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
