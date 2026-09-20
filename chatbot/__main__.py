"""CLI REPL cho chatbot RAG — phân luồng tự động bằng LLM intent classifier.

Cách chạy:
    python -m chatbot --state output/state.json

Vòng lặp: gõ câu tự nhiên → LLM classify_intent → điều hướng (xử lý nằm trong
chat_agent, helper trong helpers):
- new_topic:      tạo hướng nghiên cứu mới (hỏi goal/constraints → chạy full).
- resume_newfinal: chạy thêm iteration để ra báo cáo mới (giữ data, không reset).
- review/rerun:   ghi nhận góp ý vào hypothesis/agent; KHÔNG tự chạy lại —
                  người dùng nói "chạy lại" khi đã góp ý xong.
- question:       hỏi RAG → trả lời (ReAct loop trong chat_agent.run_rag_agent).
- unknown:        hỏi lại user.

Lệnh slash (không qua LLM): /list /show /ask /rebuild /help /quit
"""
from __future__ import annotations

import argparse
import asyncio

from chatbot.chat_agent import answer, handle_review, handle_resume_newfinal, handle_new_topic
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
  - "Tôi muốn nghiên cứu về enzymeX"        → đổi chủ đề (bỏ data cũ + chạy mới)
  - "đổi chủ đề sang proteinY"              → đổi chủ đề (bỏ data cũ + chạy mới)
  - "Giả thuyết a1 cơ chế chưa chắc"        → ghi góp ý vào a1
  - "Reflection chấm điểm quá dễ dãi"       → ghi góp ý vào reflection_agent
  - "có 2 ý tưởng giống nhau quá"           → ghi góp ý vào proximity_agent
  - "chạy lại"                              → tiếp thu góp ý, chạy thêm vòng,
                                              viết báo cáo mới (giữ nguyên data)
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
                reply = await answer(query, store, llm, top_k=top_k, verbose=True)
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
            # Hỏi goal/constraints ở đây (handler thuần, không gọi input()).
            goal, constraints = helpers.new_session_prompt(intent)
            print(f"Đang chạy hệ thống cho mục tiêu: {goal} (bỏ dữ liệu cũ) ...")
            try:
                new_mem, new_store, msg = await handle_new_topic(
                    goal, constraints, config, state_path, report_path, store_path, model_name)
                print(msg)
                if new_mem is not None:
                    memory, store = new_mem, new_store
            except Exception as e:
                print(f"Lỗi khi tạo chủ đề mới: {e}")
            continue

        if intent.kind == "resume_newfinal":
            ans = input("Chạy thêm bao nhiêu vòng lặp? [mặc định 1]: ").strip()
            n = int(ans) if ans.isdigit() else 1
            print(f"Đang chạy thêm {n} vòng (giữ nguyên giả thuyết + góp ý cũ) ...")
            try:
                memory, store, msg = await handle_resume_newfinal(
                    n, config, memory, state_path, report_path, store_path, model_name)
                print(msg)
            except Exception as e:
                print(f"Lỗi khi chạy tiếp để ra báo cáo mới: {e}")
            continue

        if intent.kind in ("review", "rerun"):
            try:
                print(await handle_review(intent, memory, llm, state_path))
                print("  💡 Gõ \"chạy lại\" để hệ thống tiếp thu góp ý và viết báo cáo mới.")
            except Exception as e:
                print(f"Lỗi khi ghi nhận góp ý: {e}")
            continue

        if intent.kind == "question":
            print("(đang truy xuất + gọi GLM...)")
            try:
                reply = await answer(line, store, llm, top_k=top_k, verbose=True)
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
    # Nếu user truyền --model thì ghi đè; còn không thì giữ model từ MODEL_DEFAULT trong .env
    # (qua LLMConfig) — tránh gửi model sai ("GLM-5.2") → 403 Forbidden trên proxy boltz.
    if args.model:
        config.llm.model = args.model
    cb = config.chatbot
    asyncio.run(_run_repl(
        state_path=args.state,
        store_path=cb.vector_store_path,
        report_path=cb.report_path,
        top_k=cb.top_k,
        model_name=config.chatbot_embedding_model(),
        config=config,
    ))


if __name__ == "__main__":
    main()
