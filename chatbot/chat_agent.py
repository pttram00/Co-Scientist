"""ChatAgent: vòng lặp ReAct RAG + các handler route cho Co-Scientist chatbot.

Gồm 2 phần:
1. ReAct RAG (giống rag_full.py mục 6–10): query → (Thought → Action recall →
   Observation) lặp → Final Answer. RAG có thể gọi recall nhiều lần với từ khoá
   khác nếu context chưa đủ, rồi trả lời DỰA VÀO đoạn tài liệu đã tra (kèm [id]
   gốc, không bịa).
2. Handler route cho từng intent kind (new_topic / resume_newfinal / review_rerun
   / question) — logic nghiệp vụ điều phối hệ thống multi-agent. REPL (__main__)
   chỉ dispatch, logic nằm ở đây.

Dùng LLMClient sẵn có (llm/client.py) — async, qua interface Anthropic /v1/messages.
Lưu ý: LLMClient.complete chỉ gửi 1 message user (không hỗ trợ multi-turn), nên ta
"flatten" history ReAct vào 1 chuỗi user text mỗi bước (pattern stateless per call).
"""
from __future__ import annotations

import re
from typing import List, Optional

from llm.client import LLMClient
from chatbot.retriever_rag import retrieve
from chatbot.vector_store import VectorStore
from chatbot import helpers
from chatbot.intent import IntentResult
from chatbot.user_comment import add_agent_feedback_comment, add_review_comment
from config import AppConfig
from memory.context_memory import ContextMemory
from orchestrator import Orchestrator

# ===========================================================================
# Phần 1: ReAct RAG
# ===========================================================================

SYSTEM_PROMPT = """Bạn là một Agent theo paradigm ReAct (Reasoning + Acting) của hệ thống \
Co-Scientist. Bạn có thể TRA CỨU TÀI LIỆU bằng tool recall trước khi trả lời, để không bịa. \
Đây chính là RAG (Retrieval-Augmented Generation). Kho tài liệu là dữ liệu THẬT từ hệ thống \
multi-agent (giả thuyết, review, báo cáo tổng quan, feedback agent).

Mỗi bước hãy trả lời ĐÚNG theo một trong hai dạng sau (không thêm gì khác):

Dạng 1 — khi cần dùng tool:
Thought: <suy luận ngắn về bước cần làm>
Action: recall(<từ khoá hoặc câu hỏi truy vấn>, k=<số 2-5, tuỳ chọn>)

Dạng 2 — khi đã đủ thông tin để trả lời:
Thought: <suy luận ngắn>
Final Answer: <câu trả lời cuối cho người dùng, dựa vào các đoạn tài liệu đã tra>

Tool có sẵn:
- recall(query, k): RAG — tìm trong kho tài liệu k đoạn liên quan nhất với query
  (mặc định k=3). Dùng NẾU câu hỏi cần kiến thức trong kho (về giả thuyết, review, báo
  cáo, feedback agent). k có thể 2-5. Có thể gọi nhiều lần với từ khoá khác nhau.

QUAN TRỌNG (cách RAG làm trong ReAct):
- Nếu câu hỏi CHẮC CHẮN cần kiến thức trong kho, bước đầu tiên hãy
  Action: recall("<từ khoá chính của câu hỏi>")
- Đọc Observation (các đoạn tài liệu trả về). Nếu đủ -> Final Answer DỰA VÀO đoạn
  vừa tra (không bịa, không lạm dụng kiến thức ngoài). Khi nhắc đến giả thuyết, luôn
  kèm id gốc trong ngoặc vuông, ví dụ [A1] để người hỏi tra cứu được.
- Nếu chưa đủ, gọi recall lần nữa với từ khoá khác.
- Nếu Context không đủ để trả lời, nói rõ "Dựa trên dữ liệu hiện tại tôi không có đủ
  thông tin để ..." — KHÔNG bịa số liệu, không bịa id giả thuyết.

Ví dụ:
Question: Báo cáo có bao nhiêu giả thuyết và xếp hạng tốt nhất là gì?
Thought: Cần đếm giả thuyết và tìm xếp hạng. Tra kho tài liệu.
Action: recall("giả thuyết xếp hạng Elo")
Observation: [hyp:A1] (score=0.42 type=hypothesis) [Giả thuyết A1] (Elo 1320, status ACTIVE)
Nội dung: ...
Thought: Đã có danh sách giả thuyết kèm Elo. Đủ để trả lời.
Final Answer: Báo cáo có ... giả thuyết. Xếp hạng cao nhất hiện là [A1] (Elo 1320) ...
"""

MAX_STEPS = 4


def parse_response(text: str) -> tuple[Optional[str], Optional[str], Optional[str]]:
    """Tách Thought/Action/Final Answer từ raw text LLM — y hệt rag_full.py.

    Trả (thought, action, final). final != None khi LLM ra Final Answer.
    action != None khi LLM ra Action. Cả 2 None -> LLM không ra định dạng đúng.
    """
    thought = action = final = None
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("Thought:"):
            thought = line[len("Thought:"):].strip()
        elif line.startswith("Action:"):
            action = line[len("Action:"):].strip()
        elif line.startswith("Final Answer:"):
            final = line[len("Final Answer:"):].strip()
    return thought, action, final


def _recall(query: str, store: VectorStore, top_k: int = 3) -> str:
    """Tool RAG: với câu hỏi/từ khoá, trả top-k đoạn tài liệu liên quan từ kho.

    Bước RETRIEVAL đóng gói thành 1 tool để ReAct gọi. Trả về text Observation.
    """
    if not isinstance(query, str) or not query.strip():
        return "ERROR: query rỗng."
    chunks = retrieve(query, store, top_k=top_k)
    if not chunks:
        return "(không có đoạn nào trong kho)"
    lines = []
    for c in chunks:
        meta = c.get("metadata", {})
        tag = meta.get("type", "")
        lines.append(f"[{c['id']}] (score={c.get('score', 0):.2f} type={tag}) {c['text']}")
    return "\n".join(lines)


def _call_action(action_str: str, store: VectorStore, top_k: int) -> str:
    """Phân tích 'recall(query, k=...)' rồi gọi _recall. Tool lỗi -> chuỗi ERROR.

    Chỉ hỗ trợ tool recall (ReAct RAG của chatbot không cần add/multiply).
    """
    try:
        name, args_part = action_str.split("(", 1)
        name = name.strip()
        args_part = args_part.strip().rstrip(")").strip()
        if name != "recall":
            return f"ERROR: tool '{name}' không tồn tại. Chỉ có: recall."

        # recall(query[, k=N]) — query là phần text, k là tham số số tuỳ chọn.
        k = top_k
        # Tách k=<số> ra nếu có.
        m_k = re.search(r"k\s*=\s*(\d+)", args_part)
        if m_k:
            k = int(m_k.group(1))
            # Bỏ phần k=... để lấy query.
            args_part = (args_part[:m_k.start()] + args_part[m_k.end():]).strip().rstrip(",").strip()

        # Query: phần còn lại, có thể nằm trong nháy hoặc trơn.
        query = args_part.strip().strip('"').strip("'").strip()
        if not query:
            return "ERROR: recall thiếu query."
        return _recall(query, store, top_k=k)
    except Exception as e:
        return f"ERROR khi chạy action '{action_str}': {e}"


def _build_user_prompt(query: str, history: List[str]) -> str:
    """Flatten câu hỏi + history ReAct (Thought/Action/Observation trước đó) thành
    1 chuỗi user text, vì LLMClient.complete chỉ gửi 1 message user."""
    if not history:
        return (
            f"{query}\n\n"
            f"Hãy ra bước ĐẦU TIÊN theo định dạng ReAct (Thought + Action, hoặc "
            f"Thought + Final Answer nếu không cần tra kho)."
        )
    return (
        f"Câu hỏi ban đầu: {query}\n\n"
        f"--- Lịch sử các bước đã làm ---\n"
        f"{chr(10).join(history)}\n"
        f"--- Hết lịch sử ---\n\n"
        f"Hãy ra bước TIẾP THEO theo định dạng ReAct. Nếu đã đủ thông tin từ "
        f"Observation, hãy ra Final Answer (dựa vào đoạn đã tra, kèm [id] gốc, "
        f"không bịa). Nếu chưa đủ, gọi recall với từ khoá khác."
    )


async def run_rag_agent(query: str, store: VectorStore, llm: LLMClient,
                        top_k: int = 5, max_steps: int = MAX_STEPS,
                        verbose: bool = True) -> str:
    """Vòng lặp ReAct RAG: query → (Thought→Action recall→Observation) lặp → Final Answer.

    verbose=True: in từng bước (Thought/Action/Observation/Final) ra console — giống
    rag_full.py. Trả text Final Answer, hoặc raw nếu LLM không ra định dạng đúng.
    """
    history: List[str] = []   # các dòng Thought/Action/Observation đã xảy ra

    for step in range(1, max_steps + 1):
        user = _build_user_prompt(query, history)
        raw = await llm.complete(SYSTEM_PROMPT, user)
        thought, action, final = parse_response(raw)

        if verbose:
            print(f"\n--- RAG Step {step} ---")
            print(f"Thought: {thought}")
            print(f"Action : {action}")

        # Final Answer -> trả ngay.
        if final is not None:
            if verbose:
                print(f"\nFinal Answer: {final}")
            return final

        # Không có Action/Final -> LLM lệch định dạng, trả raw để user thấy.
        if action is None:
            if verbose:
                print("\nLLM không ra Action/Final -> dừng.")
            return raw

        observation = _call_action(action, store, top_k)
        if verbose:
            print(f"Observation:\n{observation}")

        # Nạp lịch sử để build prompt bước sau (stateless per call).
        history.append(f"Thought: {thought}")
        history.append(f"Action: {action}")
        history.append(f"Observation: {observation}")

    if verbose:
        print("\nĐạt giới hạn bước tối đa, dừng.")
    return "(không ra Final Answer trong giới hạn bước)"


async def answer(query: str, store: VectorStore, llm: LLMClient, top_k: int = 5) -> str:
    """
    Hàm này được gọi khi user hỏi câu hỏi 
    không gọi trực tiếp run_rag_agent để tách biệt luồng ReAct với handler route
    """
    return await run_rag_agent(query, store, llm, top_k=top_k, verbose=True)


# ===========================================================================
# Phần 2: Handler route — điều phối multi-agent theo luồng nghiệp vụ chatbot
# ===========================================================================

async def handle_new_topic(intent: IntentResult, config: AppConfig,
                           state_path: str, report_path: str, store_path: str,
                           model_name: str) -> tuple[ContextMemory | None, VectorStore | None]:
    """
    Xử lý luồng "new_topic": tạo một session mới (reset state cũ) → chạy đủ số vòng lặp 
    → sinh báo cáo cuối cùng → Trả memmory mới
    """
    goal, constraints = helpers.new_session_prompt(intent)
    if not goal:
        print("  Mục tiêu trống — huỷ tạo mới.")
        return None, None
    print(f"Đang chạy hệ thống cho mục tiêu: {goal} (reset dữ liệu cũ) ...")
    orch = Orchestrator.for_new(goal, constraints, config=config)
    await orch.run_full(config.orchestrator.n_iterations,
                        state_path=state_path, report_path=report_path)
    memory = ContextMemory.load(state_path)
    store = helpers.rebuild_store(memory, store_path, report_path, model_name)
    print(f"✓ Hoàn thành. state.json đã lưu, index rebuild ({store.size} chunk).")
    return memory, store


async def handle_resume_newfinal(intent: IntentResult, config: AppConfig,
                                 memory: ContextMemory,
                                 state_path: str, report_path: str,
                                 store_path: str, model_name: str) -> tuple[ContextMemory, VectorStore]:
    """Tiếp tục trên state hiện có (KHÔNG reset) → chạy thêm n iteration để ra final mới.

    Dùng memory hiện tại của REPL, kèm start_iteration để chạy tiếp đúng số vòng."""
    ans = input(f"Chạy thêm bao nhiêu iteration? [mặc định 1]: ").strip()
    n = int(ans) if ans.isdigit() else 1
    start_it = memory.iteration
    print(f"Đang chạy thêm {n} iteration (từ iteration {start_it + 1}) để ra final mới ...")
    orch = Orchestrator(memory, config=config)   # giữ memory hiện tại, KHÔNG reset
    await orch.run_full(n_iterations=n, state_path=state_path,
                        report_path=report_path, start_iteration=start_it + 1)
    memory2 = ContextMemory.load(state_path)
    store = helpers.rebuild_store(memory2, store_path, report_path, model_name)
    print(f"✓ Xong. iteration hiện tại: {memory2.iteration}, index rebuild ({store.size} chunk).")
    return memory2, store


async def handle_review_or_rerun(intent: IntentResult, config: AppConfig,
                                 memory: ContextMemory, llm: LLMClient,
                                 state_path: str, report_path: str,
                                 store_path: str, model_name: str) -> VectorStore | None:
    """Luồng 'review' / 'rerun': gắn comment (nếu có) + rerun_step (nếu có).

    Trả VectorStore mới khi có rerun (rebuild index), None khi không rerun.
    Khi user nhận xét về 1 agent mà không yêu cầu rerun_step, in gợi ý user có
    thể gõ /run <agent> để chạy lại — KHÔNG tự chạy."""
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
        overwrite = helpers.ask_overwrite(intent.rerun_step)
        orch = Orchestrator(memory, config=config)   # dùng memory hiện tại
        msg = await orch.run_step(intent.rerun_step, overwrite=overwrite)
        print(msg)
        orch.save(state_path)
        store = helpers.rebuild_store(memory, store_path, report_path, model_name)
        print(f"  (index rebuild: {store.size} chunk)")
        return store

    # 3) Gợi ý rerun nếu user nhận xét về agent mà không yêu cầu rerun.
    if intent.target_agent and not intent.rerun_step:
        print(f"  💡 Bạn có thể gõ \"/run {intent.target_agent}\" để chạy lại "
              f"agent này (sẽ hỏi keep/overwrite).")

    return None
