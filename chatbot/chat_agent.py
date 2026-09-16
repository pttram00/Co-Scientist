"""ChatAgent: tổng hợp câu trả lời từ top-k chunk + GLM.

Luồng: query → retriever top-k → build context → GLM.complete → trả text.
Dùng LLMClient sẵn có (llm/client.py) — tự nhận fallback brotli nếu đã sửa.
"""
from __future__ import annotations

from typing import List

from llm.client import LLMClient
from chatbot.retriever_rag import retrieve
from chatbot.vector_store import VectorStore

SYSTEM_PROMPT = """Bạn là trợ lí nghiên cứu của hệ thống Co-Scientist. Trả lời
dựa CHẶN CHẼ VÀO phần Context dưới đây — đây là dữ liệu thực từ hệ thống
multi-agent (giả thuyết, review, báo cáo). Quy tắc:

- Khi nhắc đến giả thuyết, luôn kèm id gốc trong ngoặc vuông, ví dụ [A1] để người
  hỏi tra cứu được.
- Nếu Context không đủ để trả lời, nói rõ "Dựa trên dữ liệu hiện tại tôi không
  có đủ thông tin để ..." — KHÔNG bịa số liệu, không bịa id giả thuyết.
- Trả lời tiếng Việt, súc tích.
"""


def _build_context(chunks: List[dict]) -> str:
    if not chunks:
        return "(không có context — kho dữ liệu trống)"
    blocks = []
    for i, c in enumerate(chunks, 1):
        meta = c.get("metadata", {})
        tag = meta.get("type", "")
        blocks.append(f"[Context {i} | type={tag} | id={c['id']} | score={c.get('score', 0):.2f}]\n{c['text']}")
    return "\n\n".join(blocks)


async def answer(query: str, store: VectorStore, llm: LLMClient, top_k: int = 5) -> str:
    """Trả câu trả lời cho query dựa trên top-k chunk từ store."""
    chunks = retrieve(query, store, top_k=top_k)
    context = _build_context(chunks)
    user = f"Câu hỏi: {query}\n\n--- Context ---\n{context}\n--- Hết context ---"
    return await llm.complete(SYSTEM_PROMPT, user)
