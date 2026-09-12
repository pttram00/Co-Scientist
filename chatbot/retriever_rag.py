"""Retriever RAG: query → top-k chunk từ VectorStore.

Corpus nhỏ (~vài chục chunk) nên top-k cosine là đủ — chưa cần rerank. Nếu sau
này corpus lớn, có thể thêm bước LLM rerank ở đây mà không đổi interface.
"""
from __future__ import annotations

from typing import List

from chatbot.vector_store import VectorStore


def retrieve(query: str, store: VectorStore, top_k: int = 5) -> List[dict]:
    return store.query(query, top_k=top_k)
