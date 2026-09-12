"""Lớp cơ sở cho mọi agent trong hệ thống."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

from llm.client import LLMClient
from memory.context_memory import ContextMemory


class BaseAgent(ABC):
    name: str = "base_agent"

    def __init__(self, llm: LLMClient, memory: ContextMemory, embedding: Optional[object] = None):
        self.llm = llm
        self.memory = memory
        # embedding: tuỳ chọn (hiện chỉ ProximityAgent dùng — nhận EmbeddingConfig).
        # Các agent khác không truyền -> mặc định None, không bị vỡ chữ ký.
        self.embedding = embedding

    def feedback_block(self) -> str:
        """Trả về feedback mà MetaReviewAgent đã để lại cho agent này ở vòng
        trước, để chèn vào prompt (vòng lặp cải thiện liên tục)."""
        notes = self.memory.get_agent_feedback(self.name)
        if not notes:
            return ""
        joined = "\n".join(f"- {n}" for n in notes[-5:])
        return f"\n\nGhi chú cải thiện từ Meta-review agent (hãy áp dụng):\n{joined}"

    @abstractmethod
    async def run(self, **kwargs):
        """Thực thi vai trò của agent, đọc/ghi trực tiếp vào self.memory."""
        raise NotImplementedError
