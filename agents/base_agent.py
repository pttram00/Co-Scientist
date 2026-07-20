"""Lớp cơ sở cho mọi agent trong hệ thống."""
from __future__ import annotations

from abc import ABC, abstractmethod

from llm.client import LLMClient
from memory.context_memory import ContextMemory


class BaseAgent(ABC):
    name: str = "base_agent"

    def __init__(self, llm: LLMClient, memory: ContextMemory):
        self.llm = llm
        self.memory = memory

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
