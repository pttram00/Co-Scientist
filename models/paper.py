"""Data model cho một bài báo khoa học được tra cứu làm grounding.

GenerationAgent tra cứu paper từ 3 nguồn (arXiv + Semantic Scholar + OpenAlex),
gộp lại và cache vào ContextMemory. Paper đơn thuần là dữ liệu đọc-vào, không
chứa logic — để retriever và memory thao tác.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Paper:
    id: str                       # id nội bộ: "<source>:<native_id>" (vd "arxiv:2401.12345")
    title: str
    abstract: str
    year: Optional[int] = None
    citations: int = 0            # số lượt trích dẫn (lấy từ S2/OpenAlex); arXiv thường = 0
    source: str = ""               # "arxiv" | "semantic_scholar" | "openalex"

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "title": self.title,
            "abstract": self.abstract,
            "year": self.year,
            "citations": self.citations,
            "source": self.source,
        }

    @staticmethod
    def from_dict(d: dict) -> "Paper":
        return Paper(
            id=d["id"],
            title=d.get("title", ""),
            abstract=d.get("abstract", ""),
            year=d.get("year"),
            citations=int(d.get("citations", 0)),
            source=d.get("source", ""),
        )
