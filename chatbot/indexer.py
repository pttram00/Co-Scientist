"""Indexer: ContextMemory → chunks → VectorStore.

Tạo 2 nhóm chunk:
- Nhóm 1 (hypotheses + reviews): mỗi hypothesis 1 chunk (content, rationale,
  thí nghiệm đề xuất, điểm trung bình, toàn bộ reviews).
- Nhóm 2 (report + meta notes + agent_feedback): final_report.md tách theo
  heading `## `, mỗi meta_note 1 chunk, mỗi agent 1 chunk feedback.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import List

from memory.context_memory import ContextMemory
from chatbot.vector_store import VectorStore


def _hypothesis_chunks(memory: ContextMemory) -> List[dict]:
    """ 
    Chuyển các giả thuyết được lưu trong memmory để đưa vào chunk để dễ dàng truy suất hơn sau này.
    """
    chunks = []
    for h in memory.hypotheses.values():
        reviews_lines = []
        for r in h.reviews:
            tag = f"[{r.review_type}]"
            reviews_lines.append(f"  - {tag} {r.comments}".rstrip())
        reviews_block = "\n".join(reviews_lines) if reviews_lines else "  (chưa có)"
        text = (
            f"[Giả thuyết {h.id}] (Elo {h.elo_rating:.0f}, status {h.status.value})\n"
            f"Nội dung: {h.content}\n"
            f"Cơ chế: {h.rationale}\n"
            f"Thí nghiệm đề xuất: {h.suggested_experiment or '(chưa có)'}\n"
            f"Điểm trung bình: correctness {h.average_score('correctness'):.1f}"
            f"/novelty {h.average_score('novelty'):.1f}"
            f"/feasibility {h.average_score('feasibility'):.1f}\n"
            f"Phản biện:\n{reviews_block}"
        )
        chunks.append({
            "id": f"hyp:{h.id}",
            "text": text,
            "metadata": {"type": "hypothesis", "id": h.id, "elo": h.elo_rating},
        })
    return chunks


def _split_report_by_heading(markdown: str) -> List[dict]:
    """Tách final_report.md theo heading '## ' → mỗi section 1 chunk."""
    # Tìm vị trí các '## ' ở đầu dòng rôi chia thành các section lưu trữ trong chunk có dạng {id, text, metadata}
    matches = list(re.finditer(r"^##\s+(.+)$", markdown, flags=re.MULTILINE))
    if not matches:
        # Không có heading → gom cả file thành 1 chunk (nếu không rỗng).
        text = markdown.strip()
        return [{"id": "report:full", "text": text,
                 "metadata": {"type": "report", "section": "full"}}] if text else []
    chunks = []

    # 
    for i, m in enumerate(matches):
        heading = m.group(1).strip()
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(markdown)
        body = markdown[start:end].strip()
        if body:
            chunks.append({
                "id": f"report:{i}",
                "text": body,
                "metadata": {"type": "report", "section": heading},
            })
    return chunks


def _report_chunks(report_path: str) -> List[dict]:
    """ Tách final_report.md thành các chunk theo heading để dễ dàng sử dụng sau này"""
    p = Path(report_path)
    if not p.exists():
        return []
    return _split_report_by_heading(p.read_text(encoding="utf-8"))


def _meta_note_chunks(memory: ContextMemory) -> List[dict]:
    """ Mỗi meta_note trong memory thành 1 chunk để dễ dàng truy suất và sử dụng để đưa vào VectorStore để truy suất bằng RAG"""
    return [
        {
            "id": f"meta:{i}",
            "text": f"[Mẫu hình phản biện {i + 1}] {note}",
            "metadata": {"type": "meta_note"},
        }
        for i, note in enumerate(memory.meta_review_notes)
    ]


def _feedback_chunks(memory: ContextMemory) -> List[dict]:
    """ Mỗi agent_feedback trong memmory thành 1 chunk để dễ dàng truy suất và sử dụng để đưa vào VectorStore để truy suất bằng RAG"""
    chunks = []
    for agent, notes in memory.agent_feedback.items():
        if not notes:
            continue
        joined = "\n".join(f"- {n}" for n in notes)
        chunks.append({
            "id": f"feedback:{agent}",
            "text": f"[Phản hồi cho {agent}]\n{joined}",
            "metadata": {"type": "feedback", "agent": agent},
        })
    return chunks


def build_chunks(memory: ContextMemory, report_path: str = "output/final_report.md") -> List[dict]:
    """Trả list chunk đầy đủ (hypotheses + report + meta notes + feedback)."""
    chunks = []
    chunks += _hypothesis_chunks(memory)
    chunks += _report_chunks(report_path)
    chunks += _meta_note_chunks(memory)
    chunks += _feedback_chunks(memory)
    return chunks


def build_index(memory: ContextMemory, store_path: str,
                 report_path: str = "output/final_report.md",
                 model_name: str = "paraphrase-multilingual-MiniLM-L12-v2") -> VectorStore:
    """
    Build toàn bộ index và lưu vào store_path. Trả VectorStore đã nạp sẵn.
    Có nghĩa là các chunk được tạo sẽ được embed và lưu vào VectorStore để có thể truy suất bằng RAG.
    """
    chunks = build_chunks(memory, report_path=report_path)
    store = VectorStore(model_name=model_name)
    store.build(chunks, store_path)
    return store
