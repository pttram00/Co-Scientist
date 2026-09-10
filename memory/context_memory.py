"""Bộ nhớ ngữ cảnh dùng chung: hypothesis pool, đồ thị proximity, lịch sử
đấu trong tournament, ghi chú meta-review. Mọi agent đọc/ghi qua đây thay vì
truyền dữ liệu trực tiếp cho nhau -> giảm coupling giữa các agent.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from models.hypothesis import Hypothesis, HypothesisStatus, MatchResult
from models.paper import Paper


class ContextMemory:
    def __init__(self, research_goal: str, constraints: Optional[str] = None):
        self.research_goal = research_goal
        self.constraints = constraints or ""
        self.hypotheses: Dict[str, Hypothesis] = {}
        # proximity_graph[id] = list of (other_id, similarity 0-1)
        self.proximity_graph: Dict[str, List[Tuple[str, float]]] = {}
        self.match_history: List[MatchResult] = []
        self.meta_review_notes: List[str] = []
        self.agent_feedback: Dict[str, List[str]] = {}  # feedback cho từng agent
        self.iteration: int = 0
        # Pool bài báo tra cứu làm grounding; GenerationAgent cache vào đây để
        # các iteration sau không phải retrieve lại (giảm lãng phí + ổn định ngữ cảnh).
        self.papers: Dict[str, Paper] = {}

    # ---------- Hypothesis pool ----------
    def add_hypothesis(self, h: Hypothesis) -> None:
        """ Thêm giả thuyết vào bộ nhớ. Nếu đã tồn tại, ghi đè lên bản cũ."""
        self.hypotheses[h.id] = h

    def get_active_hypotheses(self) -> List[Hypothesis]:
        """ Các giả thuyết được coi là còn hiệu lực sẽ được dùng trong các pha sau """
        return [h for h in self.hypotheses.values() if h.status == HypothesisStatus.ACTIVE]

    def get_top_k(self, k: int) -> List[Hypothesis]:
        """ Lấy k các giả thuyết có elo rating cao nhất trong bộ nhớ """
        active = self.get_active_hypotheses()
        return sorted(active, key=lambda h: h.elo_rating, reverse=True)[:k]

    def mark_status(self, hypothesis_id: str, status: HypothesisStatus) -> None:
        """ Cập nhật trạng thái cho một giả thuyết """
        if hypothesis_id in self.hypotheses:
            self.hypotheses[hypothesis_id].status = status

    # ---------- Paper pool (grounding) ----------
    def add_paper(self, papers: List[Paper]) -> None:
        """Thêm/ghi đè paper vào pool theo id. Retriever đã dedup rồi nên đây chủ yếu
        là cache để tái dùng giữa các iteration."""
        for p in papers:
            self.papers[p.id] = p

    def get_papers(self) -> List[Paper]:
        """ Lấy toàn bộ pool paper đã cache (chưa sắp xếp — gọi tự sort nếu cần). """
        return list(self.papers.values())

    # ---------- Proximity ----------
    def set_proximity(self, id_a: str, id_b: str, similarity: float) -> None:
        """ Được dùng để tạo một đồ thị giữa các giả thuyết để tìm các giả thuyết tương tự nhau """
        self.proximity_graph.setdefault(id_a, []).append((id_b, similarity))
        self.proximity_graph.setdefault(id_b, []).append((id_a, similarity))

    def neighbors(self, hypothesis_id: str, min_similarity: float = 0.0) -> List[Tuple[str, float]]:
        """ Lấy các giả thuyết tương tự với giả thuyết có id = hypothesis_id """
        neigh = self.proximity_graph.get(hypothesis_id, [])
        return sorted([n for n in neigh if n[1] >= min_similarity], key=lambda x: x[1], reverse=True)

    # ---------- Tournament ----------
    def record_match(self, result: MatchResult) -> None:
        """ Nơi lưu lại kết quả của các trận đấu giữa hai giả thuyết, về sau có thể dùng để ranking, ELO,..."""
        self.match_history.append(result)

    # ---------- Meta-review feedback ----------
    def add_meta_note(self, note: str) -> None:
        """ Mỗi note là một nhận xét của meta-reviewer về các giả thuyết trong bộ nhớ. """
        self.meta_review_notes.append(note)

    def add_agent_feedback(self, agent_name: str, note: str) -> None:
        """ Thêm nhận xét của một agent về các giả thuyết trong bộ nhớ."""
        self.agent_feedback.setdefault(agent_name, []).append(note)

    def get_agent_feedback(self, agent_name: str) -> List[str]:
        """ Lấy tất cả nhận xét của một agent về các giả thuyết trong bộ nhớ."""
        return self.agent_feedback.get(agent_name, [])

    # ---------- Persist ----------
    def to_dict(self) -> dict:
        """ Chuyển bộ nhớ sang dạng dict để có thể dễ dàng truyền qua JSON hoặc lưu vào file."""
        return {
            "research_goal": self.research_goal,
            "constraints": self.constraints,
            "iteration": self.iteration,
            "hypotheses": {hid: h.to_dict() for hid, h in self.hypotheses.items()},
            "proximity_graph": self.proximity_graph,
            "match_history": [m.__dict__ for m in self.match_history],
            "meta_review_notes": self.meta_review_notes,
            "agent_feedback": self.agent_feedback,
            "papers": {pid: p.to_dict() for pid, p in self.papers.items()},
        }

    def save(self, path: str) -> None:
        """ Lưu bộ nhớ được tạo ở dạng JSON vào file. """
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)

    @staticmethod
    def load(path: str) -> "ContextMemory":
        """
        Tải bộ nhớ để sử dụng từ file JSON đã lưu.
        Trong đó có:
        - research_goal: mực tiêu nghiên cứu
        - constraints: các ràng buộc nghiên cứu
        - iteration: số vòng lặp hiện tại 
        - hypotheses: danh sách các giả thuyết
        - proximity_graph: đồ thị proximity giữa các giả thuyết → proximity graph là một đồ thị với các nút là các giả thuyết còn các cạnh là độ tương đồng giữa các giả thuyết
        - match_history: lịch sử các trận đấu giữa cấc giả thuyết
        - meta_review_notes: ghi chú của meta-reviewer về các giả thuyết
        - agent_feedback: nhận xét của các agent về các giả thuyết
        - papers: pool bài báo tra cứu làm grounding (cache để tái dùng)
        """
        with open(path, "r", encoding="utf-8") as f:
            d = json.load(f)
        mem = ContextMemory(d["research_goal"], d.get("constraints", ""))
        mem.iteration = d.get("iteration", 0)
        mem.hypotheses = {hid: Hypothesis.from_dict(h) for hid, h in d["hypotheses"].items()}
        mem.proximity_graph = {k: [tuple(x) for x in v] for k, v in d.get("proximity_graph", {}).items()}
        mem.match_history = [MatchResult(**m) for m in d.get("match_history", [])]
        mem.meta_review_notes = d.get("meta_review_notes", [])
        mem.agent_feedback = d.get("agent_feedback", {})
        mem.papers = {pid: Paper.from_dict(p) for pid, p in d.get("papers", {}).items()}
        return mem
