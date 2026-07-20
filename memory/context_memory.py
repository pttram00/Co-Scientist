"""Bộ nhớ ngữ cảnh dùng chung: hypothesis pool, đồ thị proximity, lịch sử
đấu trong tournament, ghi chú meta-review. Mọi agent đọc/ghi qua đây thay vì
truyền dữ liệu trực tiếp cho nhau -> giảm coupling giữa các agent.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from models.hypothesis import Hypothesis, HypothesisStatus, MatchResult


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

    # ---------- Hypothesis pool ----------
    def add_hypothesis(self, h: Hypothesis) -> None:
        self.hypotheses[h.id] = h

    def get_active_hypotheses(self) -> List[Hypothesis]:
        return [h for h in self.hypotheses.values() if h.status == HypothesisStatus.ACTIVE]

    def get_top_k(self, k: int) -> List[Hypothesis]:
        active = self.get_active_hypotheses()
        return sorted(active, key=lambda h: h.elo_rating, reverse=True)[:k]

    def mark_status(self, hypothesis_id: str, status: HypothesisStatus) -> None:
        if hypothesis_id in self.hypotheses:
            self.hypotheses[hypothesis_id].status = status

    # ---------- Proximity ----------
    def set_proximity(self, id_a: str, id_b: str, similarity: float) -> None:
        self.proximity_graph.setdefault(id_a, []).append((id_b, similarity))
        self.proximity_graph.setdefault(id_b, []).append((id_a, similarity))

    def neighbors(self, hypothesis_id: str, min_similarity: float = 0.0) -> List[Tuple[str, float]]:
        neigh = self.proximity_graph.get(hypothesis_id, [])
        return sorted([n for n in neigh if n[1] >= min_similarity], key=lambda x: x[1], reverse=True)

    # ---------- Tournament ----------
    def record_match(self, result: MatchResult) -> None:
        self.match_history.append(result)

    # ---------- Meta-review feedback ----------
    def add_meta_note(self, note: str) -> None:
        self.meta_review_notes.append(note)

    def add_agent_feedback(self, agent_name: str, note: str) -> None:
        self.agent_feedback.setdefault(agent_name, []).append(note)

    def get_agent_feedback(self, agent_name: str) -> List[str]:
        return self.agent_feedback.get(agent_name, [])

    # ---------- Persist ----------
    def to_dict(self) -> dict:
        return {
            "research_goal": self.research_goal,
            "constraints": self.constraints,
            "iteration": self.iteration,
            "hypotheses": {hid: h.to_dict() for hid, h in self.hypotheses.items()},
            "proximity_graph": self.proximity_graph,
            "match_history": [m.__dict__ for m in self.match_history],
            "meta_review_notes": self.meta_review_notes,
            "agent_feedback": self.agent_feedback,
        }

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)

    @staticmethod
    def load(path: str) -> "ContextMemory":
        with open(path, "r", encoding="utf-8") as f:
            d = json.load(f)
        mem = ContextMemory(d["research_goal"], d.get("constraints", ""))
        mem.iteration = d.get("iteration", 0)
        mem.hypotheses = {hid: Hypothesis.from_dict(h) for hid, h in d["hypotheses"].items()}
        mem.proximity_graph = {k: [tuple(x) for x in v] for k, v in d.get("proximity_graph", {}).items()}
        mem.match_history = [MatchResult(**m) for m in d.get("match_history", [])]
        mem.meta_review_notes = d.get("meta_review_notes", [])
        mem.agent_feedback = d.get("agent_feedback", {})
        return mem
