"""Data model cho giả thuyết khoa học và các đánh giá gắn liền."""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import List, Optional


class HypothesisStatus(str, Enum):
    ACTIVE = "active"          # đang trong vòng lặp
    DUPLICATE = "duplicate"    # bị Proximity agent gắn cờ trùng lặp
    EVOLVED = "evolved"        # đã được Evolution thay thế bằng bản cải tiến
    ARCHIVED = "archived"      # bị loại khỏi vòng lặp


class GenerationStrategy(str, Enum):
    LITERATURE_GROUNDED = "literature_grounded"
    SELF_DEBATE = "self_debate"
    ASSUMPTION_ANALYSIS = "assumption_analysis"
    EVOLUTION_COMBINE = "evolution_combine"
    EVOLUTION_SIMPLIFY = "evolution_simplify"
    EVOLUTION_ANALOGY = "evolution_analogy"


@dataclass
class Review:
    reviewer: str                     # tên agent thực hiện review
    review_type: str                  # "initial" | "full" | "simulation" | "tournament"
    correctness: float                # 0-10
    novelty: float                    # 0-10
    feasibility: float                # 0-10
    comments: str
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


@dataclass
class MatchResult:
    hypothesis_a_id: str
    hypothesis_b_id: str
    winner_id: str
    rationale: str
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


@dataclass
class Hypothesis:
    content: str                                  # phát biểu giả thuyết
    rationale: str                                 # lý do / cơ chế đề xuất
    research_goal: str
    source_agent: str
    strategy: GenerationStrategy
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    parent_ids: List[str] = field(default_factory=list)
    reviews: List[Review] = field(default_factory=list)
    elo_rating: float = 1200.0
    matches_played: int = 0
    status: HypothesisStatus = HypothesisStatus.ACTIVE
    suggested_experiment: Optional[str] = None
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def average_score(self, key: str) -> float:
        vals = [getattr(r, key) for r in self.reviews]
        return sum(vals) / len(vals) if vals else 0.0

    def to_dict(self) -> dict:
        d = self.__dict__.copy()
        d["strategy"] = self.strategy.value
        d["status"] = self.status.value
        d["reviews"] = [r.__dict__ for r in self.reviews]
        return d

    @staticmethod
    def from_dict(d: dict) -> "Hypothesis":
        d = dict(d)
        d["strategy"] = GenerationStrategy(d["strategy"])
        d["status"] = HypothesisStatus(d["status"])
        d["reviews"] = [Review(**r) for r in d.get("reviews", [])]
        return Hypothesis(**d)
