"""Chế độ mô phỏng cho giao diện: thay LLM, tra cứu bài báo và embedding bằng bản giả.

Dùng để chạy thử giao diện và xem toàn bộ luồng 3 pha mà KHÔNG cần API key, không
gọi mạng và không cần tải model embedding. Nội dung giả thuyết sinh ra là giả, chỉ
để minh hoạ luồng chạy — không có giá trị khoa học.
"""
from __future__ import annotations

import asyncio
import hashlib
import math
import random
import re
from typing import List, Optional

from models.paper import Paper

MECHS = ["autophagy", "chức năng ty thể", "chuyển hoá NAD+", "tín hiệu IGF-1",
         "loại bỏ tế bào già", "sửa chữa DNA", "ức chế viêm thần kinh"]
TARGETS = ["neuron vỏ não", "vi đệm (microglia)", "tế bào hình sao", "neuron hồi hải mã"]
ACTIONS = ["Tăng cường", "Ức chế chọn lọc"]


def fake_encoder(texts: List[str]) -> List[List[float]]:
    """Encoder giả thay sentence-transformers: bag-of-words băm vào 256 chiều, chuẩn hoá L2."""
    out = []
    for t in texts:
        v = [0.0] * 256
        for tok in t.lower().split():
            v[int(hashlib.md5(tok.encode("utf-8")).hexdigest(), 16) % 256] += 1.0
        norm = math.sqrt(sum(x * x for x in v)) or 1.0
        out.append([x / norm for x in v])
    return out


async def fake_search(queries: List[str], k_per_source: Optional[int] = None,
                      pool_size: Optional[int] = None) -> List[Paper]:
    await asyncio.sleep(0.05)
    papers = [Paper(id=f"demo:{i}-{j}", title=f"[MÔ PHỎNG] Bài báo giả về “{q}” #{j}",
                    abstract="(mô phỏng) tóm tắt giả dùng để minh hoạ phần grounding.",
                    year=2020 + j, citations=12 * j)
              for i, q in enumerate(queries) for j in range(1, 4)]
    return papers[: pool_size or 20]


def _core(text: str) -> str:
    return re.sub(r"^\[[^\]]+\]\s*", "", text).strip().lower()


class FakeLLM:
    """Nhận diện agent qua system prompt rồi trả JSON đúng schema agent đó cần."""

    def __init__(self, seed: int = 7, delay: float = 0.15):
        self.rng = random.Random(seed)
        self.delay = delay
        self.calls = {}

    def _count(self, kind):
        self.calls[kind] = self.calls.get(kind, 0) + 1

    def _hypothesis(self, prefix=""):
        act, mech, target = (self.rng.choice(ACTIONS), self.rng.choice(MECHS), self.rng.choice(TARGETS))
        return {
            "content": f"{prefix}{act} {mech} ở {target} làm chậm quá trình lão hoá.",
            "rationale": f"(mô phỏng) {mech} suy giảm theo tuổi ở {target}; điều chỉnh có thể giảm tích tụ tổn thương.",
            "suggested_experiment": f"(mô phỏng) Đo dấu ấn lão hoá ở {target} nuôi cấy sau khi can thiệp {mech}.",
        }

    async def complete_json(self, system: str, user: str, **kw):
        await asyncio.sleep(self.rng.uniform(self.delay * 0.5, self.delay * 1.5))
        if "SafetyAgent" in system:
            self._count("safety")
            return {"safe": True, "reason": "(mô phỏng) Nghiên cứu thông thường, không có dấu hiệu nguy hiểm."}
        if "Sinh các câu truy vấn" in system:
            self._count("generation")
            return {"queries": ["neuronal senescence mechanisms", "brain aging interventions"]}
        if "chọn lọc tài liệu" in system:
            self._count("generation")
            return {"ids": []}
        if "GenerationAgent" in system:
            self._count("generation")
            return self._hypothesis()
        if "ProximityAgent" in system:
            self._count("proximity")
            a = re.search(r"Giả thuyết A: (.*)", user).group(1)
            b = re.search(r"Giả thuyết B: (.*)", user).group(1)
            return {"duplicate": _core(a) == _core(b), "reason": "(mô phỏng)"}
        if "INITIAL REVIEW" in system:
            self._count("reflection")
            ok = self.rng.random() > 0.12
            return {"correctness": self.rng.randint(5, 9), "novelty": self.rng.randint(5, 9),
                    "feasibility": self.rng.randint(4, 9), "pass": ok, "safe": True, "safety_concerns": "",
                    "search_queries": ["neuronal senescence"],
                    "comments": "(mô phỏng) Hợp lý, cần kiểm tra tính mới."
                    if ok else "(mô phỏng) Lập luận có lỗ hổng rõ ràng nên bị loại."}
        if "FULL REVIEW" in system:
            self._count("reflection")
            return {"known_aspects": "(mô phỏng) Vai trò chung của cơ chế này đã được báo cáo.",
                    "novelty_assessment": "(mô phỏng) Điểm mới nằm ở loại tế bào đích.",
                    "simulation_notes": "(mô phỏng) Bước 2 của cơ chế có thể bị bù trừ bởi con đường khác.",
                    "correctness": self.rng.randint(4, 9), "novelty": self.rng.randint(2, 8),
                    "feasibility": self.rng.randint(4, 9), "comments": "(mô phỏng) Nên bổ sung nhóm đối chứng."}
        if "RankingAgent" in system:
            self._count("ranking")
            w = self.rng.choice(["A", "B"])
            return {"winner": w, "rationale": f"(mô phỏng) Giả thuyết {w} có cơ chế cụ thể và dễ kiểm chứng hơn."}
        if "EvolutionAgent" in system:
            self._count("evolution")
            label = "Kết hợp" if "COMBINE" in user else "Đơn giản hoá" if "SIMPLIFY" in user else "Tương tự hoá"
            return self._hypothesis(prefix=f"[{label}] ")
        if "MetaReviewAgent" in system:
            self._count("meta_review")
            return {"patterns": ["(mô phỏng) Nhiều giả thuyết thiếu nhóm đối chứng."],
                    "feedback": {"generation_agent": "(mô phỏng) Nêu rõ nhóm đối chứng.",
                                 "reflection_agent": "(mô phỏng) Xét thêm chi phí thí nghiệm.",
                                 "ranking_agent": "", "proximity_agent": "", "evolution_agent": ""},
                    "research_overview": "(mô phỏng) Đã khám phá: ty thể, autophagy. Chưa khám phá: tương tác neuron–mạch máu.",
                    "safety_concerns": []}
        raise AssertionError("Prompt không nhận diện được trong chế độ mô phỏng")

    async def complete(self, system: str, user: str, **kw) -> str:
        await asyncio.sleep(self.delay)
        self._count("meta_review")
        m = re.search(r"vòng lặp:\n(.*?)\n\nCác mẫu hình", user, re.S)
        top = m.group(1) if m else "(không có giả thuyết nào)"
        goal = (re.search(r"Mục tiêu nghiên cứu: (.*)", user) or [None, "(không rõ)"])[1]
        return ("# Research Overview (CHẾ ĐỘ MÔ PHỎNG)\n\n"
                "> Báo cáo này do bản LLM giả sinh ra để minh hoạ giao diện. "
                "Nội dung không có giá trị khoa học.\n\n"
                "## Tổng quan nghiên cứu\n\n"
                f"Mục tiêu: {goal}\n\n"
                "## Các giả thuyết nổi bật\n\n"
                f"{top}\n\n"
                "## Kết luận & hướng tiếp theo\n\n"
                "Chạy lại ở chế độ thật (có API key trong `.env`) để nhận báo cáo do LLM viết.\n")


def install(orchestrator, seed: int = 7, delay: float = 0.15) -> FakeLLM:
    """Gắn LLM / retriever / encoder giả vào một Orchestrator đã khởi tạo."""
    fake = FakeLLM(seed=seed, delay=delay)
    orchestrator.llm.complete_json = fake.complete_json
    orchestrator.llm.complete = fake.complete
    orchestrator.retriever.search = fake_search
    orchestrator.proximity_agent.encoder = fake_encoder
    return fake
