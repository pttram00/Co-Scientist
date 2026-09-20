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
            # Nhánh này chấm tương đồng bằng LLM, schema trả {"similarity": 0-1}.
            self._count("proximity")
            a = re.search(r"Giả thuyết A: (.*)", user)
            b = re.search(r"Giả thuyết B: (.*)", user)
            ta, tb = _core(a.group(1) if a else ""), _core(b.group(1) if b else "")
            if ta and ta == tb:
                sim = 0.97                      # trùng y hệt sau khi bỏ nhãn [Đơn giản hoá]...
            else:
                # Cùng cơ chế/đối tượng -> tương đồng cao, để demo được lọc trùng lặp.
                shared = len(set(ta.split()) & set(tb.split()))
                sim = min(0.95, 0.25 + 0.06 * shared)
            return {"similarity": round(sim, 2), "reason": "(mô phỏng)"}
        if "ReflectionAgent" in system:
            # Nhánh này chỉ có MỘT loại review (không tách initial/full).
            self._count("reflection")
            return {"simulation_notes": "(mô phỏng) Bước 2 của cơ chế có thể bị bù trừ bởi con đường khác.",
                    "correctness": self.rng.randint(4, 9), "novelty": self.rng.randint(2, 8),
                    "feasibility": self.rng.randint(4, 9),
                    "comments": "(mô phỏng) Nên bổ sung nhóm đối chứng."}
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
                                 "ranking_agent": "", "proximity_agent": "", "evolution_agent": ""}}
        # ---- Các prompt của chatbot (góp ý + phân loại ý định) ----
        if "chấm điểm theo 3 tiêu chí" in system:
            # Chấm điểm góp ý của người dùng: đoán thô theo từ khoá chê/khen để
            # chế độ mô phỏng vẫn thể hiện được ảnh hưởng của góp ý lên điểm số.
            self._count("user_comment")
            low = user.lower()
            neg = any(w in low for w in ("chưa", "không", "sai", "yếu", "kém", "trùng",
                                         "nhàm", "khó", "thiếu", "lỗ hổng"))
            base = 3.0 if neg else 7.5
            return {"correctness": base, "novelty": base, "feasibility": base}
        if "phân loại ý định" in system:
            self._count("intent")
            return _fake_intent(user)
        raise AssertionError("Prompt không nhận diện được trong chế độ mô phỏng")

    async def complete_json_tool(self, system: str, user: str, tool: dict, **kw):
        """Nhánh theo tên tool (Cách 2 — structured output). Tương đương
        `complete_json` nhưng trả đúng schema của tool thay vì parse system prompt.
        Giữ để test/sim chạy được sau khi các agent chuyển sang `complete_json_tool`."""
        await asyncio.sleep(self.rng.uniform(self.delay * 0.5, self.delay * 1.5))
        name = tool.get("name", "")
        if name == "record_search_queries":
            self._count("generation")
            return {"queries": ["neuronal senescence mechanisms", "brain aging interventions"]}
        if name == "record_selected_paper_ids":
            self._count("generation")
            return {"ids": []}
        if name == "record_hypothesis":
            self._count("generation")
            # EvolutionAgent cũng dùng tool này; nhãn chiến lược lấy từ user prompt
            # (giống `complete_json` cũ) để kết quả gần nhau và lọc trùng vẫn hoạt động.
            label = "Kết hợp" if "COMBINE" in user else "Đơn giản hoá" if "SIMPLIFY" in user else ""
            prefix = f"[{label}] " if label else ""
            return self._hypothesis(prefix=prefix)
        if name == "record_hypothesis_review":
            self._count("reflection")
            return {"simulation_notes": "(mô phỏng) Bước 2 của cơ chế có thể bị bù trừ bởi con đường khác.",
                    "correctness": self.rng.randint(4, 9), "novelty": self.rng.randint(2, 8),
                    "feasibility": self.rng.randint(4, 9),
                    "comments": "(mô phỏng) Nên bổ sung nhóm đối chứng."}
        if name == "record_match_verdict":
            self._count("ranking")
            w = self.rng.choice(["A", "B"])
            return {"winner": w, "rationale": f"(mô phỏng) Giả thuyết {w} có cơ chế cụ thể và dễ kiểm chứng hơn."}
        if name == "record_review_feedback":
            self._count("meta_review")
            return {"patterns": ["(mô phỏng) Nhiều giả thuyết thiếu nhóm đối chứng."],
                    "feedback": {"generation_agent": "(mô phỏng) Nêu rõ nhóm đối chứng.",
                                 "reflection_agent": "(mô phỏng) Xét thêm chi phí thí nghiệm.",
                                 "ranking_agent": "", "proximity_agent": "", "evolution_agent": ""}}
        raise AssertionError(f"Tool '{name}' không nhận diện được trong chế độ mô phỏng")

    async def complete(self, system: str, user: str, **kw) -> str:
        await asyncio.sleep(self.delay)
        # ReAct RAG của chatbot: trả thẳng Final Answer dựa trên Observation đã có,
        # hoặc gọi recall một lần nếu chưa tra gì.
        if "ReAct" in system:
            self._count("chat")
            if "Observation:" in user:
                m = re.search(r"Observation:\s*(.*)", user, re.S)
                seen = (m.group(1) if m else "").strip()
                ids = re.findall(r"\[(hyp:[^\]]+|report:\d+|meta:\d+|feedback:[^\]]+)\]", seen)
                ref = ", ".join(f"[{i}]" for i in dict.fromkeys(ids)) or "(không có đoạn nào)"
                return ("Thought: Đã có đoạn tài liệu, đủ để trả lời.\n"
                        f"Final Answer: (mô phỏng) Dựa trên các đoạn {ref} trong kho dữ liệu "
                        "của phiên này. Chạy ở chế độ Thật để nhận câu trả lời do LLM viết.")
            q = user.splitlines()[0][:80] if user.strip() else "kết quả"
            return f'Thought: Cần tra kho tài liệu.\nAction: recall("{q}")'

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


_AGENT_WORDS = {
    "generation_agent": ("generation", "sinh giả thuyết"),
    "proximity_agent": ("proximity", "trùng", "giống nhau"),
    "reflection_agent": ("reflection", "phản biện", "chấm điểm"),
    "ranking_agent": ("ranking", "xếp hạng", "elo"),
    "evolution_agent": ("evolution", "tiến hoá", "cải tiến"),
    "meta_review_agent": ("meta", "báo cáo"),
}


def _fake_intent(user: str) -> dict:
    """Phân loại ý định giả: bám từ khoá, đủ để demo luồng góp ý → chạy lại."""
    low = user.lower()
    m = re.search(r"Câu của user: (.*)", user)
    said = (m.group(1) if m else user).strip()
    said_low = said.lower()

    if any(w in said_low for w in ("chạy lại", "chạy thêm", "tạo lại", "sinh lại", "cập nhật báo cáo")):
        return {"kind": "resume_newfinal"}
    if any(w in said_low for w in ("nghiên cứu về", "đổi chủ đề", "chủ đề mới")):
        return {"kind": "new_topic", "research_goal": said}
    if said.strip().endswith("?") or said_low.startswith(("bao nhiêu", "có mấy", "cái nào", "là gì")):
        return {"kind": "question"}

    # Góp ý: tìm id giả thuyết được nhắc, rồi tới tên agent.
    ids = re.findall(r"\b([0-9a-f]{8})\b", said)
    if ids:
        return {"kind": "review", "target_hypothesis_id": ids[0], "comment_text": said}
    for agent, words in _AGENT_WORDS.items():
        if any(w in said_low for w in words):
            return {"kind": "review", "target_agent": agent, "comment_text": said}
    return {"kind": "review", "comment_text": said}


def install(orchestrator, seed: int = 7, delay: float = 0.15) -> FakeLLM:
    """Gắn LLM / retriever / encoder giả vào một Orchestrator đã khởi tạo."""
    fake = FakeLLM(seed=seed, delay=delay)
    install_llm(orchestrator.llm, fake)
    orchestrator.retriever.search = fake_search
    # ProximityAgent ở nhánh này chấm tương đồng bằng LLM (đã có LLM giả ở trên),
    # không dùng encoder. Chỉ vá khi agent thực sự có encoder.
    if hasattr(orchestrator.proximity_agent, "encoder"):
        orchestrator.proximity_agent.encoder = fake_encoder
    return fake


def install_llm(llm, fake: Optional[FakeLLM] = None, seed: int = 7,
                delay: float = 0.15) -> FakeLLM:
    """Gắn LLM giả vào một LLMClient rời (chatbot dùng client riêng, không qua
    Orchestrator). Trả FakeLLM để tái dùng cho các client khác trong cùng phiên."""
    fake = fake or FakeLLM(seed=seed, delay=delay)
    llm.complete_json = fake.complete_json
    llm.complete_json_tool = fake.complete_json_tool
    llm.complete = fake.complete
    return fake
