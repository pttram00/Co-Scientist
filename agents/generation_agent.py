"""Pha 1 — GenerationAgent: sinh giả thuyết khoa học mới CÓ grounding.

Mục tiêu:
    Trước khi sinh giả thuyết, tra cứu bài báo liên quan (arXiv + Semantic
    Scholar + OpenAlex, song song) để làm grounding bằng bằng chứng thật, sau
    đó mới đề xuất. Paper được cache vào memory.papers để tái dùng ở các
    iteration sau (không retrieve lại).

Luồng tổng quan:
    GenerationAgent.run(n)
    |___ (chỉ khi memory.papers chưa đủ: < min_papers_for_grounding)
    |     |___ query_expansion(): research_goal+constraints -> 3-5 query  (LLM, 1 call)
    |     |___ retriever.search(queries)       -- 3 nguồn song song, gộp+dedup+rank theo citation
    |     |___ memory.add_paper(pool)          -- cache
    |___ chọn 5-8 paper theo từng strategy     (LLM, 1 call/strategy; 2 hypothesis cùng strategy chia sẻ)
    |___ for i in range(n): _generate_one(strategy=strategies[i % 3], papers_for_strategy)
            user = research_goal + constraints + grounding_block(papers)
                 + STRATEGY_INSTRUCTIONS[strategy] + JSON_SCHEMA_HINT + feedback_block()
        song song qua asyncio.gather(return_exceptions=True) -> accept partial (KHÔNG retry)
    |___ memory.add_hypothesis(new)
    |___ return new

Quyết định:
    - 1/6 hypothesis raise -> bỏ cái lỗi, giữ phần thành công (partial), không retry.
    - Retrieval 1-nguồn fail -> bỏ nguồn đó, dùng phần còn lại; rỗng -> fallback sinh không grounding.
    - Query expansion fail -> fallback 1 query thô (= research_goal).
"""
from __future__ import annotations

import asyncio
from typing import Dict, List

from agents.base_agent import BaseAgent
from config import RetrieverConfig
from models.hypothesis import GenerationStrategy, Hypothesis
from models.paper import Paper
from retrieval.retriever import Retriever

SYSTEM_PROMPT = """Bạn là một nhà khoa học giàu kinh nghiệm, đóng vai
GenerationAgent trong một hệ thống multi-agent hỗ trợ nghiên cứu khoa học.
Nhiệm vụ của bạn là đề xuất giả thuyết khoa học MỚI, cụ thể, có thể kiểm
chứng được, bám sát mục tiêu nghiên cứu và các ràng buộc được cung cấp.
Nếu có phần "Grounding từ bài báo", hãy dựa giả thuyết vào cơ sở/bằng chứng
trong các bài báo đó (không bịa số liệu). Luôn trả lời bằng JSON theo đúng schema."""

# Mapping chiến lược -> hướng dẫn chi tiết cho LLM. Đây là cơ chế chính giúp hệ
# thống không sinh ra n giả thuyết giống nhau: cùng goal + constraints nhưng tiếp
# cận qua 3 lăng kính khác nhau.
STRATEGY_INSTRUCTIONS = {
    GenerationStrategy.LITERATURE_GROUNDED: (
        "Chiến lược: literature-grounded. Hãy dựa trên các cơ chế/lý thuyết "
        "đã được biết đến trong lĩnh vực liên quan (nếu có phần grounding, dùng "
        "chính các bài báo được nêu), đề xuất một hướng mở rộng hợp lý nhưng chưa "
        "được kiểm chứng."
    ),
    GenerationStrategy.SELF_DEBATE: (
        "Chiến lược: self-debate. Hãy mô phỏng một cuộc tranh luận ngắn giữa "
        "2 chuyên gia có quan điểm khác nhau về mục tiêu nghiên cứu, sau đó "
        "tổng hợp thành MỘT giả thuyết dung hoà được các phản biện mạnh nhất."
    ),
    GenerationStrategy.ASSUMPTION_ANALYSIS: (
        "Chiến lược: assumption analysis. Hãy liệt kê các giả định ngầm mà "
        "lĩnh vực này thường mặc nhiên chấp nhận, chọn MỘT giả định đáng ngờ "
        "nhất, và đề xuất giả thuyết dựa trên việc đảo ngược/nới lỏng giả định đó."
    ),
}

JSON_SCHEMA_HINT = """Schema JSON trả về (1 object):
{
  "content": "<phát biểu giả thuyết, 1-3 câu, cụ thể và kiểm chứng được>",
  "rationale": "<lý do / cơ chế đề xuất, 3-6 câu>",
  "suggested_experiment": "<thí nghiệm/phân tích đề xuất để kiểm chứng, 1-3 câu>"
}"""

QUERY_EXPANSION_SCHEMA = """Schema JSON trả về:
{
  "queries": ["<query 3-7 từ, đa góc nhìn>", "...", "..."]
}"""

SELECT_PAPERS_SCHEMA = """Schema JSON trả về:
{
  "ids": ["<list id các paper được chọn, đúng 5-8 id>"]
}"""


class GenerationAgent(BaseAgent):
    name = "generation_agent"

    def __init__(self, llm, memory, retriever: Retriever, config: RetrieverConfig):
        # Gọi thủ công super().__init__ để truyền thêm retriever + config
        # (BaseAgent chỉ nhận llm + memory, nên khởi tạo retriever tại đây).
        super().__init__(llm, memory)
        self.retriever = retriever
        self.config = config

    # ------------------------------------------------ API công khai

    async def run(self, n_hypotheses: int = 6) -> List[Hypothesis]:
        """Sinh n_hypotheses giả thuyết có grounding. Partial: nếu 1 vài hypothesis
        raise, vẫn trả phần thành công (không retry, không crash pha 1)."""
        pool = await self._ensure_papers()

        # 3 chiến lược luân phiên cho n hypothesis. Với n=6 -> mỗi strategy 2 lần.
        # 2 hypothesis cùng strategy DÙNG CHUNG tập paper đã chọn -> tiết kiệm 1 LLM
        # call chọn paper và đảm bảo cùng grounding cho cùng strategy.
        strategies = list(STRATEGY_INSTRUCTIONS.keys())
        strategies_used = [strategies[i % len(strategies)] for i in range(n_hypotheses)]
        distinct = []
        for s in strategies_used:
            if s not in distinct:
                distinct.append(s)

        papers_by_strategy: Dict[GenerationStrategy, List[Paper]] = {}
        # Chọn paper song song cho các strategy distinct (≤3 LLM call).
        select_tasks = [self._select_papers_llm(s, pool) for s in distinct]
        select_results = await asyncio.gather(*select_tasks, return_exceptions=True)
        for s, res in zip(distinct, select_results):
            if isinstance(res, Exception) or not res:
                papers_by_strategy[s] = []
            else:
                papers_by_strategy[s] = res

        # Sinh hypothesis song song. return_exceptions -> 1 lỗi không rớt cả mảng.
        gen_tasks = [
            self._generate_one(strategies_used[i], papers_by_strategy[strategies_used[i]])
            for i in range(n_hypotheses)
        ]
        results = await asyncio.gather(*gen_tasks, return_exceptions=True)

        new_hypotheses: List[Hypothesis] = []
        for idx, r in enumerate(results):
            if isinstance(r, Exception):
                # Partial: log và bỏ qua hypothesis lỗi — vẫn giữ phần thành công.
                print(f"[GenerationAgent] hypothesis #{idx} lỗi, bỏ qua: {r}")
                continue
            new_hypotheses.append(r)

        for h in new_hypotheses:
            self.memory.add_hypothesis(h)
        return new_hypotheses

    # ------------------------------------------ Bước 1-4: đảm bảo papers

    async def _ensure_papers(self) -> List[Paper]:
        """Trả pool paper để grounding. Nếu memory.papers đủ -> dùng cache, không
        retrieve lại. Nếu chưa đủ -> query expansion + retrieval + cache."""
        cached = self.memory.get_papers()
        if len(cached) >= self.config.min_papers_for_grounding:
            return cached

        queries = await self._expand_queries()
        pool = await self.retriever.search(queries)
        if pool:
            self.memory.add_paper(pool)
        return pool  # Có thể rỗng -> _generate_one sẽ sinh không grounding (fallback)

    # -------------------------------------------------- Bước 2: query expansion

    async def _expand_queries(self) -> List[str]:
        """1 LLM call: từ research_goal + constraints sinh 3-5 query đa góc nhìn.
        Fail -> fallback 1 query thô = research_goal (không raise)."""
        system = (
            "Bạn là trợ lí nghiên cứu. Sinh các câu truy vấn (query) tìm kiếm "
            "bài báo khoa học liên quan, đa góc nhìn (khía cạnh phương pháp, "
            "khía cạnh cơ chế, khía cạnh ứng dụng...). Mỗi query 3-7 từ."
        )
        user = (
            f"Mục tiêu nghiên cứu: {self.memory.research_goal}\n"
            f"Ràng buộc/bối cảnh: {self.memory.constraints or '(không có)'}\n\n"
            f"Sinh 3-5 query.\n{QUERY_EXPANSION_SCHEMA}"
        )
        try:
            data = await self.llm.complete_json(system, user)
            queries = data.get("queries") or []
            queries = [q.strip() for q in queries if q and q.strip()]
            if queries:
                return queries[:5]
        except Exception as e:
            print(f"[GenerationAgent] query expansion lỗi, dùng goal thô: {e}")
        # Fallback: 1 query thô (không raise).
        return [self.memory.research_goal]

    # ------------------------------------ Bước 6.0: chọn paper theo strategy

    async def _select_papers_llm(self, strategy: GenerationStrategy, pool: List[Paper]) -> List[Paper]:
        """1 LLM call: chọn 5-8 paper từ pool phù hợp cho strategy này.
        Fail -> fallback top citation của pool. Validate id phải thuộc pool."""
        k = self.config.papers_per_strategy
        if not pool:
            return []
        if len(pool) <= k:
            # Pool nhỏ hơn k -> dùng hết, không cần gọi LLM chọn (tiết kiệm cost).
            return pool

        catalog = "\n".join(
            f"- id={p.id} | citation={p.citations} | year={p.year or '?'} | title={p.title}"
            for p in pool
        )
        system = (
            "Bạn là trợ lí chọn lọc tài liệu. Từ danh sách bài báo, chọn những bài "
            "phù hợp nhất để làm grounding (cơ sở lý thuyết/bằng chứng) cho việc đề "
            "xuất giả thuyết theo một chiến lược nhất định."
        )
        strategy_hint = {
            GenerationStrategy.LITERATURE_GROUNDED: "ưu tiên paper citation cao, cơ chế/lý thuyết chặt chẽ.",
            GenerationStrategy.SELF_DEBATE: "ưu tiên đa dạng quan điểm (có thể cả paper disagreement).",
            GenerationStrategy.ASSUMPTION_ANALYSIS: "ưu tiên paper lâu đời/nhiều trích dẫn thể hiện giả định ngầm.",
        }.get(strategy, "ưu tiên citation cao.")
        user = (
            f"Chiến lược cần grounding: {strategy.value} — {strategy_hint}\n\n"
            f"Danh sách paper:\n{catalog}\n\n"
            f"Chọn {k} paper phù hợp nhất. {SELECT_PAPERS_SCHEMA}"
        )
        try:
            data = await self.llm.complete_json(system, user)
            ids = data.get("ids") or []
        except Exception as e:
            print(f"[GenerationAgent] _select_papers_llm lỗi, dùng top citation: {e}")
            ids = []

        by_id = {p.id: p for p in pool}
        selected: List[Paper] = []
        for pid in ids:
            if pid in by_id and by_id[pid] not in selected:
                selected.append(by_id[pid])
            if len(selected) >= k:
                break
        # Fallback / fill: nếu LLM chọn < k hợp lệ, bổ sung top citation còn thiếu.
        if len(selected) < k:
            remaining = [p for p in pool if p not in selected]
            remaining.sort(key=lambda p: p.citations, reverse=True)
            selected.extend(remaining[: k - len(selected)])
        return selected[:k]

    # ----------------------------------------------------- Bước 7: generate

    async def _generate_one(self, strategy: GenerationStrategy, papers: List[Paper]) -> Hypothesis:
        """Sinh 1 hypothesis. papers có thể rỗng -> fallback sinh không grounding."""
        user = (
            f"Mục tiêu nghiên cứu: {self.memory.research_goal}\n"
            f"Ràng buộc/bối cảnh: {self.memory.constraints or '(không có)'}\n\n"
            f"{self._grounding_block(papers)}\n\n"
            f"{STRATEGY_INSTRUCTIONS[strategy]}\n\n{JSON_SCHEMA_HINT}"
            f"{self.feedback_block()}"
        )
        data = await self.llm.complete_json(SYSTEM_PROMPT, user)
        return Hypothesis(
            content=data["content"],
            rationale=data["rationale"],
            research_goal=self.memory.research_goal,
            source_agent=self.name,
            strategy=strategy,
            suggested_experiment=data.get("suggested_experiment"),
        )

    # ------------------------------------------------------- helpers

    @staticmethod
    def _grounding_block(papers: List[Paper]) -> str:
        """Render block 'Grounding từ bài báo' vào prompt. Rỗng nếu không có paper."""
        if not papers:
            return ""  # Fallback: sinh không grounding.
        lines = []
        for p in papers:
            lines.append(f"[{p.id}] (citation={p.citations}, year={p.year or '?'}) {p.title}")
            if p.abstract:
                lines.append(p.abstract)
        return "Grounding từ bài báo (hãy dựa giả thuyết vào bằng chứng này):\n" + "\n".join(lines)
