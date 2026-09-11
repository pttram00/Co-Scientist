"""Retriever: tra cứu paper từ 3 nguồn song song, gộp + dedup + rank.

Được GenerationAgent dùng làm grounding. Theo đặc tả:
 - 3 nguồn: arXiv (XML/Atom), Semantic Scholar (JSON), OpenAlex (JSON) — song song.
 - Mỗi nguồn chạy độc lập, try/except: 1 nguồn sập vẫn dùng phần còn lại (partial).
 - Fusion: dedup theo title chuẩn hoá -> giữ bản citation cao; rank citation desc.
 - Cutoff top pool_size.

KHÔNG raise khi 1 nguồn fail — trả phần còn lại; toàn fail -> trả list rỗng
(GenerationAgent tự fallback sinh không grounding).
"""
from __future__ import annotations

import asyncio
import logging
import re
import xml.etree.ElementTree as ET
from typing import List, Optional

import httpx

from config import RetrieverConfig
from models.paper import Paper

logger = logging.getLogger("retriever")

# arXiv Atom feed namespace — phải dùng để parse title/summary/published.
_ARXIV_NS = {"atom": "http://www.w3.org/2005/Atom"}


def _normalize_title(title: str) -> str:
    """Chuẩn hoá title để dedup: lowercase + bỏ dấu câu/whitespace thừa.

    Hai paper có thể post trên arXiv lẫn S2/OpenAlex với title gần giống nhau;
    so sánh dạng chuẩn này là đủ để coi là cùng một bài.
    """
    t = title.lower().strip()
    t = re.sub(r"[^\w\s]", " ", t)      # dấu câu -> space
    t = re.sub(r"\s+", " ", t).strip()
    return t


class Retriever:
    def __init__(self, config: RetrieverConfig):
        self.config = config
        headers = {"User-Agent": config.user_agent}
        # httpx.AsyncClient dùng chung cho cả 3 nguồn (reuse connection pool).
        self._client = httpx.AsyncClient(
            timeout=config.request_timeout,
            headers=headers,
            follow_redirects=True,
        )

    async def aclose(self) -> None:
        await self._client.aclose()

    # ---------------------------------------------------------------- API

    async def search(self, queries: List[str]) -> List[Paper]:
        """Tra cứu song song 3 nguồn cho từng query, gộp + dedup + rank.

        Trả List[Paper] đã fusion, số lượng <= pool_size. Không bao giờ raise
        do lỗi nguồn — chỉ log warning và bỏ nguồn đó.
        """
        # Mỗi query đánh vào cả 3 nguồn song song. Flatten tất cả (query×nguồn)
        # thành 1 đợt gather, return_exceptions=True để 1 lỗi không làm rớt cả mảng.
        coros = []
        for q in queries:
            for source_fn in (self._search_arxiv, self._search_semantic_scholar, self._search_openalex):
                coros.append(source_fn(q, self.config.k_per_source))

        raw_batches = await asyncio.gather(*coros, return_exceptions=True)

        flat: List[Paper] = []
        for r in raw_batches:
            if isinstance(r, Exception):
                # Partial: ghi log rồi tiếp tục — vẫn dùng kết quả của các nguồn khác.
                logger.warning("Một nguồn retrieval lỗi, bỏ qua: %s", r)
                continue
            if r:
                flat.extend(r)

        if not flat:
            logger.warning("Retrieval trả về 0 paper (tất cả nguồn rỗng/lỗi). GenerationAgent sẽ fallback sinh không grounding.")
            return []

        fused = self._fuse(flat)
        logger.info("Retriever: %d raw -> %d sau fusion (pool_size=%d)",
                    len(flat), len(fused), len(fused))
        return fused

    # ---------------------------------------------------------- Fusion

    def _fuse(self, papers: List[Paper]) -> List[Paper]:
        """Dedup theo title chuẩn hoá (giữ bản citation cao) + rank theo citation desc.

        cutoff top pool_size. """
        by_key: dict[str, Paper] = {}
        for p in papers:
            key = _normalize_title(p.title)
            if not key:
                continue
            existing = by_key.get(key)
            if existing is None or p.citations > existing.citations:
                # Giữ bản có citation cao hơn (metadata của S2/OpenAlex đầy đủ hơn arXiv).
                by_key[key] = p

        uniq = list(by_key.values())
        # Rank: citation desc. (Relevance đã ở mức query-level khi API sort; tới đây
        # ta ưu tiên paper có nhiều trích dẫn — đúng "ưu tiên lượt trích dẫn cao" của đặc tả.)
        uniq.sort(key=lambda p: p.citations, reverse=True)
        return uniq[: self.config.pool_size]

    # ----------------------------------------------------- Nguồn: arXiv

    async def _search_arxiv(self, query: str, k: int) -> List[Paper]:
        """arXiv API trả Atom XML. Parse title/summary/published.

        arXiv không có citation count -> citations=0 (!!rank fusion sẽ ưu tiên S2/OpenAlex).
        """
        params = {
            "search_query": f"all:{query}",
            "start": 0,
            "max_results": k,
            "sortBy": "relevance",
        }
        try:
            resp = await self._get("http://export.arxiv.org/api/query", params)
        except Exception as e:
            logger.error(
                "arXiv failed | query = %s | error = %s",
                query,
                e
            )
            raise

        if not resp:
            return []
        papers: List[Paper] = []
        try:
            root = ET.fromstring(resp)
        except ET.ParseError as e:
            logger.warning("arXiv: parse XML lỗi: %s", e)
            return []
        for entry in root.findall("atom:entry", _ARXIV_NS):
            id_el = entry.find("atom:id", _ARXIV_NS)
            title_el = entry.find("atom:title", _ARXIV_NS)
            summary_el = entry.find("atom:summary", _ARXIV_NS)
            pub = entry.find("atom:published", _ARXIV_NS)
            if id_el is None or title_el is None:
                continue
            # arXiv id có dạng http://arxiv.org/abs/2401.12345v1 -> lấy phần cuối.
            arxiv_id = id_el.text.strip().rsplit("/", 1)[-1]
            year: Optional[int] = None
            if pub is not None and pub.text:
                try:
                    year = int(pub.text[:4])
                except ValueError:
                    year = None
            papers.append(Paper(
                id=f"arxiv:{arxiv_id}",
                title=" ".join(title_el.text.split()),
                abstract=(summary_el.text.strip() if summary_el is not None and summary_el.text else ""),
                year=year,
                citations=0,
                source="arxiv",
            ))
        return papers

    # ----------------------------------------------- Nguồn: Semantic Scholar

    async def _search_semantic_scholar(self, query: str, k: int) -> List[Paper]:
        """S2 Graph API. Có key thì rate limit cao hơn; không key vẫn chạy được."""
        url = "https://api.semanticscholar.org/graph/v1/paper/search"
        params = {
            "query": query,
            "limit": k,
            "fields": "title,abstract,year,citationCount,externalIds",
        }
        headers = {}
        if self.config.semantic_scholar_api_key:
            headers["x-api-key"] = self.config.semantic_scholar_api_key
        try:
            resp = await self._get(url, params, headers=headers)
        except Exception as e:
            logger.error(
                "Semantic-Scholar failed | query = %s | error = %s", 
                query,
                e
            )
            raise

        if not resp:
            return []
        try:
            data = resp.json()
        except Exception as e:
            logger.warning("S2: parse JSON lỗi: %s", e)
            return []
        papers: List[Paper] = []
        for item in data.get("data", []) or []:
            paper_id = item.get("paperId")
            if not paper_id:
                continue
            title = item.get("title") or ""
            abstract = item.get("abstract") or ""
            if not title or not abstract:
                # Bỏ paper thiếu title hoặc abstract (không thể làm grounding được).
                continue
            papers.append(Paper(
                id=f"semantic_scholar:{paper_id}",
                title=title,
                abstract=abstract,
                year=item.get("year"),
                citations=int(item.get("citationCount") or 0),
                source="semantic_scholar",
            ))
        return papers

    # ----------------------------------------------------- Nguồn: OpenAlex

    async def _search_openalex(self, query: str, k: int) -> List[Paper]:
        """OpenAlex Works API. Abstract lưu dạng inverted index -> phục hồi thành text."""
        url = "https://api.openalex.org/works"
        params = {
            "search": query,
            "per-page": k,
        }
        if self.config.openalex_mailto:
            params["mailto"] = self.config.openalex_mailto
        try:
            resp = await self._get(url, params)
        except Exception as e:
            logger.error(
                "OpenAlex failed | query = %s | error = %s",
                query,
                e
            )
            raise
        if not resp:
            return []
        try:
            data = resp.json()
        except Exception as e:
            logger.warning("OpenAlex: parse JSON lỗi: %s", e)
            return []
        papers: List[Paper] = []
        for item in data.get("results", []) or []:
            oa_id = (item.get("id") or "").rsplit("/", 1)[-1]
            if not oa_id:
                continue
            title = item.get("display_name") or item.get("title") or ""
            abstract = _openalex_abstract(item.get("abstract_inverted_index"))
            if not title or not abstract:
                continue
            year = None
            py = item.get("publication_year")
            if isinstance(py, int):
                year = py
            # cited_by_count nằm ở top-level của work object.
            cit = int(item.get("cited_by_count") or 0)
            papers.append(Paper(
                id=f"openalex:{oa_id}",
                title=title,
                abstract=abstract,
                year=year,
                citations=cit,
                source="openalex",
            ))
        return papers

    # ------------------------------------------------------- HTTP helper

    async def _get(self, url: str, params: dict, headers: Optional[dict] = None) -> Optional[httpx.Response]:
        """GET có retry nhẹ (theo config.max_retries) cho transient timeout/5xx.

        Trả Response hoặc None (khi retry hết). KHÔNG raise -> partial fallback.
        """
        h = dict(self._client.headers)
        if headers:
            h.update(headers)
        last_err = None
        for attempt in range(self.config.max_retries + 1):
            try:
                r = await self._client.get(url, params=params, headers=h)
                if r.status_code >= 500:
                    last_err = RuntimeError(f"HTTP {r.status_code}")
                    await asyncio.sleep(min(2 ** attempt, 8))
                    continue
                return r
            except (httpx.TimeoutException, httpx.TransportError) as e:
                last_err = e
                await asyncio.sleep(min(2 ** attempt, 8))
        logger.warning("GET %s thất bại sau %d lần: %s", url, self.config.max_retries + 1, last_err)
        return None


def _openalex_abstract(inv: Optional[dict]) -> str:
    """OpenAlex lưu abstract dạng inverted index: {word: [pos1, pos2, ...]}.
    Phục hồi lại thành text theo vị trí."""
    if not inv:
        return ""
    # Tính độ dài tối đa để đặt word đúng vị trí.
    max_pos = 0
    for positions in inv.values():
        for p in positions:
            if p > max_pos:
                max_pos = p
    words = [""] * (max_pos + 1)
    for word, positions in inv.items():
        for p in positions:
            words[p] = word
    return " ".join(words).strip()
