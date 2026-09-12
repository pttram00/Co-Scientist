"""Vector store cho RAG: embed chunks (sentence-transformers) + cosine search.

Lưu/đọc index dạng JSON giản đơn ( đủ cho corpus nhỏ ~ vài chục chunk).
Embedding model mặc định: paraphrase-multilingual-MiniLM-L12-v2 — đa ngữ
(có tiếng Việt), nhẹ (~120MB), chạy CPU được.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional

import numpy as np


class VectorStore:
    def __init__(self, model_name: str = "paraphrase-multilingual-MiniLM-L12-v2"):
        self.model_name = model_name
        # Lazy-load: SentenceTransformer chỉ nạp khi cần embed (lần đầu chậm ~3s,
        # tốn RAM ~120MB). Nếu chỉ load index có sẵn (không query) thì không tải.
        self._model = None
        self._chunks: List[dict] = []          # [{id, text, metadata}]
        self._matrix: Optional[np.ndarray] = None  # (n, d) đã normalize

    # -------------------------------------------------- model
    def _ensure_model(self):
        if self._model is None:
            # Import tách ra để file này vẫn import được khi chưa cài torch.
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def embed(self, texts: List[str]) -> np.ndarray:
        """Trả matrix (n, d) đã L2 normalize → cosine = dot product."""
        model = self._ensure_model()
        vecs = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        arr = np.asarray(vecs, dtype=np.float32)
        # Nhân đôi bảo đảm normalize (encode đã normalize nhưng theo L2 numpy đôi khi
        # có sai số float; ta chuẩn lại để dot product = cosine chính xác).
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return arr / norms

    # -------------------------------------------------- build / persist
    def build(self, chunks: List[dict], path: str) -> None:
        """Embed text của từng chunk rồi lưu JSON. chunks: [{id, text, metadata}]."""
        texts = [c["text"] for c in chunks]
        vecs = self.embed(texts) if texts else np.zeros((0, 1), dtype=np.float32)
        data = {
            "version": 1,
            "model": self.model_name,
            "chunks": [
                {
                    "id": c["id"],
                    "text": c["text"],
                    "metadata": c.get("metadata", {}),
                    "embedding": vecs[i].tolist() if len(vecs) else [],
                }
                for i, c in enumerate(chunks)
            ],
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)
        # Nạp lại vào memory để dùng ngay.
        self.load(path)

    def load(self, path: str) -> None:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.model_name = data.get("model", self.model_name)
        self._chunks = [
            {"id": c["id"], "text": c["text"], "metadata": c.get("metadata", {})}
            for c in data.get("chunks", [])
        ]
        embs = [c.get("embedding", []) for c in data.get("chunks", [])]
        if embs and embs[0]:
            arr = np.asarray(embs, dtype=np.float32)
            norms = np.linalg.norm(arr, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            self._matrix = arr / norms
        else:
            self._matrix = None

    # -------------------------------------------------- query
    def query(self, text: str, top_k: int = 5) -> List[dict]:
        """Top-k chunk giống query nhất (cosine). Trả list {id, text, metadata, score}."""
        if self._matrix is None or len(self._chunks) == 0:
            return []
        q = self.embed([text])[0]
        scores = self._matrix @ q              # (n,) — cosine vì đã normalize
        k = min(top_k, len(self._chunks))
        # argpartition lấy k chỉ số lớn nhất, rồi sort giảm dần.
        idx = np.argpartition(-scores, k - 1)[:k]
        idx = idx[np.argsort(-scores[idx])]
        out = []
        for i in idx:
            c = self._chunks[int(i)]
            out.append({
                "id": c["id"],
                "text": c["text"],
                "metadata": c["metadata"],
                "score": float(scores[int(i)]),
            })
        return out

    @property
    def size(self) -> int:
        return len(self._chunks)
