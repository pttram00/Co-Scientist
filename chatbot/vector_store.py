"""Vector store cho RAG: embed chunks (sentence-transformers) + cosine search.

Lưu/đọc index dạng JSON giản đơn ( đủ cho corpus nhỏ ~ vài chục chunk).
Embedding model mặc định: paraphrase-multilingual-MiniLM-L12-v2 — đa ngữ
(có tiếng Việt), nhẹ (~120MB), chạy CPU được.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, List, Optional

import numpy as np

# Hàm encode: danh sách text -> danh sách vector (giống Encoder của ProximityAgent),
# để chế độ mô phỏng tiêm encoder giả và chế độ thật dùng lại model đã nạp sẵn.
Encoder = Callable[[List[str]], List[List[float]]]


class VectorStore:
    def __init__(self, model_name: str = "paraphrase-multilingual-MiniLM-L12-v2",
                 encoder: Optional[Encoder] = None):
        self.model_name = model_name
        # Lazy-load: SentenceTransformer chỉ nạp khi cần embed (lần đầu chậm ~3s,
        # tốn RAM ~120MB). Nếu chỉ load index có sẵn (không query) thì không tải.
        self._model = None
        # encoder tiêm từ ngoài: dùng thay SentenceTransformer nếu có. Cho phép
        # (a) chế độ mô phỏng chạy không cần torch, (b) tái dùng encoder mà
        # ProximityAgent đã nạp, khỏi giữ 2 model trong RAM.
        self._encoder = encoder
        self._chunks: List[dict] = []              # [{id, text, metadata}]
        self._matrix: Optional[np.ndarray] = None  # (n, d) đã normalize

    # -------------------------------------------------- model
    def _ensure_model(self):
        """ Hàm này được dùng để đảm bảo model được nạp khi cần thiết, tránh nạp model khi chỉ load index có sẵn."""
        if self._model is None:
            # Import tách ra để file này vẫn import được khi chưa cài torch.
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def embed(self, texts: List[str]) -> np.ndarray:
        """
        Trả matrix (n,d) embedding đã normalize - nghĩa là ở đây sẽ chuẩn hóa vector embedding để có độ dài bằng 1
        → từ đó có thể tính cosine similarity bằng dot product.
        """
        if self._encoder is not None:
            vecs = self._encoder(texts)                       # encoder tiêm ngoài
        else:
            model = self._ensure_model()
            vecs = model.encode(texts,
                                 normalize_embeddings=True,
                                 show_progress_bar=False)     # chuyển đổi danh sách văn bản sang số học
        arr = np.asarray(vecs, dtype=np.float32)              # ở đây chuyển vector embedding sang dạng numpy array với kiểu dữ liệu float32

        norms = np.linalg.norm(arr, axis=1, keepdims=True)    # norms đại diện cho độ lớn( chiều dài) của từng vector
        norms[norms == 0] = 1.0                               # nếu vector rỗng thì độ lớn sẽ = 1 để tránh chia cho 0
        return arr / norms                      # đây là chuẩn hóa L2 cho từng vector



    def build(self, chunks: List[dict], path: str) -> None:
        """Embed text của từng chunk rồi lưu JSON. chunks: [{id, text, metadata}]."""
        texts = [c["text"] for c in chunks]
        # Nếu có text thì embed còn không thì sẽ tạo một matrix rỗng để tránh lỗi
        vecs = self.embed(texts) if texts else np.zeros((0, 1), dtype=np.float32)
        # tạo một data để lưu trữ thông tin để có thể dễ dàng truy suất và sử dụng sau này.
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
        # ta sẽ lưu data vào file JSON để có thể tái sử dụng dễ hơn sau này
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)
        # Nạp lại vào memory để dùng ngay.
        self.load(path)

    def load(self, path: str) -> None:
        """ Load index từ file JSON đã lưu, nếu có embedding thì nạp vào matrix"""
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



    def query(self, text: str, top_k: int = 5) -> List[dict]:
        """
        Top-k chunk giống query nhất (cosine). Trả list {id, text, metadata, score}.
        Ở đây là nới làm việc với dữ liệu đầu vào từ người dùng.
        """
        if self._matrix is None or len(self._chunks) == 0:
            return []
        q = self.embed([text])[0]                    # khi này embed text đầu vào của người dùng
        scores = self._matrix @ q                    # đây là bước nhân ma trận giữa sơ sở dữ liệu và query ban đầu → kết quả sinh ra là điểm cosine similarity
        k = min(top_k, len(self._chunks))

        # argpartition lấy k chỉ số lớn nhất, rồi sort giảm dần.
        # đây là một thuật toán giúp lấy nhanh các giá trị có điểm số cao nhất mà không cần sắp xếp
        idx = np.argpartition(-scores, k - 1)[:k]    
        idx = idx[np.argsort(-scores[idx])] # sắp xếp trên tập nhỏ k
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
