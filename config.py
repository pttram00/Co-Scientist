"""Cấu hình chung cho toàn bộ framework multi-agent."""
from dataclasses import dataclass, field
import os
from pathlib import Path
from dotenv import load_dotenv  # Sửa từ load_config thành load_dotenv

# Nạp biến môi trường từ file .env nằm cạnh config.py, bất kể thư mục làm việc hiện tại.
# (load_dotenv() không tham số chỉ tìm trong cwd — dễ trượt khi chạy từ thư mục khác.)
_ENV_PATH = Path(__file__).resolve().parent / ".env"
load_dotenv(dotenv_path=_ENV_PATH)


@dataclass
class LLMConfig:
    model: str = "GLM-5.2"
    max_tokens: int = 2000
    temperature: float = 0.7
    api_key: str = field(default_factory=lambda: os.environ.get("ANTHROPIC_AUTH_TOKEN", ""))
    api_base: str = field(default_factory=lambda: os.environ.get("ANTHROPIC_BASE_URL", "https://api.anthropic.com"))
    max_retries: int = 3
    max_concurrency: int = 5  # giới hạn số lời gọi LLM song song / agent


@dataclass
class RetrieverConfig:
    # Số paper tối đa mong muốn thu về MỖI NGUỒN cho MỘT query trước khi gộp.
    # 3 nguồn × N query × k_per_query là số call/số record raw.
    k_per_source: int = 10
    # Sau khi gộp + dedup + rank theo citation, giữ top pool_size làm pool grounding.
    pool_size: int = 20
    # Ngưỡng tối thiểu để coi cache memory.papers đã "đủ": nếu đủ thì skip retrieval.
    min_papers_for_grounding: int = 5
    # Số paper tối đa truyền vào 1 prompt (_select_papers_llm).
    papers_per_strategy: int = 7
    # Semantic Scholar API key (tùy chọn; trống -> rate limit thấp hơn nhưng vẫn dùng được).
    # Chỉ đọc từ .env / biến môi trường — KHÔNG ghi key thật vào code (code được push lên GitHub).
    semantic_scholar_api_key: str = field(
        default_factory=lambda: os.environ.get("SEMANTIC_SCHOLAR_API_KEY", "")
    )
    # OpenAlex khuyến khích mailto (polite pool) — không bắt buộc; trống vẫn gọi được.
    openalex_mailto: str = field(
        default_factory=lambda: os.environ.get("OPENALEX_MAILTO", "")
    )
    # arXiv yêu cầu User-Agent định danh rõ; dùng khi gửi request.
    user_agent: str = "Co-Scientist/0.1 (research-assistant; mailto:research@example.com)"
    request_timeout: float = 30.0   # giây — cho từng call tới 3 nguồn API ngoài.
    max_retries: int = 2             # retry từng nguồn khi timeout/429/5xx.
    # Số HTTP request song song tối đa tới các nguồn ngoài. Reflection tra cứu cho từng
    # giả thuyết nên cần chặn để không bị rate limit (arXiv, S2 không key).
    max_concurrent_requests: int = 4
    # --- Tra cứu trong full review của ReflectionAgent (nhỏ hơn grounding của Generation) ---
    review_max_queries: int = 2        # số query tối đa / giả thuyết (lấy từ initial review)
    review_k_per_source: int = 5       # số paper / nguồn / query
    review_papers_per_review: int = 6  # số paper đưa vào prompt full review


@dataclass
class OrchestratorConfig:
    n_iterations: int = 3
    hypotheses_per_iteration: int = 6          # số giả thuyết Generation sinh mỗi vòng
    matches_per_iteration: int = 10             # số trận đấu Ranking chạy mỗi vòng
    top_k_for_evolution: int = 4                # số giả thuyết top được Evolution cải tiến
    # Ngưỡng cosine (embedding) để coi 1 cặp là "nghi trùng". Cặp nghi trùng còn phải được
    # LLM xác nhận mới bị đánh dấu duplicate, nên ngưỡng có thể thấp hơn một chút để ít bỏ sót.
    # Nên hiệu chỉnh lại bằng một bộ cặp mẫu khi đổi model embedding.
    proximity_duplicate_threshold: float = 0.80
    # Số cặp nghi trùng tối đa hỏi LLM mỗi lượt (cặp chưa kịp hỏi sẽ được hỏi ở lượt sau).
    proximity_max_llm_checks_per_iteration: int = 10
    output_dir: str = "output"


@dataclass
class EmbeddingConfig:
    # Cấu hình cho sentence embedding dùng trong ProximityAgent (cosine similarity).
    # Model chạy local qua sentence-transformers; lần đầu dùng sẽ auto-download về
    # cache HuggingFace (~/.cache/huggingface).
    # Model đa ngôn ngữ vì giả thuyết viết tiếng Việt (all-MiniLM-L6-v2 chỉ huấn luyện
    # cho tiếng Anh và tokenizer uncased bỏ dấu tiếng Việt).
    model_name: str = "paraphrase-multilingual-MiniLM-L12-v2"
    device: str = "cpu"                          # "cpu" | "cuda" — cpu an toàn mặc định
    batch_size: int = 32                         # số câu encode cùng lúc trong 1 batch


@dataclass
class AppConfig:
    llm: LLMConfig = field(default_factory=LLMConfig)
    retriever: RetrieverConfig = field(default_factory=RetrieverConfig)
    orchestrator: OrchestratorConfig = field(default_factory=OrchestratorConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
