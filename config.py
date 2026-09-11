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
    model: str = "glm-5.2"
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
    semantic_scholar_api_key: str = field(
        default_factory=lambda: os.environ.get("SEMANTIC_SCHOLAR_API_KEY", "s2k-SS3V2nHplA6qHyX8QPQVqubfAb2iXzDXT1DDa2EO")
    )
    # OpenAlex khuyến khích mailto (polite pool) — không bắt buộc; trống vẫn gọi được.
    openalex_mailto: str = field(
        default_factory=lambda: os.environ.get("OPENALEX_MAILTO", "")
    )
    # arXiv yêu cầu User-Agent định danh rõ; dùng khi gửi request.
    user_agent: str = "Co-Scientist/0.1 (research-assistant; mailto:research@example.com)"
    request_timeout: float = 30.0   # giây — cho từng call tới 3 nguồn API ngoài.
    max_retries: int = 2             # retry từng nguồn khi timeout/5xx.


@dataclass
class OrchestratorConfig:
    n_iterations: int = 3
    hypotheses_per_iteration: int = 6          # số giả thuyết Generation sinh mỗi vòng
    matches_per_iteration: int = 10             # số trận đấu Ranking chạy mỗi vòng
    top_k_for_evolution: int = 4                # số giả thuyết top được Evolution cải tiến
    proximity_duplicate_threshold: float = 0.85  # ngưỡng coi là trùng lặp
    output_dir: str = "output"


@dataclass
class AppConfig:
    llm: LLMConfig = field(default_factory=LLMConfig)
    retriever: RetrieverConfig = field(default_factory=RetrieverConfig)
    orchestrator: OrchestratorConfig = field(default_factory=OrchestratorConfig)
 