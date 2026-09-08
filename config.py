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
    orchestrator: OrchestratorConfig = field(default_factory=OrchestratorConfig)
 