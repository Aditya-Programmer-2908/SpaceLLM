from pydantic_settings import BaseSettings
from pathlib import Path


class Settings(BaseSettings):
    # ── HuggingFace ──────────────────────────────────────────────
    HF_TOKEN: str = ""
    HF_REPO_ID: str = "AdityaPS/SpaceLLM_v1"     # base adapter repo
    HF_ORG: str = "AdityaPS"                       # org/user for new versions

    # ── Model ────────────────────────────────────────────────────
    BASE_MODEL_ID: str = "openai/gpt-oss-20b"
    DEVICE_MAP: str = "auto"                        # "auto" | "cpu" | "cuda:0"
    LOAD_IN_4BIT: bool = False                      # set True if VRAM limited
    MAX_NEW_TOKENS: int = 512
    TEMPERATURE: float = 0.7
    TOP_P: float = 0.9

    # ── Database ─────────────────────────────────────────────────
    DB_PATH: str = "spacellm.db"

    # ── MAPE-K thresholds ────────────────────────────────────────
    MIN_FEEDBACK_FOR_RETRAIN: int = 20   # how many corrected samples needed
    MIN_NEG_RATIO: float = 0.3           # if neg-feedback ratio ≥ this → flag
    BERTSCORE_THRESHOLD: float = 0.82    # responses below this are "low quality"
    RETRAIN_SCHEDULE_HOURS: int = 24     # how often MAPE-K controller runs

    # ── Training ─────────────────────────────────────────────────
    LORA_R: int = 16
    LORA_ALPHA: int = 32
    LORA_DROPOUT: float = 0.05
    TRAIN_EPOCHS: int = 3
    TRAIN_BATCH_SIZE: int = 2
    GRADIENT_ACCUM: int = 4
    LEARNING_RATE: float = 2e-5
    WARMUP_STEPS: int = 50
    OUTPUT_DIR: str = "lora_checkpoints"

    # ── Server ───────────────────────────────────────────────────
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    CORS_ORIGINS: list[str] = ["http://localhost:3000", "http://localhost:5500",
                               "http://127.0.0.1:5500"]

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()
CHECKPOINT_DIR = Path(settings.OUTPUT_DIR)
CHECKPOINT_DIR.mkdir(exist_ok=True)
