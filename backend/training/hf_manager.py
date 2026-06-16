"""
training/hf_manager.py
-----------------------
Standalone helpers for pushing and pulling LoRA adapters to/from HuggingFace.
Can be used independently of the MAPE-K loop.
"""

import logging
from pathlib import Path
from config import settings

logger = logging.getLogger(__name__)


def push_adapter(local_dir: str, version_tag: str, private: bool = False) -> str:
    """Push a local checkpoint directory to HuggingFace.

    Returns the full repo_id.
    """
    from huggingface_hub import HfApi
    api = HfApi(token=settings.HF_TOKEN or None)
    repo_id = f"{settings.HF_ORG}/{version_tag}"
    api.create_repo(repo_id=repo_id, exist_ok=True, private=private)
    api.upload_folder(
        folder_path=local_dir,
        repo_id=repo_id,
        commit_message=f"SpaceLLM adapter — {version_tag}",
    )
    logger.info("Pushed %s → https://huggingface.co/%s", local_dir, repo_id)
    return repo_id


def pull_adapter(repo_id: str, local_dir: str) -> str:
    """Download adapter files from HuggingFace into local_dir."""
    from huggingface_hub import snapshot_download
    path = snapshot_download(
        repo_id=repo_id,
        local_dir=local_dir,
        token=settings.HF_TOKEN or None,
    )
    logger.info("Downloaded %s → %s", repo_id, path)
    return path


def list_versions() -> list[dict]:
    """List all SpaceLLM adapter versions in the HF org."""
    from huggingface_hub import HfApi
    api = HfApi(token=settings.HF_TOKEN or None)
    models = api.list_models(author=settings.HF_ORG, search="SpaceLLM")
    return [
        {"id": m.id, "last_modified": str(m.last_modified)}
        for m in models
    ]
