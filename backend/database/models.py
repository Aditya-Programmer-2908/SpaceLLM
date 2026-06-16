"""
SQLAlchemy ORM models for SpaceLLM MAPE-K persistence.

Tables
------
interactions      — every query + response pair
feedback          — thumbs up/down + optional correction text
training_samples  — curated examples ready for LoRA fine-tuning
adapter_versions  — registry of every pushed HF adapter
mape_runs         — log of each MAPE-K controller execution
"""

from datetime import datetime
from sqlalchemy import (
    Column, Integer, String, Float, Text,
    Boolean, DateTime, ForeignKey, JSON,
)
from database.db import Base


class Interaction(Base):
    __tablename__ = "interactions"

    id             = Column(Integer, primary_key=True, index=True)
    session_id     = Column(String(64), index=True, nullable=True)
    user_query     = Column(Text, nullable=False)
    model_response = Column(Text, nullable=False)
    model_version  = Column(String(32), default="SpaceLLM_v1")
    bertscore      = Column(Float, nullable=True)          # computed async
    latency_ms     = Column(Float, nullable=True)
    created_at     = Column(DateTime, default=datetime.utcnow)


class Feedback(Base):
    __tablename__ = "feedback"

    id              = Column(Integer, primary_key=True, index=True)
    interaction_id  = Column(Integer, ForeignKey("interactions.id"), index=True)
    feedback_type   = Column(String(16), nullable=False)   # "positive" | "negative"
    correction_text = Column(Text, nullable=True)          # only for negative
    is_processed    = Column(Boolean, default=False)       # moved to training?
    created_at      = Column(DateTime, default=datetime.utcnow)


class TrainingSample(Base):
    __tablename__ = "training_samples"

    id              = Column(Integer, primary_key=True, index=True)
    feedback_id     = Column(Integer, ForeignKey("feedback.id"), nullable=True)
    prompt          = Column(Text, nullable=False)
    completion      = Column(Text, nullable=False)          # the corrected answer
    source          = Column(String(32), default="human_correction")
    quality_score   = Column(Float, nullable=True)
    used_in_version = Column(String(32), nullable=True)    # e.g. "SpaceLLM_v2"
    created_at      = Column(DateTime, default=datetime.utcnow)


class AdapterVersion(Base):
    __tablename__ = "adapter_versions"

    id           = Column(Integer, primary_key=True, index=True)
    version_tag  = Column(String(32), unique=True, nullable=False)  # "SpaceLLM_v2"
    hf_repo_id   = Column(String(128), nullable=False)
    base_version = Column(String(32), nullable=True)       # what it was trained from
    bertscore    = Column(Float, nullable=True)
    train_samples= Column(Integer, nullable=True)
    notes        = Column(Text, nullable=True)
    extra_meta   = Column(JSON, nullable=True)
    pushed_at    = Column(DateTime, default=datetime.utcnow)


class MapeRun(Base):
    __tablename__ = "mape_runs"

    id              = Column(Integer, primary_key=True, index=True)
    triggered_by    = Column(String(32), default="scheduler")  # "scheduler"|"manual"
    status          = Column(String(16), default="running")    # "running"|"done"|"failed"
    samples_found   = Column(Integer, default=0)
    retrain_decided = Column(Boolean, default=False)
    new_version     = Column(String(32), nullable=True)
    log             = Column(Text, nullable=True)
    started_at      = Column(DateTime, default=datetime.utcnow)
    finished_at     = Column(DateTime, nullable=True)
