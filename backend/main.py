"""
main.py — SpaceLLM Backend Entry Point
=======================================
Starts the FastAPI app, initialises the database, loads the model,
and schedules the MAPE-K controller loop.

Run with:
    uvicorn main:app --host 0.0.0.0 --port 8000 --reload
"""

import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from apscheduler.schedulers.asyncio import AsyncIOScheduler

from config import settings
from database.db import init_db
from core.inference import load_model
from mape_k.controller import run_mape_cycle
from api.routes import generate, feedback, health

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

scheduler = AsyncIOScheduler()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # ── Startup ──────────────────────────────────────────────────────────
    logger.info("Initialising database …")
    await init_db()

    logger.info("Loading SpaceLLM model …")
    asyncio.create_task(load_model())   # non-blocking; health endpoint shows status

    # Schedule MAPE-K controller
    scheduler.add_job(
        run_mape_cycle,
        trigger="interval",
        hours=settings.RETRAIN_SCHEDULE_HOURS,
        id="mape_k_loop",
        replace_existing=True,
    )
    scheduler.start()
    logger.info(
        "MAPE-K scheduler started — runs every %dh", settings.RETRAIN_SCHEDULE_HOURS
    )

    yield   # app is running

    # ── Shutdown ─────────────────────────────────────────────────────────
    scheduler.shutdown(wait=False)
    logger.info("SpaceLLM backend shutdown complete.")


app = FastAPI(
    title="SpaceLLM Mission Control API",
    version="1.0.0",
    description=(
        "Domain-specific LLM backend with MAPE-K continual learning "
        "for Space Missions, Astronomy, and Aerospace Engineering."
    ),
    lifespan=lifespan,
)

# ── CORS ─────────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routers ───────────────────────────────────────────────────────────────────
app.include_router(generate.router, tags=["Inference"])
app.include_router(feedback.router, tags=["Feedback"])
app.include_router(health.router,   tags=["Health & Admin"])
