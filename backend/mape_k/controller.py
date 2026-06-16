"""
mape_k/controller.py
--------------------
The MAPE-K controller: ties Monitor → Analyze → Plan → Execute together.
Called by APScheduler every N hours AND can be triggered manually via the
/admin/mape-run endpoint.
"""

import logging
from typing import Optional

from database.db import AsyncSessionLocal
from database import knowledge as kb
from mape_k import analyze, plan, execute

logger = logging.getLogger(__name__)


async def run_mape_cycle(triggered_by: str = "scheduler") -> dict:
    """
    Execute one full MAPE-K cycle.

    Returns a summary dict suitable for API responses.
    """
    async with AsyncSessionLocal() as db:
        run_id = await kb.start_mape_run(db, triggered_by=triggered_by)
        log_lines: list[str] = []

        try:
            # ── ANALYZE ──────────────────────────────────────────────────
            report = await analyze.analyze(db)
            log_lines.append(
                f"[A] feedback={report.total_feedback} "
                f"neg_ratio={report.neg_ratio:.2%} "
                f"new_samples={len(report.new_sample_ids)}"
            )

            # ── PLAN ─────────────────────────────────────────────────────
            retrain_plan = await plan.plan(db, report)
            log_lines.append(f"[P] {retrain_plan.reason}")

            new_version: Optional[str] = None

            if retrain_plan.should_retrain:
                # ── EXECUTE ──────────────────────────────────────────────
                log_lines.append(
                    f"[E] Starting training → {retrain_plan.new_version_tag}"
                )
                result = await execute.execute(db, retrain_plan)

                if result.success:
                    new_version = result.new_version_tag
                    log_lines.append(
                        f"[E] Success — {result.hf_repo_id} "
                        f"BERTScore={result.bertscore}"
                    )
                else:
                    log_lines.append(f"[E] Failed — {result.error}")

            await kb.finish_mape_run(
                db,
                run_id=run_id,
                status="done",
                samples_found=len(report.new_sample_ids),
                retrain_decided=retrain_plan.should_retrain,
                new_version=new_version,
                log="\n".join(log_lines),
            )

            return {
                "run_id": run_id,
                "status": "done",
                "retrained": retrain_plan.should_retrain,
                "new_version": new_version,
                "log": log_lines,
            }

        except Exception as exc:
            logger.error("MAPE-K cycle failed: %s", exc, exc_info=True)
            await kb.finish_mape_run(
                db,
                run_id=run_id,
                status="failed",
                log=f"Exception: {exc}",
            )
            return {"run_id": run_id, "status": "failed", "error": str(exc)}
