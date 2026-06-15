"""
AI-Augmented SOC Detection Engine — Unified API.

Serves both structured IDS (LightGBM on network flows) and semantic
log analysis (Sentence-BERT + Isolation Forest) through a single
API surface with shared alert schema and MITRE mapping.
"""

import asyncio
import logging
import time
from contextlib import asynccontextmanager
from typing import Dict, List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from ..analyzers.semantic import SemanticAnalyzer
from ..analyzers.structured import StructuredAnalyzer
from ..core.alert_builder import AlertBuilder
from ..core.router import InputRouter
from ..core.schemas import (
    AnalysisEngine,
    AnalysisResult,
    DetectionRequest,
    SOCAlert,
)
from ..mitre.mapper import MitreMapper
from ..monitoring.drift import DriftMonitor
from ..rules.engine import RuleEngine

# ------------------------------------------------------------------ #
#  Logging                                                             #
# ------------------------------------------------------------------ #

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("api")

# ------------------------------------------------------------------ #
#  Global state  (populated during lifespan)                           #
# ------------------------------------------------------------------ #

structured_analyzer: Optional[StructuredAnalyzer] = None
semantic_analyzer: Optional[SemanticAnalyzer] = None
input_router: Optional[InputRouter] = None
alert_builder: Optional[AlertBuilder] = None


# ------------------------------------------------------------------ #
#  Lifespan                                                            #
# ------------------------------------------------------------------ #

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models and initialise shared services on startup."""
    global structured_analyzer, semantic_analyzer, input_router, alert_builder

    # ── Load analyzers ──
    logger.info("Loading analyzers …")

    structured_analyzer = StructuredAnalyzer.load("outputs/checkpoints")
    if structured_analyzer.is_ready():
        logger.info("✅ Structured IDS analyzer loaded.")
    else:
        logger.warning("⚠️  Structured IDS analyzer not available.")

    semantic_analyzer = SemanticAnalyzer.load("models/siem_model")
    if semantic_analyzer.is_ready():
        logger.info("✅ Semantic analyzer loaded.")
    else:
        logger.warning("⚠️  Semantic analyzer not available.")

    # ── Shared services ──
    rule_engine = RuleEngine()
    mitre_mapper = MitreMapper()
    drift_monitor = DriftMonitor()

    input_router = InputRouter(
        known_feature_columns=structured_analyzer.feature_columns,
    )
    alert_builder = AlertBuilder(
        mitre_mapper=mitre_mapper,
        rule_engine=rule_engine,
        drift_monitor=drift_monitor,
    )

    logger.info("API initialisation complete.")
    yield
    logger.info("Shutting down.")


# ------------------------------------------------------------------ #
#  App                                                                 #
# ------------------------------------------------------------------ #

app = FastAPI(
    title="AI-Augmented SOC Detection Engine",
    description=(
        "Unified detection API combining structured network-flow IDS "
        "(LightGBM on CIC-IDS2017) and semantic textual log anomaly "
        "detection (Sentence-BERT + Isolation Forest)."
    ),
    version="2.0.0",
    lifespan=lifespan,
)


# ------------------------------------------------------------------ #
#  Helper                                                              #
# ------------------------------------------------------------------ #

async def _run_analysis(
    request: DetectionRequest,
    engines: List[AnalysisEngine],
) -> SOCAlert:
    """
    Dispatch *request* to the specified engines, fuse results, and
    return a ``SOCAlert``.
    """
    start = time.time()
    results: List[AnalysisResult] = []

    # Run engines (potentially in parallel via to_thread)
    tasks = []

    if AnalysisEngine.STRUCTURED in engines and structured_analyzer and structured_analyzer.is_ready():
        tasks.append(asyncio.to_thread(structured_analyzer.analyze, request))

    if AnalysisEngine.SEMANTIC in engines and semantic_analyzer and semantic_analyzer.is_ready():
        tasks.append(asyncio.to_thread(semantic_analyzer.analyze, request))

    if tasks:
        results = list(await asyncio.gather(*tasks))

    processing_time_ms = (time.time() - start) * 1000

    return alert_builder.build(request, results, processing_time_ms)


# ------------------------------------------------------------------ #
#  Endpoints                                                           #
# ------------------------------------------------------------------ #

@app.post(
    "/detect",
    response_model=SOCAlert,
    summary="Auto-route detection",
    description=(
        "Inspects the request and routes to the appropriate engine(s). "
        "Send `flow_features` for IDS, `log_text` for semantic analysis, "
        "or both for dual analysis."
    ),
)
async def detect(request: DetectionRequest):
    """Auto-routing detection endpoint."""
    engines = input_router.route(request)
    return await _run_analysis(request, engines)


@app.post(
    "/detect/flow",
    response_model=SOCAlert,
    summary="Structured flow analysis",
    description="Explicit structured IDS analysis on network-flow features.",
)
async def detect_flow(request: DetectionRequest):
    """Explicit structured-only detection."""
    if not request.flow_features:
        raise HTTPException(
            status_code=422,
            detail="flow_features required for /detect/flow",
        )
    return await _run_analysis(request, [AnalysisEngine.STRUCTURED])


@app.post(
    "/detect/log",
    response_model=SOCAlert,
    summary="Semantic log analysis",
    description="Explicit semantic anomaly detection on free-text log lines.",
)
async def detect_log(request: DetectionRequest):
    """Explicit semantic-only detection."""
    if not request.log_text:
        raise HTTPException(
            status_code=422,
            detail="log_text required for /detect/log",
        )
    return await _run_analysis(request, [AnalysisEngine.SEMANTIC])


@app.get("/health", summary="Health check")
async def health():
    """Return per-engine health and drift status."""
    return {
        "status": "healthy",
        "engines": {
            "structured_ids": (
                structured_analyzer.health()
                if structured_analyzer
                else {"ready": False}
            ),
            "semantic_anomaly": (
                semantic_analyzer.health()
                if semantic_analyzer
                else {"ready": False}
            ),
        },
        "drift": (
            alert_builder.drift_monitor.get_status()
            if alert_builder
            else {}
        ),
        "version": "2.0.0",
    }

