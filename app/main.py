"""HTTP API (Section 9): POST /chat and GET /health.

Built on Starlette (the ASGI toolkit FastAPI is built on) and served by uvicorn on
port 8000. Ingestion runs in a background thread at start-up; /health returns 503 until
the index is built and the engine is ready.
"""
from __future__ import annotations

import logging
import threading

from starlette.applications import Starlette
from starlette.concurrency import run_in_threadpool
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route

from app.config import Settings, get_settings
from app.data.db import Database
from app.ingestion.pipeline import build_index, save_index
from app.llm.client import LLMClient
from app.logging_setup import configure_logging, log_event
from app.orchestration.engine import Engine
from app.orchestration.session import SessionOwnershipError, SessionStore
from app.retrieval.retriever import KnowledgeBase

log = logging.getLogger("app")

MAX_MESSAGE_CHARS = 4000


class AppState:
    def __init__(self):
        self.ready = threading.Event()
        self.failed: str | None = None
        self.engine: Engine | None = None
        self.db: Database | None = None
        self.sessions = SessionStore()


def build_engine(settings: Settings, llm: LLMClient | None = None, kb: KnowledgeBase | None = None) -> Engine:
    if kb is None:
        index = build_index(settings.corpus_dir, settings.manifest_path, settings.ocr_lang)
        save_index(index, settings.index_path)
        kb = KnowledgeBase(index)
    llm = llm or LLMClient(settings.llm_provider, settings.llm_model, settings.llm_timeout_s,
                           hard_limit=settings.corpus_token_hard_limit)
    return Engine(Database(settings.db_path), kb, llm, settings.today, settings.corpus_token_budget,
                  settings.corpus_token_hard_limit)


def create_app(settings: Settings | None = None, engine_factory=None, background: bool = True) -> Starlette:
    settings = settings or get_settings()
    configure_logging(settings.log_level, settings.log_file)
    state = AppState()
    state.db = Database(settings.db_path)

    def _init():
        try:
            state.engine = (engine_factory or build_engine)(settings)
            state.ready.set()
            log_event(log, "service_ready", llm_provider=settings.llm_provider, app_today=settings.today().isoformat())
        except Exception as exc:  # keep /health at 503 and say why in the logs
            state.failed = type(exc).__name__
            log.exception("startup_failed")

    async def health(request: Request) -> JSONResponse:
        if state.ready.is_set():
            return JSONResponse({"status": "ok"})
        return JSONResponse({"status": "starting" if not state.failed else "error"}, status_code=503)

    async def chat(request: Request) -> JSONResponse:
        employee_id = (request.headers.get("x-employee-id") or "").strip()
        requester = state.db.get_employee(employee_id) if employee_id else None
        if requester is None:
            return JSONResponse({"detail": "Missing or unknown X-Employee-Id"}, status_code=401)

        try:
            body = await request.json()
        except Exception:
            return JSONResponse({"detail": "Body must be valid JSON"}, status_code=400)
        if not isinstance(body, dict):
            return JSONResponse({"detail": "Body must be a JSON object"}, status_code=400)
        session_id, message = body.get("session_id"), body.get("message")
        if not isinstance(session_id, str) or not session_id.strip():
            return JSONResponse({"detail": "session_id must be a non-empty string"}, status_code=400)
        if not isinstance(message, str) or not message.strip() or len(message) > MAX_MESSAGE_CHARS:
            return JSONResponse({"detail": f"message must be a non-empty string of at most {MAX_MESSAGE_CHARS} characters"},
                                status_code=400)

        try:
            session = state.sessions.get_or_create(session_id, requester.employee_id)
        except SessionOwnershipError:
            log_event(log, "session_owner_mismatch", employee_id=requester.employee_id)
            return JSONResponse({"detail": "This session belongs to another employee"}, status_code=403)

        if not state.ready.is_set():
            return JSONResponse({"detail": "Service is starting"}, status_code=503)

        def run():
            with state.sessions.lock(session_id):  # one message at a time per session
                return state.engine.handle(requester, session, message.strip())

        resp = await run_in_threadpool(run)
        return JSONResponse(resp.to_json())

    app = Starlette(routes=[Route("/chat", chat, methods=["POST"]), Route("/health", health, methods=["GET"])])
    app.state.svc = state

    if background:
        threading.Thread(target=_init, name="ingestion", daemon=True).start()
    else:
        _init()
    return app



def get_app() -> Starlette:  # uvicorn --factory entry point
    return create_app()
