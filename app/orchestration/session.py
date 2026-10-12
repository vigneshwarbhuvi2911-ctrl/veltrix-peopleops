"""In-memory conversation state, keyed by session_id and bound to one employee."""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field

MAX_HISTORY_TURNS = 6
SESSION_TTL_S = 12 * 3600


@dataclass
class Proposal:
    name: str      # "apply_leave" | "cancel_leave"
    args: dict     # exactly the tool arguments that will be executed on confirmation


@dataclass
class Session:
    session_id: str
    employee_id: str
    history: list[tuple[str, str]] = field(default_factory=list)  # (role, text) - no names / emails / phones
    pending: Proposal | None = None
    draft: dict | None = None  # partial details from a clarify turn about a leave action
    last_seen: float = field(default_factory=time.time)

    def add_turn(self, user_text: str, assistant_text: str) -> None:
        self.history.append(("user", user_text))
        self.history.append(("assistant", assistant_text))
        self.history = self.history[-2 * MAX_HISTORY_TURNS:]


class SessionOwnershipError(Exception):
    pass


class SessionStore:
    def __init__(self):
        self._sessions: dict[str, Session] = {}
        self._locks: dict[str, threading.Lock] = {}
        self._guard = threading.Lock()

    def get_or_create(self, session_id: str, employee_id: str) -> Session:
        with self._guard:
            self._evict()
            s = self._sessions.get(session_id)
            if s is None:
                s = Session(session_id, employee_id)
                self._sessions[session_id] = s
                self._locks[session_id] = threading.Lock()
            elif s.employee_id != employee_id:
                raise SessionOwnershipError()
            s.last_seen = time.time()
            return s

    def lock(self, session_id: str) -> threading.Lock:
        with self._guard:
            return self._locks.setdefault(session_id, threading.Lock())

    def _evict(self) -> None:
        now = time.time()
        for sid in [k for k, v in self._sessions.items() if now - v.last_seen > SESSION_TTL_S]:
            # Keep the owner binding even after eviction so a session can never be taken over.
            self._sessions[sid].history.clear()
            self._sessions[sid].pending = None
            self._sessions[sid].draft = None
