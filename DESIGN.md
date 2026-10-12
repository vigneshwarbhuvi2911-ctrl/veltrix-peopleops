# Design notes

## 1. Ingesting tables and the scanned document

Each format has a parser (`app/ingestion/parsers.py`), and all of them emit heading, text and
table blocks in reading order:

- **PDF:** pdfplumber; headings come from font size and weight. Words inside a
  `find_tables()` box are swapped for one table block at the same position.
- **DOCX:** read in body order with heading styles.
- **HTML / Markdown:** tags, `#` headings and pipe tables.

Tables become Markdown with their header row, and long ones are split by rows with the
header repeated. The "Document ID / Version / …" card is recognised by its labels and
dropped.

A PDF page with almost no text layer is rendered at 300 dpi and OCR'd with Tesseract
(`ocr.py`); its locator is `p. N`. Holiday dates are read from calendar table cells at
ingestion (`holidays.py`), skipping restricted or optional sections. Scans fall back to
reading text lines. Text addressed to an AI is quarantined and logged.

The index is rebuilt at every start, and nothing from the corpus is written into code or
prompts.

## 2. Chunking and retrieval

One chunk is one section (≤ ~380 tokens), because sections are both the unit of a rule and
the citation locator. Scoring is BM25 with one model per role, built only from the documents
that role can read. The planner adds synonyms to the query.

Lexical retrieval suits a small corpus full of exact terms. It is also deterministic across
test runs, CPU-only, and needs no external service.

Pure BM25 failed here: FAQ entries are phrased like user questions and pushed the governing
policy out of the context. Packing (`retriever.py`) is therefore two-phase:

1. Every relevant document gets its best chunk, in precedence order.
2. The rest of the budget is filled by score, discounting repeat chunks from one document.

The exact `cl100k_base` count of the source block is then trimmed to 2,800 tokens.
`LLMClient` refuses any call over 3,000 and logs `corpus_tokens` for every call.

## 3. Implementing Section 5

Code decides everything that doesn't need reading comprehension (`eligibility.py`):

- **Eligibility:** access level, region or GLOBAL, and `effective_date` ≤ reference date.
- **Upcoming documents:** readable, for the region, and not yet effective.
- **Ranking:** a dense rank on `(authority, scope, effective_date)`. Equal keys share a rank,
  which is the "tests don't decide" case.

The reference date is `APP_TODAY` unless the planner extracts a past date. Each source sent
to the model carries `status` and `precedence_rank`. The prompt then applies the rest
mechanically:

- per fact, use the strongest IN FORCE source that states it;
- if same-rank sources disagree, report both values and call it a conflict;
- mention a not-yet-effective source only if it outranks the source used.

This reproduces the worked example: for India, v3 outranks v2 and is mentioned; for
Germany, the addendum outranks v3, so v3 is not mentioned. The model cites source labels,
which code maps to `(doc_id, locator)`, so it cannot invent either.

## 4. Where access control is enforced

| Concern | File / function |
|---|---|
| Identity from `X-Employee-Id` (401), body (400) | `main.py: chat` |
| Session bound to the first employee (403) | `session.py: SessionStore.get_or_create` |
| Restricted documents removed before scoring | `eligibility.py: partition_documents`, called by `retriever.py: KnowledgeBase.retrieve` |
| Per-role BM25 statistics | `retriever.py: KnowledgeBase.__init__` |
| View rules: self / direct reports / HR | `permissions.py: can_view_leave_of`, `require_view`, called inside each tool in `leave_tools.py` |
| Managers-only team view | `permissions.py: require_manager` in `leave_tools.get_team_leave` |
| Writes on own data only | `leave_tools.apply_leave` / `cancel_leave` use `ToolContext.requester`; `db.cancel_request` filters by owner |
| Name → ID only among visible employees; bulk refusal | `engine.py: _resolve_employee`, `_leave_data` |

The LLM only names a target. Role and region always come from the database.

## 5. Personal data

- **Prompts:** the planner gets ID, role and region. The composer gets
  `Employee.safe_profile()` (no name, email or phone) and ID-only tool results. Session
  history is sanitised, and names are omitted from it.
- **Answers:** `pii.py: PiiGuard.sanitize` runs on every answer. It strips employee phone
  numbers and phone-like patterns (dates and UUIDs are left alone). It masks emails as
  `r***@domain`, except team mailboxes found in the corpus. Names appear only in templates,
  and only for employees the requester may see.
- **Logs:** JSON lines with IDs, route, tools, `corpus_tokens` and latency. Message text is
  never logged, and a filter scrubs email and phone patterns as a second barrier.

## 6. Failure modes found

| Failure | Status |
|---|---|
| FAQ crowded out the governing policy | Fixed: precedence-aware packing |
| IDF on the candidate subset devalued topic words | Fixed: per-role corpus BM25 |
| "Sept" normalisation broke "September", dropping a holiday | Fixed; a test checks every seeded `working_days` |
| Restricted holidays at risk of counting as closures | Fixed: restricted/optional sections skipped |
| Metadata card leaked when a PDF drew it without table lines | Fixed: label filter |
| "thanks" would have confirmed a proposal | Fixed: an affirmative word is required |
| Model rejecting `temperature`; retries exceeding 30 s | Fixed: fallback; retry only after fast failures |
| Per-fact precedence and "same rule" detection rely on the model | Open; reduced by explicit ranks and label-only citations |

## 7. What breaks first at 10,000 documents

1. **Retrieval precision.** Fix: hybrid BM25 + CPU embeddings (FAISS or pgvector) with
   rank fusion and a reranker.
2. **The 3,000-token budget.** Phase 1 can no longer give every relevant document a slot.
   Fix: group versions into policy families at ingestion and keep only the winning and
   upcoming versions. Also pre-extract rule facts (topic, value, document) into a table, so
   precedence runs in SQL.
3. **Start-up ingestion and OCR** against the 10-minute limit. Fix: an incremental,
   content-hashed index built by a separate worker.
4. **In-memory sessions and index** tied to one worker. Fix: move them to Redis or Postgres
   and run stateless replicas.
