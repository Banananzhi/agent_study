CREATE TABLE agent_run (
    run_id TEXT PRIMARY KEY,
    account_id TEXT NOT NULL REFERENCES media_account(account_id),
    idempotency_key TEXT NOT NULL,
    request_hash TEXT NOT NULL,
    request_json TEXT NOT NULL,
    account_snapshot_json TEXT NOT NULL,
    strategy_snapshot_json TEXT NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('PENDING', 'RUNNING', 'WAITING_APPROVAL', 'COMPLETED', 'FAILED', 'CANCELLED')),
    error_json TEXT,
    version INTEGER NOT NULL CHECK (version >= 1),
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    UNIQUE (account_id, idempotency_key)
);
CREATE INDEX idx_run_account_status ON agent_run (account_id, status, created_at);

CREATE TABLE agent_task (
    task_id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL REFERENCES agent_run(run_id),
    task_key TEXT NOT NULL,
    task_type TEXT NOT NULL CHECK (task_type IN ('coordinator', 'research', 'planning', 'content', 'review', 'persist_export')),
    ordinal INTEGER NOT NULL CHECK (ordinal >= 0),
    input_json TEXT NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('PENDING', 'READY', 'RUNNING', 'COMPLETED', 'FAILED', 'SKIPPED')),
    attempts INTEGER NOT NULL DEFAULT 0 CHECK (attempts >= 0),
    result_json TEXT,
    skip_reason TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    UNIQUE (run_id, task_key),
    UNIQUE (run_id, task_id),
    UNIQUE (run_id, ordinal),
    CHECK ((status IN ('COMPLETED', 'FAILED') AND result_json IS NOT NULL)
        OR (status NOT IN ('COMPLETED', 'FAILED') AND result_json IS NULL)),
    CHECK ((status = 'SKIPPED' AND skip_reason IS NOT NULL)
        OR (status != 'SKIPPED' AND skip_reason IS NULL))
);
CREATE INDEX idx_task_run_status ON agent_task (run_id, status, ordinal);

CREATE TABLE task_dependency (
    run_id TEXT NOT NULL,
    task_id TEXT NOT NULL,
    dependency_id TEXT NOT NULL,
    ordinal INTEGER NOT NULL,
    PRIMARY KEY (task_id, dependency_id),
    FOREIGN KEY (run_id, task_id) REFERENCES agent_task(run_id, task_id),
    FOREIGN KEY (run_id, dependency_id) REFERENCES agent_task(run_id, task_id),
    CHECK (task_id != dependency_id)
);

CREATE TABLE run_event (
    event_id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL REFERENCES agent_run(run_id),
    seq INTEGER NOT NULL CHECK (seq >= 1),
    event_type TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    UNIQUE (run_id, seq)
);
