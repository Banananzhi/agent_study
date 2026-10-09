CREATE TABLE research_source (
    source_id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL REFERENCES agent_run(run_id),
    task_id TEXT NOT NULL,
    fingerprint TEXT NOT NULL,
    snapshot_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    UNIQUE (task_id, fingerprint),
    FOREIGN KEY (run_id, task_id) REFERENCES agent_task(run_id, task_id)
);
CREATE INDEX idx_source_run ON research_source (run_id, created_at, source_id);

CREATE TABLE tool_execution (
    execution_id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL REFERENCES agent_run(run_id),
    task_id TEXT NOT NULL,
    tool_name TEXT NOT NULL,
    arguments_json TEXT NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('RUNNING', 'SUCCEEDED', 'FAILED')),
    result_json TEXT,
    error_code TEXT,
    attempts INTEGER NOT NULL DEFAULT 0 CHECK (attempts >= 0),
    duration_ms INTEGER NOT NULL DEFAULT 0 CHECK (duration_ms >= 0),
    created_at TEXT NOT NULL,
    finished_at TEXT,
    FOREIGN KEY (run_id, task_id) REFERENCES agent_task(run_id, task_id),
    CHECK ((status = 'RUNNING' AND result_json IS NULL AND finished_at IS NULL)
        OR (status != 'RUNNING' AND result_json IS NOT NULL AND finished_at IS NOT NULL))
);
CREATE INDEX idx_execution_task ON tool_execution (run_id, task_id, created_at);
