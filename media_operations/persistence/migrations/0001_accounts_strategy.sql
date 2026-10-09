CREATE TABLE media_account (
    account_id TEXT PRIMARY KEY,
    tenant_id TEXT NOT NULL,
    user_id TEXT NOT NULL,
    version INTEGER NOT NULL CHECK (version >= 1),
    status TEXT NOT NULL CHECK (status IN ('ACTIVE', 'PAUSED')),
    snapshot_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE INDEX idx_account_owner ON media_account (tenant_id, user_id, status);

CREATE TABLE account_revision (
    account_id TEXT NOT NULL REFERENCES media_account(account_id),
    version INTEGER NOT NULL CHECK (version >= 1),
    snapshot_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    PRIMARY KEY (account_id, version)
);

CREATE TABLE content_strategy (
    strategy_id TEXT PRIMARY KEY,
    account_id TEXT NOT NULL REFERENCES media_account(account_id),
    version INTEGER NOT NULL CHECK (version >= 1),
    snapshot_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    UNIQUE (account_id, version)
);

CREATE TABLE account_goal (
    goal_id TEXT PRIMARY KEY,
    strategy_id TEXT NOT NULL REFERENCES content_strategy(strategy_id),
    metric TEXT NOT NULL,
    target_value REAL CHECK (target_value >= 0),
    period_start TEXT NOT NULL,
    period_end TEXT NOT NULL CHECK (period_end >= period_start),
    rationale TEXT NOT NULL
);

