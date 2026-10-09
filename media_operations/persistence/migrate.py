"""有版本和校验和的事务迁移；不使用会隐式提交的 executescript。"""

import hashlib
import sqlite3
from datetime import UTC, datetime
from pathlib import Path


MIGRATIONS_DIR = Path(__file__).with_name("migrations")


class MigrationError(RuntimeError):
    pass


def open_database(path: str | Path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path, timeout=30, isolation_level=None)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    connection.execute("PRAGMA busy_timeout = 30000")
    return connection


def _statements(script):
    pending = ""
    for line in script.splitlines(keepends=True):
        pending += line
        if sqlite3.complete_statement(pending):
            yield pending
            pending = ""
    if pending.strip() and any(
        line.strip() and not line.lstrip().startswith("--") for line in pending.splitlines()
    ):
        raise MigrationError("迁移包含不完整 SQL 语句")


def migrate(path: str | Path, migrations_dir: Path = MIGRATIONS_DIR):
    files = sorted(migrations_dir.glob("[0-9][0-9][0-9][0-9]_*.sql"))
    versions = [int(file.name.split("_", 1)[0]) for file in files]
    if not files or versions != list(range(1, len(files) + 1)):
        raise MigrationError("迁移版本必须从 0001 开始连续且不重复")
    connection = open_database(path)
    try:
        connection.execute("PRAGMA journal_mode = WAL")
        connection.execute("BEGIN IMMEDIATE")
        connection.execute("""CREATE TABLE IF NOT EXISTS schema_migrations (
            version INTEGER PRIMARY KEY, name TEXT NOT NULL,
            checksum TEXT NOT NULL, applied_at TEXT NOT NULL
        )""")
        applied = {row["version"]: row for row in connection.execute("SELECT * FROM schema_migrations")}
        if sorted(applied) != versions[:len(applied)]:
            raise MigrationError("数据库迁移历史与当前代码版本不兼容")
        for version, file in zip(versions, files):
            # 统一换行，避免 Windows/Linux 检出换行不同造成校验和漂移。
            script = file.read_text(encoding="utf-8")
            checksum = hashlib.sha256(script.encode("utf-8")).hexdigest()
            if version in applied:
                if applied[version]["checksum"] != checksum or applied[version]["name"] != file.name:
                    raise MigrationError(f"已应用的迁移被修改：{file.name}")
                continue
            for statement in _statements(script):
                connection.execute(statement)
            connection.execute(
                "INSERT INTO schema_migrations VALUES (?, ?, ?, ?)",
                (version, file.name, checksum, datetime.now(UTC).isoformat()),
            )
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()

