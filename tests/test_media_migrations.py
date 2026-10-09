import sqlite3
import tempfile
import unittest
from contextlib import closing
from pathlib import Path

from media_operations.persistence.migrate import MigrationError, migrate


class MediaMigrationTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.database = self.root / "media.sqlite3"
        self.migrations = self.root / "migrations"
        self.migrations.mkdir()

    def write_migration(self, name, script):
        (self.migrations / name).write_text(script, encoding="utf-8")

    def test_fresh_database_and_repeated_migration(self):
        migrate(self.database)
        migrate(self.database)
        with closing(sqlite3.connect(self.database)) as connection, connection:
            self.assertEqual(connection.execute("SELECT version FROM schema_migrations ORDER BY version").fetchall(), [(1,), (2,), (3,)])
            self.assertEqual(connection.execute("PRAGMA journal_mode").fetchone()[0], "wal")
            names = {row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")}
            self.assertTrue({"media_account", "account_revision", "account_goal", "agent_run", "agent_task", "run_event"} <= names)
            self.assertNotIn("memories", names)

    def test_incremental_upgrade_preserves_existing_data(self):
        self.write_migration("0001_first.sql", "CREATE TABLE sample (value TEXT);\n")
        migrate(self.database, self.migrations)
        with closing(sqlite3.connect(self.database)) as connection, connection:
            connection.execute("INSERT INTO sample VALUES ('preserved')")
        self.write_migration("0002_second.sql", "ALTER TABLE sample ADD COLUMN extra TEXT;\n")
        migrate(self.database, self.migrations)
        with closing(sqlite3.connect(self.database)) as connection, connection:
            self.assertEqual(connection.execute("SELECT * FROM sample").fetchall(), [("preserved", None)])

    def test_failed_upgrade_rolls_back_ddl_and_version(self):
        self.write_migration("0001_first.sql", "CREATE TABLE sample (value TEXT);\n")
        migrate(self.database, self.migrations)
        self.write_migration("0002_broken.sql", "CREATE TABLE transient (value TEXT);\nINSERT INTO missing VALUES (1);\n")
        with self.assertRaises(sqlite3.OperationalError):
            migrate(self.database, self.migrations)
        with closing(sqlite3.connect(self.database)) as connection, connection:
            self.assertIsNone(connection.execute("SELECT name FROM sqlite_master WHERE name='transient'").fetchone())
            self.assertEqual(connection.execute("SELECT version FROM schema_migrations").fetchall(), [(1,)])
        self.write_migration("0002_broken.sql", "CREATE TABLE transient (value TEXT);\n")
        migrate(self.database, self.migrations)

    def test_checksum_protects_already_applied_migrations(self):
        self.write_migration("0001_first.sql", "CREATE TABLE sample (value TEXT);\n")
        migrate(self.database, self.migrations)
        self.write_migration("0001_first.sql", "CREATE TABLE sample (changed TEXT);\n")
        with self.assertRaises(MigrationError):
            migrate(self.database, self.migrations)

    def test_missing_or_non_contiguous_history_is_rejected(self):
        self.write_migration("0002_only.sql", "CREATE TABLE sample (value TEXT);\n")
        with self.assertRaises(MigrationError):
            migrate(self.database, self.migrations)
        self.write_migration("0001_first.sql", "CREATE TABLE first (value TEXT);\n")
        migrate(self.database, self.migrations)
        (self.migrations / "0002_only.sql").unlink()
        with self.assertRaises(MigrationError):
            migrate(self.database, self.migrations)

    def test_incomplete_sql_is_not_partially_applied(self):
        self.write_migration("0001_first.sql", "CREATE TABLE sample (value TEXT);\nCREATE TABLE incomplete (")
        with self.assertRaises(MigrationError):
            migrate(self.database, self.migrations)
        with closing(sqlite3.connect(self.database)) as connection, connection:
            self.assertIsNone(connection.execute("SELECT name FROM sqlite_master WHERE name='sample'").fetchone())


if __name__ == "__main__":
    unittest.main()
