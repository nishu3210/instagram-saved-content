"""Database migration utilities."""

import logging
from datetime import datetime, timezone

from database import db
from sqlalchemy import inspect, text

logger = logging.getLogger(__name__)


class DatabaseMigration:
    """Handle database migrations."""

    VERSION_TABLE = "schema_version"

    def __init__(self):
        self.connection = db.engine.connect()

    def init_version_table(self):
        """Initialize version tracking table."""
        self.connection.execute(
            text(f"""
            CREATE TABLE IF NOT EXISTS {self.VERSION_TABLE} (
                version INTEGER PRIMARY KEY,
                applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                description TEXT
            )
        """)
        )
        self.connection.commit()

    def get_current_version(self) -> int:
        """Get current schema version."""
        try:
            result = self.connection.execute(
                text(f"SELECT MAX(version) FROM {self.VERSION_TABLE}")
            )
            version = result.scalar()
            return version or 0
        except Exception:
            return 0

    def migrate(self):
        """Run all pending migrations."""
        self.init_version_table()
        current = self.get_current_version()

        migrations = [
            (1, "Add timezone support", self._migrate_v1),
            (2, "Add performance indexes", self._migrate_v2),
            (3, "Normalize JSON fields", self._migrate_v3),
        ]

        for version, description, migration_func in migrations:
            if version > current:
                logger.info(f"Applying migration {version}: {description}")
                try:
                    migration_func()
                    self._record_migration(version, description)
                    logger.info(f"Migration {version} complete")
                except Exception as e:
                    logger.error(f"Migration {version} failed: {e}")
                    raise

    def _record_migration(self, version: int, description: str):
        """Record successful migration."""
        self.connection.execute(
            text(f"""
                INSERT INTO {self.VERSION_TABLE} (version, description)
                VALUES (:version, :description)
            """),
            {"version": version, "description": description},
        )
        self.connection.commit()

    def _migrate_v1(self):
        """Add timezone support to datetime columns."""
        tables = ["posts", "analysis", "action_tasks", "conversations", "messages"]
        for table in tables:
            try:
                self.connection.execute(
                    text(f"""
                        ALTER TABLE {table} 
                        ADD COLUMN IF NOT EXISTS created_at_tz TIMESTAMP
                    """)
                )
            except Exception as e:
                logger.warning(f"Could not migrate {table}: {e}")
        self.connection.commit()

    def _migrate_v2(self):
        """Add performance indexes."""
        indexes = [
            "CREATE INDEX IF NOT EXISTS idx_posts_username ON posts(username)",
            "CREATE INDEX IF NOT EXISTS idx_analysis_category ON analysis(category)",
            "CREATE INDEX IF NOT EXISTS idx_tasks_status_due ON action_tasks(status, due_date)",
        ]
        for idx_sql in indexes:
            try:
                self.connection.execute(text(idx_sql))
            except Exception as e:
                logger.warning(f"Could not create index: {e}")
        self.connection.commit()

    def _migrate_v3(self):
        """Normalize JSON fields - clean up double-encoded data."""
        try:
            # Get all analysis rows
            result = self.connection.execute(
                text("SELECT id, topics, learning_points, action_items FROM analysis")
            )
            rows = result.fetchall()

            for row in rows:
                row_id = row[0]
                fields = {
                    "topics": row[1],
                    "learning_points": row[2],
                    "action_items": row[3],
                }
                updates = {}

                for field_name, value in fields.items():
                    if value and isinstance(value, str):
                        # Try to decode and re-encode properly
                        try:
                            import json

                            decoded = json.loads(value)
                            if isinstance(decoded, str):
                                decoded = json.loads(decoded)  # Double-encoded
                            if isinstance(decoded, list):
                                updates[field_name] = json.dumps(
                                    decoded, ensure_ascii=False
                                )
                        except (json.JSONDecodeError, TypeError):
                            # Keep as-is if can't decode
                            pass

                if updates:
                    set_clause = ", ".join([f"{k} = :{k}" for k in updates.keys()])
                    updates["id"] = row_id
                    self.connection.execute(
                        text(f"UPDATE analysis SET {set_clause} WHERE id = :id"),
                        updates,
                    )

            self.connection.commit()
        except Exception as e:
            logger.error(f"JSON normalization failed: {e}")


def run_migrations():
    """Run all migrations."""
    migrator = DatabaseMigration()
    migrator.migrate()
