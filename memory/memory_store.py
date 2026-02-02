import sqlite3
import json
from typing import Dict, Any

DB_NAME = "rag_memory.db"

class MemoryStore:
    def __init__(self, db_name: str = DB_NAME):
        self.db_name = db_name
        self._init_db()

    def _get_conn(self):
        conn = sqlite3.connect(self.db_name)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self):
        conn = self._get_conn()
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS conversation_state (
                conversation_id TEXT PRIMARY KEY,
                state_json TEXT,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        conn.commit()
        conn.close()

    # --------------------------------------------------
    # API pública
    # --------------------------------------------------
    def get_state(self, conversation_id: str) -> Dict[str, Any]:
        conn = self._get_conn()
        cur = conn.cursor()
        cur.execute(
            "SELECT state_json FROM conversation_state WHERE conversation_id = ?",
            (conversation_id,),
        )
        row = cur.fetchone()
        conn.close()

        if not row:
            return {}

        try:
            return json.loads(row["state_json"])
        except Exception:
            return {}

    def update_state(self, conversation_id: str, patch: Dict[str, Any]):
        current = self.get_state(conversation_id)
        current.update(patch)

        conn = self._get_conn()
        conn.execute(
            """
            INSERT INTO conversation_state (conversation_id, state_json)
            VALUES (?, ?)
            ON CONFLICT(conversation_id)
            DO UPDATE SET
                state_json = excluded.state_json,
                updated_at = CURRENT_TIMESTAMP
            """,
            (conversation_id, json.dumps(current)),
        )
        conn.commit()
        conn.close()
