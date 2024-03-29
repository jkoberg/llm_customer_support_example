import json
import sqlite3
from typing import Iterable

from message_store.message_store import MessageStore
from models import RawMessage, StoredMessage


class SqliteMessageStore(MessageStore):
    initial_sql = """
    CREATE TABLE IF NOT EXISTS Conversations(
        conversation_id TEXT,
        UNIQUE (conversation_id)
    );
    CREATE TABLE IF NOT EXISTS Messages(
        message_id TEXT,
        conversation_id TEXT,
        timestamp TEXT,
        channel TEXT,
        party TEXT,
        message TEXT,
        UNIQUE (message_id)
    );
    CREATE TABLE IF NOT EXISTS LLMResponses(
        conversation_id TEXT,
        model_version TEXT,
        response JSONB,
        UNIQUE (conversation_id, model_version)
    );    
    """

    def __init__(self, db_path: str) -> None:
        self.conn = sqlite3.connect(db_path)
        with self.conn:
            self.conn.execute(self.initial_sql)

    def insert_conversation_id(self, conversation_id: str) -> None:
        self.conn.execute(
            "INSERT OR IGNORE INTO Conversations (conversation_id) VALUES (?)",
            (conversation_id,)
        )

    def add_message(self, msg: RawMessage):
        with self.conn:
            self.insert_conversation_id(msg.conversation_id)
            self.conn.execute(
                "INSERT OR IGNORE INTO Messages(message_id, conversation_id, timestamp, channel, party, message) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (msg.id, msg.conversation_id, msg.created_at, msg.channel, msg.party, msg.sanitized)
            )

    def add_llm_response(self, conversation_id: str, model: str, llm_response: object):
        with self.conn:
            self.insert_conversation_id(conversation_id)
            self.conn.execute(
                "INSERT OR IGNORE INTO LLMResponses(conversation_id, model_version, response) VALUES (?, ?, ?)",
                (conversation_id, model, json.dumps(llm_response))
            )

    def get_llm_response(self, conversation_id: str) -> object:
        curs = self.conn.execute(
            "select response from LLMResponses where conversation_id = ? LIMIT 1",
            (conversation_id,)
            )
        row = curs.fetchone()
        if row:
            return json.loads(row[0])

    def get_conversation_ids(self) -> Iterable[str]:
        with self.conn:
            for row in self.conn.execute("SELECT conversation_id FROM Conversations"):
                yield row[0]

    def get_messages_for_conversation(self, conversation_id: str) -> Iterable[StoredMessage]:
        for row in self.conn.execute("""
            SELECT message_id, conversation_id, timestamp, channel, party, message 
            FROM Messages 
            WHERE conversation_id = ? 
            ORDER BY timestamp
            """, (conversation_id,)
        ):
            yield StoredMessage(*row)

    def get_evaluated_conversations(self):
        with self.conn:
            for row in self.conn.execute("SELECT conversation_id, response from LLMResponses"):
                conversation_id = row[0]
                messages = list(self.get_messages_for_conversation(conversation_id))
                yield (conversation_id, messages, json.loads(row[1]))
