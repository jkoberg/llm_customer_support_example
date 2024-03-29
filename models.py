from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime
from typing import Iterable


# Represents a message received or sent via SMS or phone, between a customer and agent.
@dataclass
class RawMessage:
    """
    Represents a row read from the conversations CSV. Note: These field names must match
    the column names in the CSV; since the object is constructed from a dictionary
    representing the row.
    """
    id: str
    conversation_id: str
    message_format: str
    is_inbound: str
    created_at: str
    sanitized: str


@dataclass
class StoredMessage:
    id: str
    conversation_id: str
    channel: str
    party: str
    timestamp: datetime
    sanitized: str

    @staticmethod
    def from_raw_message(m: RawMessage) -> StoredMessage:
        if m.message_format == "note":
            channel = "Agent Note"
        elif m.message_format == "media":
            channel = "Phone"
        elif m.message_format == "text":
            channel =  "SMS"
        else:
            channel = "Unknown"

        if m.is_inbound == "TRUE":
            party = "Customer"
        else:
            party = "Agent"

        return StoredMessage(
            m.id,
            m.conversation_id,
            channel,
            party,
            datetime.fromisoformat(m.created_at),
            m.sanitized
        )


@dataclass
class ConversationTranscript:
    conversation_id: str
    messages: Iterable[StoredMessage]