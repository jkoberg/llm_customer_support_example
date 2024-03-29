from collections import defaultdict
from collections.abc import Iterable
from typing import Dict
from sortedcontainers import SortedList

from message_store.message_store import MessageStore
from models import RawMessage, StoredMessage, ConversationTranscript


class InMemoryMessageStore(MessageStore):
    """Stores parsed messages by Conversation ID, sorted by timestamp.
       This is assumed to yield them in correct, threaded order.
    """
    def __init__(self):
        self.messages_by_conversation: Dict[str, SortedList[StoredMessage]] = defaultdict(lambda: SortedList(key=lambda msg: msg.timestamp))
        self.llm_responses: Dict[str, (str, object)] = {}

    def add_message(self, m: RawMessage):
        self.messages_by_conversation[m.conversation_id].add(StoredMessage.from_raw_message(m))

    def add_llm_response(self, conversation_id: str, model: str, llm_response: object):
        self.llm_responses[conversation_id] = (model, llm_response)

    def get_llm_response(self, conversation_id: str) -> object:
        return self.llm_responses[conversation_id]

    def get_conversation_ids(self) -> Iterable[str]:
        return self.messages_by_conversation.keys()

    def get_transcript(self, conversation_id: str):
        messages = self.messages_by_conversation[conversation_id]
        return ConversationTranscript(conversation_id, messages)

    def get_evaluated_conversations(self):
        for (conversation_id, llm_response) in self.llm_responses.items():
            transcript = self.get_transcript(conversation_id)
            yield (transcript, llm_response)



