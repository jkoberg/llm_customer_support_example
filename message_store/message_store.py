from abc import ABCMeta, abstractmethod
from typing import Iterable, Tuple

from models import RawMessage, StoredMessage, ConversationTranscript
from llm.model import LLMResponse


class MessageStore(metaclass=ABCMeta):
    @abstractmethod
    def add_message(self, message: RawMessage) -> None:
        raise NotImplementedError

    @abstractmethod
    def add_llm_response(self, conversation_id: str, model: str, llm_response: LLMResponse) -> None:
        raise NotImplementedError

    @abstractmethod
    def get_llm_response(self, conversation_id: str) -> LLMResponse:
        raise NotImplementedError

    @abstractmethod
    def get_conversation_ids(self) -> Iterable[str]:
        raise NotImplementedError

    @abstractmethod
    def get_transcript(self, conversation_id: str) -> ConversationTranscript:
        raise NotImplementedError

    @abstractmethod
    def get_evaluated_conversations(self) -> Iterable[Tuple[ConversationTranscript, LLMResponse]]:
        raise NotImplementedError
