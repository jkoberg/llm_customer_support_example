from __future__ import annotations

from abc import ABCMeta
from dataclasses import dataclass

from models import ConversationTranscript


class ConversationLLMClient(metaclass=ABCMeta):
    def query(self, conversation: ConversationTranscript) -> LLMResponse:
        raise NotImplementedError


class LLMClientError(Exception):
    def __init__(self, message: str):
        self.message = message


@dataclass
class LLMResponse:
    summary: str
    communication_quality: int
    needs_sales: bool
    needs_service: bool
    needs_scheduling: bool
    needs_manager: bool
    is_urgent: bool
    is_closed: bool
    customer_sentiment: str
    customer_name: str
    agent_name: str
    required_info: str
    meta_improvements: str
