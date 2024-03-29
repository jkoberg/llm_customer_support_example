import json
from json import JSONDecodeError
from typing import List

import openai
from openai.types.chat import ChatCompletionMessageParam, ChatCompletion

from llm.model import ConversationLLMClient, LLMClientError, LLMResponse
from models import ConversationTranscript
from prompt_formatting import format_prompt


class OpenAIClient:
    """Holds an instantiated OpenAI API client, and allows queries against it."""
    def __init__(self, model="gpt-4-turbo-preview", keyfile="openai_key.txt"):
        api_key = open(keyfile, "r").read().strip()
        self.model_name = model
        self.client = openai.OpenAI(api_key=api_key)

    def query(self, inputs: List[ChatCompletionMessageParam]) -> ChatCompletion:
        response = self.client.chat.completions.create(
            model=self.model_name,
            response_format={"type": "json_object"},
            messages=inputs
        )
        return response


class OpenAIConversationClient(ConversationLLMClient):
    def __init__(self, client: OpenAIClient):
        self.client = client

    def query(self, transcript: ConversationTranscript):
        inputs = format_prompt(transcript.messages)
        response = self.client.query(inputs)
        for choice in response.choices:
            if choice.finish_reason == "stop":
                try:
                    return LLMResponse(**json.loads(choice.message.content))
                except JSONDecodeError as e:
                    raise LLMClientError("ERROR in LLM response JSON decode: {}".format(e.msg))
            else:
                return None
