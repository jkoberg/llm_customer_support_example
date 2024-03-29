import pprint
from itertools import islice

from llm.model import ConversationLLMClient, LLMClientError
from llm.openai import OpenAIClient, OpenAIConversationClient
from message_store.in_memory_message_store import InMemoryMessageStore
from message_store.message_store import MessageStore
from prompt_formatting import format_raw_messages
from parse_csv import *


def main(store: MessageStore, llm_client: ConversationLLMClient, input_file: str, sample_count: 10):
    """
    Load all the conversations,
    Then, for each conversation, format the message thread for ChatGPT, and query it to produce a JSON response
    Store all the responses for which GPT "stopped" in the `succeeded_responses` dictionary.
    Store all the failed responses (over-length or another reason) in the `failed_responses` dictionary.
    """

    # Read CSV file to gather conversation threads.
    for raw_message in read_csv(input_file):
        store.add_message(raw_message)

    # Select some conversations to pass through the LLM model.
    sample_conversation_ids = islice(store.get_conversation_ids(), 0, sample_count)
    sample_conversations = (store.get_transcript(cid) for cid in sample_conversation_ids)

    # Query the LLM client with the conversations
    for transcript in sample_conversations:
        try:
            response = llm_client.query(transcript)
            store.add_llm_response(transcript.conversation_id, "", response)
        except LLMClientError as e:
            print("Unable to query LLM for conversation {}: {}".format(transcript.conversation_id, e.message))

    # Print out what we found for examination
    for transcript, llm_response in store.get_evaluated_conversations():
        conversation_id = transcript.conversation_id
        thread_text: str = '\n'.join(format_raw_messages(transcript.messages))
        print("""\n\nMessage thread for conversation ID {}:\n{}""".format(conversation_id, thread_text))
        print("""For conversation ID {}, we determined the following:\n{}\n""".format(
            conversation_id,
            pprint.pformat(llm_response)
        ))


if __name__ == "__main__":
    store = InMemoryMessageStore()
    llm_client = OpenAIConversationClient(OpenAIClient())
    input_file = "conversations.csv"
    main(store, llm_client, input_file, sample_count=10)


