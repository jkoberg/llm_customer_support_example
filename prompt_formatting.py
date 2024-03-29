from typing import List, Iterable
from openai.types.chat import ChatCompletionMessageParam

from models import RawMessage, StoredMessage


def format_raw_messages(messages: Iterable[StoredMessage]) -> Iterable[str]:
    "Format a raw conversation message into a more suitable form for a ChatGPT prompt."
    for message in messages:
        yield """{} (via {}) at {}: {}""".format(
            message.party,
            message.channel,
            message.timestamp,
            message.sanitized.replace("\n", " ").replace("\r", "")
        )


def format_prompt(messages: Iterable[StoredMessage]) -> List[ChatCompletionMessageParam]:
    """
    Format all the messages in a conversation into a Chat Completion request for the OpenAI API.

    This will include the system prompt and other instructions to return specific formatted JSON
    that can drive further behavior from our system.
    """
    conversation_body = "\n\n".join(format_raw_messages(messages))
    return [
        {
            "role": "system",
            "content": "You will analyze transcripts of interactions with car dealership customers and answer " +
                       "specific questions about the transcripts.  You must respond with JSON formatted output."
        },
        {
            "role": "user",
            "content": """
                The transcript with the customer is delimited by triple quotes. Please reply with JSON object containing the following fields: 
                
                "communication_quality": On a scale from 1 to 5, with 1 being perfect and 5 being difficult, how much effort did the customer have to put into being understood or getting a relevant reply?

                "needs_sales": whether the request needs attention from the sales department; true or false.

                "needs_service": whether the request needs attention from a customer service agent; true or false.
                
                "needs_scheduling": True or False, whether the customer needs to schedule or reschedule a class.

                "needs_manager": whether the request needs attention from a manager. true or false.

                "is_urgent": whether the request is time-sensitive; true or false.

                "is_closed": whether the request requires further assistance, or has been satisfied. true or false.
                
                "customer_sentiment": A string describing the customer's expressed sentiment.  Either "very positive", "positive", "very negative", "negative", or "neutral".

                "customer_name": null if you cannot determine the customer name. Otherwise a string with the customer's name.

                "agent_name": null if you cannot determine the name of agent the customer needs to speak to. Otherwise a string with the agent's name.

                "required_info": null if no additional information is required from the customer.  Otherwise, a sentence prompting the customer for the needed information.

                "summary": Summarize the current action required, or resolved status, of this transcript in 15 words or less.

                "meta_improvements": A string, describing what information could have been provided, if any, to help you complete this task.
            """
        },
        {
            "role": "user",
            "content": '"""\n{}\n"""'.format(conversation_body)
        }
    ]

