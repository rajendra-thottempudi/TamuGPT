from data import getKClosest
import openai  # for calling the OpenAI API
import pandas as pd  # for storing text and embeddings data
import tiktoken  # for counting tokens
import os

def num_tokens(text: str, model: str = "gpt-3.5-turbo") -> int:
    """Return the number of tokens in a string."""
    encoding = tiktoken.encoding_for_model(model)
    # print(len(encoding.encode(text)))
    return len(encoding.encode(text))

def query_message(
    query: str,
    model: str,
    token_budget: int
) -> str:
    """Return a message for GPT, with relevant source texts pulled from a dataframe."""
    closest_strings = getKClosest(query)
    introduction = 'Use the below Texas A&M Information to answer the subsequent question. If the answer cannot be found in the article, write "I could not find an answer."'
    question = f"\n\nQuestion: {query}"
    message = introduction
    for string in closest_strings:
        next_article = f'\n\n TexasA&M Information :\n"""\n{string}\n"""'
        if (
            num_tokens(message + next_article + question, model=model)
            > token_budget
        ):
            break
        else:
            message += next_article
    return message + question


def ask(
    query: str,
    model: str = "gpt-3.5-turbo",
    token_budget: int = 4000,
    print_message: bool = False,
) -> str:
    """Answers a query using GPT and a dataframe of relevant texts and embeddings."""
    message = query_message(query, model=model, token_budget=token_budget)
    if print_message:
        print(message)
    
    secret_key = os.environ.get('SECRET_KEY')
    openai.api_key = secret_key
    messages = [
        {"role": "system", "content": "You answer questions about Customer Service Management, ServiceNow."},
        {"role": "user", "content": message},
    ]
    response = openai.chat.completions.create(
        model=model,
        messages=messages,
        temperature = 0,
        max_tokens = 2000,
    )
    # print(response)
    response_message = response.choices[0].message.content
    return response_message