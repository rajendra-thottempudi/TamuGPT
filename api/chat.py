from flask import Flask, jsonify, request
from flask_cors import CORS, cross_origin
from flask import jsonify
from flask_cors import CORS
from urllib.parse import unquote
# from llm import ask
import openai  # for calling the OpenAI API
import pandas as pd  # for storing text and embeddings data
import tiktoken  # for counting tokens
import os
from pinecone import Pinecone
import re



app = Flask(__name__)
cors = CORS(app, resources={r"/*": {"origins": "*"}})
app.config['CORS_HEADERS'] = 'Content-Type'


@app.route("/")
@cross_origin()
def home():
    return "Flask Vercel Example - Hello World", 200


def getEmbedding(sentence):
  secret_key = os.environ.get('SECRET_KEY')
  openai.api_key = secret_key
  query_embedding_response = openai.embeddings.create(
        model="text-embedding-3-small",
        input= sentence,
  )

  q_embed = query_embedding_response.data[0].embedding
  return q_embed


def getPineConeIndex():
  apiKey = "6aa7b0bc-9d95-46c4-b9e1-88b3b659a554"
  index_name = "tamuembed"
  pc = Pinecone(api_key=apiKey)
  index = pc.Index(index_name)
  return index

def getKClosest(question):
  db_index = getPineConeIndex()
  query_vector = [question]
  embed = getEmbedding(query_vector)
  closest = db_index.query(vector = embed, top_k = 20, include_metadata=True)  
  # print(closest)
  closest_strings = []
  
  for result in closest['matches']:
    text = result['metadata']['text']
    closest_strings.append(text)
    # print(f"id : {result['id']}")

  return closest_strings

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

# Route to get a string as a response after taking input from the user
@app.route('/<message>', methods=['GET'])
@cross_origin()
def get_book(message: str):
    message = unquote(message)
    response = ask(message)
    return jsonify({"message": response})

if __name__ == '__main__':
    app.run(port=8080)
