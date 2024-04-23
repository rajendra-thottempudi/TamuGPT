import re
from langchain.text_splitter import CharacterTextSplitter
from transformers import AutoTokenizer, AutoModel
import torch
from pinecone import Pinecone
import configparser

def loadData(path):
  # Path to your text file in Google Drive
  file_path = '/content/drive/My Drive/DirectedStudies/data.txt'
  # Read the text file
  with open(file_path, 'r') as file:
    raw_text = file.read()
  # Remove extra spaces and newlines
  cleaned_text = re.sub(r'\s+', '  ', raw_text)
  return cleaned_text

def splitDataInChunks(data):
  text_splitter = CharacterTextSplitter(
    separator = ".",
    chunk_size = 200,
    chunk_overlap = 100,
    length_function = len,
  )

  texts = text_splitter.split_text(data)
  return texts

def generateEmbeddings(texts):
  # Load model from HuggingFace Hub
  tokenizer = AutoTokenizer.from_pretrained('BAAI/bge-small-en-v1.5')
  model = AutoModel.from_pretrained('BAAI/bge-small-en-v1.5')
  model.eval()
  # Tokenize sentences
  encoded_input = tokenizer(texts, padding=True, truncation=True, return_tensors='pt')
  # for s2p(short query to long passage) retrieval task, add an instruction to query (not add instruction for passages)
  # encoded_input = tokenizer([instruction + q for q in queries], padding=True, truncation=True, return_tensors='pt')
  # Compute token embeddings
  with torch.no_grad():
    model_output = model(**encoded_input)
    # Perform pooling. In this case, cls pooling.
    sentence_embeddings = model_output[0][:, 0]
  # normalize embeddings
  sentence_embeddings = torch.nn.functional.normalize(sentence_embeddings, p=2, dim=1)
  # print("Sentence embeddings:", sentence_embeddings)
  return sentence_embeddings

def getPineConeIndex():
  # config = configparser.ConfigParser()
  # config.read('/config.json')
  # Get apiKey and index_name from config file
  apiKey = "6aa7b0bc-9d95-46c4-b9e1-88b3b659a554"
  index_name = "tamugpt"
  pc = Pinecone(api_key=apiKey)
  index = pc.Index(index_name)
  return index

def uploadDatatoDB(index, data, sentence_embeddings):
  data = [{"id": str(i), "values": embedding} for i, embedding in enumerate(sentence_embeddings)]
  index.upsert(data)
  index.describe_index_stats()

def getKClosest(question):
  db_index = getPineConeIndex()
  query_vector = [question]
  embed = generateEmbeddings(query_vector).tolist()
  closest = db_index.query(vector = embed, top_k = 10, include_metadata=True)  
  # print(closest)
  closest_strings = []
  
  for result in closest['matches']:
    text = result['metadata']['text']
    closest_strings.append(text);
    # print(f"id : {result['id']}")

  return closest_strings


def upload(path):
  # Read config file
  config = configparser.ConfigParser()
  config.read('/config.json')
  # Get apiKey and index_name from config file
  apiKey = config.get('DEFAULT', 'apiKey')
  index_name = config.get('DEFAULT', 'index_name')
  # Load data
  data = loadData(path)
  # Split data into chunks
  chunks = splitDataInChunks(data)
  # Get PineCone index
  db_index = getPineConeIndex(apiKey, index_name)
  embeddings = generateEmbeddings(chunks);
  uploadDatatoDB(db_index, index_name, embeddings);



