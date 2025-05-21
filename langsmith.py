from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv
load_dotenv()

llm=ChatGroq(groq_api_key=os.environ['GROQ_API_KEY'],model='llama3-8b-8192')

#import traceable
from langsmith import Client, traceable

import numpy as np

client=Client()
text=['I love you','I love everything','I love my family','Shifting is tedious','My favourite color is blue']
@traceable(client=client,name='Ollama_embed',run_type='embedding',metadata={'dimensions':768})
def get_embedding(text):
    chunking=RecursiveCharacterTextSplitter(chunk_size=500,chunk_overlap=100)
    chunks=chunking.create_documents(text)
    embeddings=OllamaEmbeddings(model='nomic-embed-text')
    vector=np.array(embeddings)
    return vector

get_embedding(text)

