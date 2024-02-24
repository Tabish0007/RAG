# RAG
RAG
    Install the necessary libraries:

!pip install pypdf
!pip install -q transformers einops accelerate langchain bitsandbytes
!pip install sentence_transformers
!pip install llama-index
!pip install chromadb
    Import the libraries :

import logging
import sys

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

from llama_index import VectorStoreIndex, SimpleDirectoryReader, ServiceContext
from llama_index.llms import HuggingFaceLLM
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from llama_index import LangchainEmbedding, ServiceContext
from llama_index.prompts.prompts import SimpleInputPrompt
from transformers import AutoTokenizer, FalconForCausalLM
import transformers
import torch

Simply just after downloading all dependencies click on the streamlit application upload the pdf and run it cheers!!
