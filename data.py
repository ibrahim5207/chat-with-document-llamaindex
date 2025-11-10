from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext, load_index_from_storage
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding
import os
from llama_index.core import Settings
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
# Local Embeddings (No API key needed)
Settings.embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Local LLM (Ollama model like llama3, mistral, etc.)
Settings.llm = Ollama(model="llama3", request_timeout=120)

# === Setup paths ===
DATA_DIR = "data"
STORAGE_DIR = "./storage"

# === Use local Ollama models ===
llm = Ollama(model="llama3")
embed_model = OllamaEmbedding(model_name="nomic-embed-text")

# === Create or load index ===
if not os.path.exists(STORAGE_DIR):
    print("Creating new index from local documents...")
    documents = SimpleDirectoryReader(DATA_DIR).load_data()
    index = VectorStoreIndex.from_documents(documents, llm=llm, embed_model=embed_model)
    index.storage_context.persist(persist_dir=STORAGE_DIR)
    print("Index created and saved locally!")
else:
    print("Loading existing local index...")
    storage_context = StorageContext.from_defaults(persist_dir=STORAGE_DIR)
    index = load_index_from_storage(storage_context, llm=llm, embed_model=embed_model)
    print("Index loaded successfully!")

# === Query system ===
query_engine = index.as_query_engine(similarity_top_k=3)

print("\nLocal Knowledge Chatbot (type 'exit' to quit)\n")
while True:
    query = input("You: ")
    if query.lower() in ["exit", "quit"]:
        print("Goodbye!")
        break
    response = query_engine.query(query)
    print("Bot:", response, "\n")
    DATA_DIR = r"C:\Users\Student\Documents\myfiles"
DATA_DIR = r"C:\Users\Student\Documents\myfiles"
DATA_DIR = r"C:\Users\Student\test\project 2\data"
git add .
