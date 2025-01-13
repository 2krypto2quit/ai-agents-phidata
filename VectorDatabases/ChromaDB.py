from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings  # Updated import
from langchain_community.vectorstores import Chroma 
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
import os

# Load environment variables
load_dotenv()
api_key = os.getenv('TOGETHER_API_KEY')

# Initialize the model with Together AI's endpoint
model = ChatOpenAI(
    model="mistralai/Mixtral-8x7B-Instruct-v0.1",
    api_key=api_key,
    base_url="https://api.together.xyz/v1",
    temperature=0.7
)

# Load documents
loader = DirectoryLoader("C:/documents") 
documents = loader.load()

# Split text into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=500
)
chunks = text_splitter.split_documents(documents)

# Initialize embeddings with updated import
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2"
)

# Create vector store
vector_store = Chroma.from_documents(chunks, embeddings)

# Query the vector store
query = "Explain the dolphins behavior in the wild"
results = vector_store.similarity_search(query, k=5)

# Print results
for i, result in enumerate(results):
    print(f"Result {i + 1}:\n{result.page_content}\n")