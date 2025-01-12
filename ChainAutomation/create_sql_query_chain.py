from langchain_openai import ChatOpenAI
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_sql_query_chain
from langchain_community.utilities import SQLDatabase
from dotenv import load_dotenv
import os

# Set up the database connection
db = SQLDatabase.from_uri("sqlite:///Chinook.db")

# Initialize the model with Together AI's endpoint
llm = ChatOpenAI(
    model="mistralai/Mixtral-8x7B-Instruct-v0.1",
    api_key=os.getenv('TOGETHER_API_KEY'),
    base_url="https://api.together.xyz/v1",
    temperature=0  # Keeping temperature at 0 for consistent SQL generation
)

# Create the SQL query chain
chain = create_sql_query_chain(llm, db)

# Execute the chain
response = chain.invoke({"question": "How many employees are there"})
print(response)
