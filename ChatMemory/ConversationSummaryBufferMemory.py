from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.memory import ChatMessageHistory
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()
api_key = os.getenv('TOGETHER_API_KEY')

# Initialize the model with Together AI's endpoint
llm = ChatOpenAI(
    model="mistralai/Mixtral-8x7B-Instruct-v0.1",
    api_key=api_key,
    base_url="https://api.together.xyz/v1",
    temperature=1
)

# Create a prompt template
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful AI assistant with a good memory of our conversation."),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}")
])

# Create the chain
chain = prompt | llm

# Initialize chat history
history = ChatMessageHistory()

# Create chain with history
chain_with_history = RunnableWithMessageHistory(
    chain,
    lambda session_id: history,
    input_messages_key="input",
    history_messages_key="history"
)

# Define session ID
session_id = "user_123"

# Series of interactions
print("First interaction:")
print(chain_with_history.invoke(
    {"input": "I am writing a book about mathematics, can you explain what is mathematics?"},
    config={"configurable": {"session_id": session_id}}
).content)

print("\nSecond interaction:")
print(chain_with_history.invoke(
    {"input": "I am writing a book and I live in Los Angeles"},
    config={"configurable": {"session_id": session_id}}
).content)

print("\nThird interaction:")
print(chain_with_history.invoke(
    {"input": "My name is Karel, what is the book I am writing about?"},
    config={"configurable": {"session_id": session_id}}
).content)

print("\nFinal question:")
print(chain_with_history.invoke(
    {"input": "What did we talk about?"},
    config={"configurable": {"session_id": session_id}}
).content)