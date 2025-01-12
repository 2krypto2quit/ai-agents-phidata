from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from dotenv import load_dotenv
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

# Create the prompt template with history
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful AI assistant."),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}")
])

# Create the basic chain
chain = prompt | model

# Initialize chat history
history = ChatMessageHistory()

# Create chain with history
chain_with_history = RunnableWithMessageHistory(
    chain,
    lambda session_id: history,
    input_messages_key="input",
    history_messages_key="history"
)

# Define a session_id
session_id = "user_123"

# Test the chain with multiple interactions
print("\nFirst interaction:")
print(chain_with_history.invoke(
    {"input": "Hi, my name is Bob"},
    config={"configurable": {"session_id": session_id}}
).content)

print("\nSecond interaction:")
print(chain_with_history.invoke(
    {"input": "What's my name?"},
    config={"configurable": {"session_id": session_id}}
).content)