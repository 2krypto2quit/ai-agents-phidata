import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# Load environment variables
load_dotenv()
api_key = os.getenv('TOGETHER_API_KEY')

# Initialize the LLM with Together AI
llm = ChatOpenAI(
    model="mistralai/Mixtral-8x7B-Instruct-v0.1",
    api_key=api_key,
    base_url="https://api.together.xyz/v1",
    temperature=0.7
)

# Initialize memory
memory = ChatMessageHistory()
session_id = "user_123"

# Define the prompt
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful AI assistant."),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}")
])

# Create the chain
chain = prompt | llm

# Create the runnable with message history
runnable = RunnableWithMessageHistory(
    chain,
    lambda session_id: memory,
    input_messages_key="input",
    history_messages_key="history"
)

def process_chat(runnable, user_input):
    response = runnable.invoke(
        {"input": user_input},
        config={"configurable": {"session_id": session_id}}
    ).content
    return response

print("Agent is ready to chat. Type 'exit' or 'quit' to end the conversation.")

if __name__ == "__main__":
    while True:
        user_input = input("You: ")
        if user_input.lower() in ['exit', 'quit']:
            break
        response = process_chat(runnable, user_input)
        print(f"AI: {response}")