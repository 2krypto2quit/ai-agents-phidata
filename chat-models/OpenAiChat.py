from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

def setup_chat():
    # Initialize the ChatOpenAI object
    chat = ChatOpenAI(
        temperature=0.7,
        model_name="gpt-3.5-turbo"
    )
    return chat

def chat_loop(chat):
    # Initial system message to set the context
    messages = [
        SystemMessage(content="You are a helpful assistant.")
    ]
    
    print("Welcome to the chat! Type 'quit' to exit.")
    
    while True:
        # Get user input
        user_input = input("\nYou: ")
        
        # Check for quit command
        if user_input.lower() == 'quit':
            print("Goodbye!")
            break
        
        # Add user message to the messages list
        messages.append(HumanMessage(content=user_input))
        
        try:
            # Get the response from the chat model
            response = chat(messages)
            
            # Print the assistant's response
            print("\nAssistant:", response.content)
            
            # Add the assistant's response to the messages list
            messages.append(response)
            
        except Exception as e:
            print(f"An error occurred: {str(e)}")

def main():
    # Check if API key is set
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY not found in environment variables")
        return
        
    chat = setup_chat()
    chat_loop(chat)

if __name__ == "__main__":
    main()