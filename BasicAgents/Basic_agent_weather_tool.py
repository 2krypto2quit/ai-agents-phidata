import os
from langchain_together import Together
from langchain.agents import create_openai_tools_agent
from langchain.prompts import ChatPromptTemplate
from langchain_community.agent_toolkits.load_tools import load_tools
from langchain.agents import AgentExecutor
from dotenv import load_dotenv
from langchain.schema.messages import HumanMessage, AIMessage
from typing import Any, Dict, List

load_dotenv()
together_api_key = os.getenv("TOGETHER_API_KEY")

# Initialize Together with chat configuration
llm = Together(
    model="mistralai/Mixtral-8x7B-Instruct-v0.1",
    together_api_key=together_api_key,
    temperature=0.7,
    max_tokens=1024,
    chat_format='chatml'  # Specify chat format
)

# Custom wrapper to make Together work with agent
class TogetherChatWrapper:
    def __init__(self, llm):
        self.llm = llm
    
    def predict_messages(self, messages: List[Any], **kwargs: Any) -> AIMessage:
        # Convert messages to a format Together can understand
        formatted_messages = []
        for message in messages:
            if isinstance(message, HumanMessage):
                formatted_messages.append({"role": "user", "content": message.content})
            elif isinstance(message, AIMessage):
                formatted_messages.append({"role": "assistant", "content": message.content})
        
        # Get response from Together
        response = self.llm.predict(str(formatted_messages))
        return AIMessage(content=response)

# Wrap the Together model
chat_model = TogetherChatWrapper(llm)

# Load the search tool
tools = load_tools(["ddg-search"])

# Update the prompt template for search
template = """
You are a helpful assistant. Use the search tool to find
the current weather in {location}. Look for recent weather information 
and summarize it concisely.
{agent_scratchpad}
"""
prompt_template = ChatPromptTemplate.from_template(template)

# Create the agent
agent = create_openai_tools_agent(
    llm=chat_model,
    tools=tools,
    prompt=prompt_template
)

# Create an executor for the agent
executor = AgentExecutor(agent=agent, tools=tools)

# Define the task for the agent
task = {"location": "New York"}

try:
    # Invoke the agent to perform the task
    response = executor.invoke(task)
    print(response['output'])
except Exception as e:
    print(f"An error occurred: {str(e)}")