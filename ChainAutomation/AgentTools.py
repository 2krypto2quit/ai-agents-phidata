from langchain.agents import create_openai_tools_agent, AgentExecutor
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
import os

# Load environment variables
load_dotenv()
api_key = os.getenv('TOGETHER_API_KEY')

@tool
def search(query: str) -> str:
    """Search for information about a query."""
    return f"Results for: {query}. AGI has been achieved!"

# Initialize the model with Together AI's endpoint
model = ChatOpenAI(
    model="mistralai/Mixtral-8x7B-Instruct-v0.1",
    api_key=api_key,
    base_url="https://api.together.xyz/v1",
    temperature=0.7
)

# Create a more detailed prompt template
prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a helpful AI assistant. When you use a tool, first explain why you're using it, 
    then process its response to provide a meaningful answer. Always provide a clear final answer based on the tool's response."""),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

tools = [search]

# Create the agent with the updated prompt
agent = create_openai_tools_agent(model, tools, prompt)

# Create the agent executor
agent_executor = AgentExecutor(
    agent=agent, 
    tools=tools, 
    verbose=True,
    handle_parsing_errors=True  # Added to better handle parsing errors
)

# Run the agent
result = agent_executor.invoke(
    {"input": "What's the latest news about hurricanes?"})
print("\nFinal Output:", result['output'])