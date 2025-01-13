import os
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage

load_dotenv()
api_key = os.getenv('TOGETHER_API_KEY')

agent = ChatOpenAI(
    model="mistralai/Mixtral-8x7B-Instruct-v0.1",
    api_key=api_key,
    base_url="https://api.together.xyz/v1",
    temperature=0.7
)

task = "Find the current weather in New York City."

response = agent.invoke([HumanMessage(content=task)])

print(response.content) 