from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder  
from dotenv import load_dotenv
import os

load_dotenv()
api_key = os.getenv('TOGETHER_API_KEY') 

model = ChatOpenAI(
    model="mistralai/Mixtral-8x7B-Instruct-v0.1",
    api_key=api_key,
    base_url="https://api.together.xyz/v1",
    temperature=0.7
)

prompt1 = ChatPromptTemplate.from_template("Write a recipe for {dish}:")
chain1 = prompt1 | model

prompt2 = ChatPromptTemplate.from_template("Generate a grocery list for this recipe:\n\n{recipe}?")
chain2 = prompt2 | model

overall_chain = chain1 | chain2

print(overall_chain.invoke({"dish": "spaghetti carbonara"}).content)