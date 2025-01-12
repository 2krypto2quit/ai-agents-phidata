from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda 
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

def route(info): 

    if "program" in info:
        return upperlower_step
    else:
        return uppercase_step
    
    def lowerupper(test_str):
        res=""

        for idx in range(len(test_str)):
            if idx % 2 == 0:
                res += test_str[idx].upper()
            else:
                res += test_str[idx].lower()
                return res

# Create the first prompt
prompt = ChatPromptTemplate.from_template("Tell me a {adjective} joke about {topic}:")

#do a transformation
uppercase_step = RunnableLambda(lambda x: x.upper())
upperlower_step = RunnableLambda(lambda x: x.lower())

#run the chain
chain = prompt | model | StrOutputParser() | uppercase_step | RunnableLambda(route) 

result = chain.invoke({"adjective": "funny", "topic": "chickens"})

print(result)

result = chain.invoke({"adjective": "funny", "topic": "fishing"})

print(result)
