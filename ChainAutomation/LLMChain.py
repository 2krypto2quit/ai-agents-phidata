from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
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

# Create the first prompt
prompt = ChatPromptTemplate.from_template("Tell me a {adjective} joke about {topic}:")

#run the chain
#output response will be a string
chain = prompt | model | StrOutputParser()

result = chain.invoke({"adjective": "funny", "topic": "chickens"})

print(result)
