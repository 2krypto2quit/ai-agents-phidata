from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()
api_key = os.getenv('TOGETHER_API_KEY')  # Changed to use Together API key

# Define the prompt template
template = "Tell me a joke about {topic}."
prompt_template = ChatPromptTemplate.from_template(template)

# Create a chat model instance with Together AI
chat_model = ChatOpenAI(
    model="mistralai/Mixtral-8x7B-Instruct-v0.1",  # Changed to Together AI model
    api_key=api_key,
    base_url="https://api.together.xyz/v1",  # Added Together AI base URL
    temperature=0.7
)

# Define an output parser to extract the content
output_parser = StrOutputParser()

# Define an additional step to uppercase the output
uppercase_step = RunnableLambda(lambda x: x.upper())

# Build and extend the chain
chain = prompt_template | chat_model | output_parser | uppercase_step

# Provide input variables
input_variables = {"topic": "computers"}

# Run the chain
result = chain.invoke(input_variables)
print(result)