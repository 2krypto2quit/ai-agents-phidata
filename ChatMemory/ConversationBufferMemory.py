from langchain.memory import ConversationBufferMemory
from langchain_openai import ChatOpenAI
from langchain.chains import ConversationChain
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()
api_key = os.getenv('TOGETHER_API_KEY')

# Initialize the model with Together AI's endpoint
llm = ChatOpenAI(
    model="mistralai/Mixtral-8x7B-Instruct-v0.1",
    api_key=api_key,
    base_url="https://api.together.xyz/v1",
    temperature=0
)

# Create the conversation chain with memory
chain = ConversationChain(
    llm=llm,
    verbose=True,
    memory=ConversationBufferMemory()
)

# First interaction
chain.invoke(input="The book topic is mathematics, can you explain what is mathematics?")

# Second interaction using memory
result = chain.invoke(input="What is the book about?")
print(result["response"])