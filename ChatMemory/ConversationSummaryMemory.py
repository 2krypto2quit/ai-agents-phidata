from langchain.memory import ConversationSummaryMemory
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
    temperature=1
)

# Create the conversation chain with summary memory
chain = ConversationChain(
    llm=llm,
    verbose=True,
    memory=ConversationSummaryMemory(llm=llm)
)

# First interaction
chain.invoke(
    input="I am writting a book about mathematics, can you explain what is mathematics?"
)

# Second interaction
chain.invoke(
    input="My name is Karel,what is the book I am writing about?"
)

# Ask about previous conversation
result = chain.invoke(input="What did we talked about?")
print(result["response"])