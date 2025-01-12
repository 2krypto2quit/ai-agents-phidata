from langchain_openai import ChatOpenAI
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()
api_key = os.getenv('TOGETHER_API_KEY')

# Create the prompt template with more explicit formatting
prompt = ChatPromptTemplate.from_template(
    """Analyze the following information about color preferences:

{context}

Please summarize everyone's favorite colors."""
)

# Initialize the model with Together AI's endpoint
llm = ChatOpenAI(
    model="mistralai/Mixtral-8x7B-Instruct-v0.1",
    api_key=api_key,
    base_url="https://api.together.xyz/v1",
    temperature=0.7
)

# Create the document chain
chain = create_stuff_documents_chain(llm, prompt)

# Define the documents
docs = [
    Document(page_content="Jesse loves red but not yellow"),
    Document(page_content="Jamal loves green but not as much as he loves orange")
]

# Format the documents into a string before passing to the chain
formatted_docs = "\n".join(doc.page_content for doc in docs)

try:
    # Run the chain with explicit formatting
    result = chain.invoke({
        "context": formatted_docs
    })
    print(result)
except Exception as e:
    print(f"Error: {str(e)}")
    # Try alternative formatting if needed
    try:
        result = chain.invoke({
            "context": docs
        })
        print(result)
    except Exception as e:
        print(f"Alternative attempt error: {str(e)}")