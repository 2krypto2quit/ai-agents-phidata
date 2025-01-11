from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")  # Make sure this matches your .env variable name

# Initialize the LLM model
llm = ChatGoogleGenerativeAI(
    model="gemini-pro",
    google_api_key=api_key,
    temperature=0.7  # You can adjust this between 0 and 1
)

# Create prompt template
prompt_template = ChatPromptTemplate.from_template("""Write a {length} story about: {content}""")
prompt = prompt_template.format(length="{sentences}-sentence", 
                                content="The hometown of the legendary scientist, {who}")

# Generate response
print(llm.invoke(prompt.format(who="Albert Einstein",
                               sentences=3
                               )).content)
print("---------------------------------------------------")
print(llm.invoke(prompt.format(who="Marie Curie",
                               sentences = 5
                               )).content)

