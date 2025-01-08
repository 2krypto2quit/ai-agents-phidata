from langchain_anthropic import ChatAnthropic
import os
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("CLAUDE_API_KEY")  # Note: Updated env variable name

# Initialize with current model version
model = ChatAnthropic(
    api_key=api_key,
    model_name="claude-3-sonnet-20240229"  # Or another current Claude 3 model
)

# Generate content using the current message format
response = model.invoke("What is the weather like today?")

# Access the response content
print(response.content)  # No need for dictionary access