import google.generativeai as genai
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()
api_key = os.getenv('GEMINI_API_KEY')

# Configure the API
genai.configure(api_key=api_key)

# Create a template function
def create_prompt(role, user, task):
    return f"""
Role: {role}
User: {user}
Task: {task}
"""

# Create the model instance
model = genai.GenerativeModel('gemini-pro')

# Define input variables
input_variables = {
    "role": "Teacher",
    "user": "Alice",
    "task": "Explain the theory of relativity."
}

# Format the prompt
formatted_prompt = create_prompt(**input_variables)

# Send to model and get response
response = model.generate_content(formatted_prompt)
print(response.text)