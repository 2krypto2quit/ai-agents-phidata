import google.generativeai as genai
from dotenv import load_dotenv
import os

# Load environment variables and configure
load_dotenv()
genai.configure(api_key=os.getenv('GEMINI_API_KEY'))

# Initialize the model
model = genai.GenerativeModel('gemini-pro')

# Define your input
place = "California"

# Create and send the prompt
prompt = f"What is the capital of {place}?"
response = model.generate_content(prompt)

print(response.text)