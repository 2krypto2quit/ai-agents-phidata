import google.generativeai as genai
import os
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=api_key)

# Choose the Gemini model
model = genai.GenerativeModel('gemini-1.5-flash-latest')
# Generate content
response = model.generate_content("Hello, how are you?")
# Print the content of the response
print(response.text)