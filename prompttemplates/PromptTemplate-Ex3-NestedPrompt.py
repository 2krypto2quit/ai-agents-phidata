import google.generativeai as genai
from dotenv import load_dotenv
import os

# Load environment variables and configure
load_dotenv()
genai.configure(api_key=os.getenv('GEMINI_API_KEY'))

# Initialize the model
model = genai.GenerativeModel('gemini-pro')

def get_story(who, sentences):
    prompt = f"Write a {sentences}-sentence story about: The hometown of the legendary scientist, {who}"

    response = model.generate_content(prompt)
    return response.text

# Generate response
einstein_story = get_story("Albert Einstein", 3)
print(einstein_story)
print("---------------------------------------------------")
newton_story = get_story("Isaac Newton", 5)
print(newton_story) 