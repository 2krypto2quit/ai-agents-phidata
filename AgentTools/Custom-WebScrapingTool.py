import os
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from langchain.tools import Tool
from langchain_community.llms import Together

# Load environment variables
load_dotenv()
together_api_key = os.getenv('TOGETHER_API_KEY')

# Initialize TogetherAI
llm = Together(
    model="mistralai/Mixtral-8x7B-Instruct-v0.1",
    together_api_key=together_api_key,
    temperature=0.7
)

# Define a web scraping tool function
def web_scraper(url):
    try:
        response = requests.get(url)
        soup = BeautifulSoup(response.content, "html.parser")
        return {"content": soup.get_text()}
    except Exception as e:
        return {"content": str(e)}

# Define the web scraping tool
web_scraping_tool = Tool(
    name="web_scraper",
    func=web_scraper,
    description="A tool to scrape information from a website"
)

try:
    # Example usage
    url = "https://www.coindesk.com/"
    
    # Get the scraped content
    scraped_content = web_scraping_tool.invoke(url)['content']
    
    # Use TogetherAI to analyze the content
    prompt = f"Analyze and summarize this web content:\n{scraped_content[:2000]}"  # Limiting content length for safety
    analysis = llm.invoke(prompt)
    
    print("Scraped Content:")
    print(scraped_content[:500] + "...\n")  # Show first 500 chars for brevity
    print("AI Analysis:")
    print(analysis)

except Exception as e:
    print(f"An error occurred: {str(e)}")