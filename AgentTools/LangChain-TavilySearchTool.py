import os
from dotenv import load_dotenv
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.llms import Together

# Load environment variables
load_dotenv()

# Get API keys from environment variables
together_api_key = os.getenv('TOGETHER_API_KEY')
tavily_api_key = os.getenv('TAVILY_API_KEY')

try:
    # Initialize Tavily Search tool with API key
    search_tool = TavilySearchResults(tavily_api_key=tavily_api_key)
    
    # Initialize TogetherAI
    llm = Together(
        model="mistralai/Mixtral-8x7B-Instruct-v0.1",  # You can change the model
        together_api_key=together_api_key,
        temperature=0.7
    )

    # Example search query
    query = "Olympic results"
    
    # Get search results
    search_results = search_tool.invoke(query)
    
    # Use TogetherAI to process the results
    prompt = f"Given these search results about '{query}', provide a concise summary:\n{search_results}"
    response = llm.invoke(prompt)
    
    print("Search Results:")
    print(search_results)
    print("\nAI Summary:")
    print(response)

except Exception as e:
    print(f"An error occurred: {str(e)}")