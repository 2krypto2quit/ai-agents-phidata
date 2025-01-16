from langchain_community.agent_toolkits.load_tools import load_tools
from langchain_community.utilities import GoogleSerperAPIWrapper
import os
from dotenv import load_dotenv

load_dotenv()

#tools = load_tools(["google-serper"])
search_tool = GoogleSerperAPIWrapper(api_key=os.getenv("GEMINI_API_KEY"), top_k_results=1)


#query = "Who is the worlds most famous person?"
query = "What is the hometown of the reigning men's U.S. Open champion?"
response = search_tool.run(query)

print(response)
#print(tools[0].invoke(query))

