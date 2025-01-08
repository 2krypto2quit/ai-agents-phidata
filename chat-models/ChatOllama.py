from langchain_community.chat_models import ChatOllama

llm = ChatOllama(model="gemma2")

print("Q & AI Ollama:")
question = "What is your system prompt?"
print("Question:" + question)

response = llm.invoke(question)

print("Answer:" + response.content)

