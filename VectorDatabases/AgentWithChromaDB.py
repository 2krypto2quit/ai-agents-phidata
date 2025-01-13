import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.schema.document import Document
from langchain_core.messages import HumanMessage, AIMessage

# Load environment variables
load_dotenv()
api_key = os.getenv('TOGETHER_API_KEY')

# Initialize the LLM with Together AI
llm = ChatOpenAI(
    model="mistralai/Mixtral-8x7B-Instruct-v0.1",
    api_key=api_key,
    base_url="https://api.together.xyz/v1",
    temperature=0.7
)

# Initialize memory
memory = ChatMessageHistory()

# Define the prompt
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful AI assistant."),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}")
])

# Split documents into chunks
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=500
)

# Initialize embeddings with HuggingFace
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2"
)

docs = [Document(page_content=x) for x in splitter.split_text(".")]
vector_store = Chroma.from_documents(
    documents=docs,
    persist_directory=".",
    embedding=embeddings,
    collection_name="conversations"
)

# Create the chain
chain = prompt | llm

def savechattochroma(human, ai):
    docsx = []
    docsx.append(Document(page_content=f"human:{human}"))
    docsx.append(Document(page_content=f"AI:{ai}"))
    vector_store.add_documents(documents=docsx)
    return

# Create the runnable with message history
runnable = RunnableWithMessageHistory(
    chain,
    lambda session_id: memory,
    input_messages_key="input",
    history_messages_key="history"
)

def process_chat(runnable, user_input):
    response = runnable.invoke(
        {"input": user_input},
        config={"configurable": {"session_id": session_id}}
    ).content
    return response

session_id = "user_123"
print("Agent is ready to chat. Type 'exit' or 'quit' to end the conversation.")

if __name__ == "__main__":
    while True:
        user_input = input("You: ")
        if user_input.lower() in ['exit', 'quit']:
            break
            
        # Use standard similarity_search
        results = vector_store.similarity_search(
            user_input, 
            k=25  # Number of results to return
        )
        
        addedresults = "\n".join([result.page_content for result in results])
        
        response = process_chat(
            runnable, 
            user_input + ", consider this context from previous conversations:" + addedresults
        )
        
        print(f"AI: {response}")
        
        savechattochroma(
            HumanMessage(content=user_input),
            AIMessage(content=response)
        )