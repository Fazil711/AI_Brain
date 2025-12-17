import os
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_classic.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings 
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import Chroma
from langchain_classic.chains import RetrievalQA
from langchain_classic.agents import initialize_agent, Tool, AgentType
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_classic.memory import ConversationBufferMemory
from dotenv import load_dotenv
from datetime import datetime

load_dotenv()

# 1. LOADERS
def load_documents(file_paths):
    docs = []
    for path in file_paths:
        if path.endswith(".pdf"):
            loader = PyPDFLoader(path)
            docs.extend(loader.load())
        elif path.endswith(".txt"):
            loader = TextLoader(path)
            docs.extend(loader.load())
    return docs

# 2. SPLITTING
def split_documents(docs):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    return text_splitter.split_documents(docs)

# 3. VECTOR DB
def create_vector_db(splits):
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    vectordb = Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
        persist_directory="./chroma_db"
    )
    return vectordb

# 4. AGENT
def setup_agent(vectordb):
    llm = ChatGoogleGenerativeAI(
        model="gemma-3-27b", 
        temperature=0, 
        convert_system_message_to_human=True
    )
    
    # Tool 1: The "Reading" Tool 
    retriever = vectordb.as_retriever(search_kwargs={"k": 3})
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm, 
        chain_type="stuff", 
        retriever=retriever
    )
    
    rag_tool = Tool(
        name="Personal Knowledge Base",
        func=qa_chain.run,
        description="Useful for answering questions based on the uploaded documents. ALWAYS use this tool first if the user asks about the document content."
    )
    
    # Tool 2: The "Search" Tool
    search_tool = Tool(
        name="Web Search",
        func=DuckDuckGoSearchRun().run,
        description="Useful for finding current information, news, or general knowledge not found in the documents."
    )

    tools = [rag_tool, search_tool]

    memory = ConversationBufferMemory(
        memory_key="chat_history", 
        return_messages=True
    )

    today = datetime.now().strftime("%A, %B %d, %Y")
    user_location = "Mumbai, India"

    agent_kwargs = {
        "prefix": (
            "You are a helpful AI assistant. Today's date is {today}.\n"
            "The user is located in {user_location}.\n"
            "You have access to the following tools:\n\n"
            "{tools}\n\n"
            "When answering, you MUST return a valid JSON blob.\n"
            "IMPORTANT: If your answer contains quotes, you MUST escape them (e.g., \"output\": \"She said \\\"Hello\\\"\").\n"
            "Do not use markdown code blocks (like ```json). Just return the raw JSON."
        )
    }

    agent = initialize_agent(
        tools, 
        llm, 
        agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
        verbose=True,
        memory=memory,
        agent_kwargs=agent_kwargs,
        handle_parsing_errors=True 
    )
    
    return agent