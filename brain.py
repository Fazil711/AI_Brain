import os
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_classic.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings 
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_classic.chains import RetrievalQA
from langchain_classic.agents import initialize_agent, Tool, AgentType
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_classic.memory import ConversationBufferMemory
from langchain_community.tools import YouTubeSearchTool 
from langchain_community.document_loaders import YoutubeLoader
import youtube_transcript_api
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

# 4. YOUTUBE
def get_youtube_transcript(video_url):
    try:
        loader = YoutubeLoader.from_youtube_url(
            video_url, 
            add_video_info=False, 
            language=["en", "hi"] 
        )
        
        docs = loader.load()
        
        if docs:
            return docs[0].page_content[:4000]
        else:
            return "No transcript found."
            
    except Exception as e:
        return f"Error fetching transcript: {str(e)}"
    
# 5. AGENT
def setup_agent(vectordb, model_choice="Google Gemini"):
    
    if model_choice == "Google Gemini":
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash-lite", 
            temperature=0, 
            convert_system_message_to_human=True
        )
    elif model_choice == "OpenAI GPT-4o":
        llm = ChatOpenAI(
            model_name="gpt-4o", 
            temperature=0
        )

    search_tool = Tool(
        name="Web Search",
        func=DuckDuckGoSearchRun().run,
        description="Useful for finding current information, news, or general knowledge."
    )

    youtube_tool = Tool(
        name="YouTube Analyzer",
        func=get_youtube_transcript,
        description="Useful for summarizing or answering questions about YouTube videos. Input should be a full YouTube URL."
    )

    tools = [search_tool, youtube_tool]

    if vectordb:
        retriever = vectordb.as_retriever(search_kwargs={"k": 3})
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm, 
            chain_type="stuff", 
            retriever=retriever
        )
        rag_tool = Tool(
            name="Personal Knowledge Base",
            func=qa_chain.run,
            description=(
                "Useful for answering questions based on the uploaded documents. "
                "IMPORTANT: Do not pass generic terms like 'document', 'file', or 'what is this' to this tool. "
                "Instead, paraphrased the query to be specific. "
                "Example: If user asks 'what is this?', input 'Summarize the main topics of the document'. "
                "Example: If user asks 'explain', input 'Explain the core concepts found in the text'."
            )
        )
        tools.append(rag_tool)

    memory = ConversationBufferMemory(
        memory_key="chat_history", 
        return_messages=True
    )
    
    today = datetime.now().strftime("%A, %B %d, %Y")
    user_location = "Mumbai, India" 

    agent_kwargs = {
        "prefix": (
            f"You are a helpful AI assistant. Today's date is {today}.\n"
            f"The user is located in {user_location}.\n"
            "You have access to the following tools:\n\n"
            "{tools}\n\n"
            "IMPORTANT NOTES:\n"
            "1. For weather/news, ALWAYS append user location to the query.\n"
            "2. Return a valid JSON blob. Escape quotes inside strings."
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