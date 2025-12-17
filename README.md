# 🧠 AI Knowledge Agent (Personal Brain)

**A multi-modal AI Agent that acts as a "Second Brain"—capable of reading documents, searching the live web, and analyzing YouTube videos in a single conversation.**

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![LangChain](https://img.shields.io/badge/LangChain-Agents-green)
![Streamlit](https://img.shields.io/badge/Streamlit-UI-red)
![Google Gemini](https://img.shields.io/badge/Model-Gemini%201.5%20Flash-orange)

## 🚀 Overview

This project goes beyond a simple chatbot. It is an **Agentic RAG System** built with LangChain. Instead of just answering from training data, the AI acts as a reasoning engine that dynamically selects the best tool for the job:
* **Need current news?** It searches the web.

<img width="800" height="500" alt="Image" src="https://github.com/user-attachments/assets/c2d911be-3782-4bb2-a92f-b526a85a1fa3" />

* **Need to understand a PDF?** It retrieves specific chunks from your uploaded files.

<img width="800" height="500" alt="Image" src="https://github.com/user-attachments/assets/0deadb6f-25bc-4b6a-9f3f-76009d061907" />

* **Need to summarize a video?** It fetches and analyzes YouTube transcripts.

<img width="800" height="500" alt="Image" src="https://github.com/user-attachments/assets/31331596-b16a-42fc-aa72-d8a0b9f2d45b" />

* **Need to remember context?** It uses a persistent SQLite database to store chat history.

## ✨ Key Features

* **🕵️ Agentic Search:** Uses a "Router" architecture to autonomously decide whether to use internal documents, Google Search, or YouTube analysis.
* **📚 RAG (Retrieval Augmented Generation):** Ingests PDFs/Text files, splits them into chunks, embeds them locally (HuggingFace), and stores them in ChromaDB.
* **🎥 YouTube Intelligence:** Can watch (transcribe) YouTube videos via URL and answer questions about specific timestamped content.
* **🧠 Multi-Model Brain:** Instant toggle between **Google Gemini 1.5 Flash** (Free/Fast) and **OpenAI GPT-4o** (High Reasoning).
* **💾 Persistent Memory:** Uses an integrated **SQLite** database to save chat history, ensuring conversations persist across reloads.
* **☁️ Cloud Ready:** Deployed with `pysqlite3` fixes to ensure compatibility with Streamlit Community Cloud.

## 🛠️ Tech Stack

* **Framework:** LangChain (Python)
* **Frontend:** Streamlit
* **LLMs:** Google Gemini 1.5 Flash, OpenAI GPT-4o
* **Vector Database:** ChromaDB (Local persistence)
* **Embeddings:** HuggingFace (`all-MiniLM-L6-v2`) — *Runs locally for privacy & cost savings.*
* **Search Tool:** DuckDuckGo Search
* **Memory:** SQLite3

## 📂 Project Structure

```bash
├── app.py              # Frontend UI (Streamlit) & Session Management
├── brain.py            # Core Logic: Agents, Tools, RAG Pipeline
├── history.py          # Database Logic: SQLite handling for chat memory
├── requirements.txt    # Dependencies
├── .env                # API Keys (Not pushed to GitHub)
└── packages.txt        # System-level dependencies for Linux deployment
```
## ⚙️ Installation & Setup

1. **Clone the Repository**

```Bash
git clone [https://github.com/yourusername/ai-brain.git](https://github.com/yourusername/ai-brain.git)
cd ai-brain
```

2. **Create a Virtual Environment**

```Bash
python -m venv venv
# Windows:
.\venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate
```

3. **Install Dependencies**

```Bash
pip install -r requirements.txt
```

4. **Set up API Keys Create a .env file in the root directory and add your keys:**

```Bash

GOOGLE_API_KEY="your_google_api_key_here"
OPENAI_API_KEY="your_openai_api_key_here" # Optional
```
5. **Run the App**

```Bash
streamlit run app.py
```

## 🤖 Usage Guide

1.  **Web Search:** Just ask a question! *"Who won the 2024 Cricket World Cup?"* — The agent will browse the web.
2.  **Document Analysis:**
    * Open the Sidebar.
    * Upload a PDF (e.g., a research paper).
    * Click **"Process & Add to Brain"**.
    * Ask: *"Summarize the key findings in the uploaded paper."*
3.  **YouTube Summary:**
    * Paste a video URL: *"Summarize this video: https://youtube.com/..."*
    * The agent will download the transcript and analyze it.
4.  **Switch Models:** Use the radio button in the sidebar to swap between Gemini and GPT-4o on the fly.
5.  **Reset:** Click **"Reset Brain"** to wipe the vector database and start fresh.

## 🚧 Challenges & Solutions

* **Streamlit Cloud & SQLite:** Streamlit runs on Linux with an older SQLite version that breaks ChromaDB.
    * *Solution:* Implemented `pysqlite3-binary` swap in `app.py` to force a modern SQLite version.
* **Rate Limiting:** Google's Embedding API has strict rate limits.
    * *Solution:* Switched to **HuggingFace Local Embeddings** (`sentence-transformers`), which is free, faster, and private.
* **Windows File Locking:** Windows often locks temp files, causing crashes during cleanup.
    * *Solution:* Added robust error handling and Garbage Collection (`gc.collect()`) to safely manage file deletion.