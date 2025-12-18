import pysqlite3
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
import streamlit as st
import os
import shutil
from history import init_db, load_history, save_message, clear_history
from brain import load_documents, split_documents, create_vector_db, setup_agent

st.set_page_config(page_title="Personal Brain 🧠", layout="wide")
st.title("🤖 AI Knowledge Agent (Personal Brain)")

if "db_initialized" not in st.session_state:
    init_db()
    st.session_state.db_initialized = True

if "messages" not in st.session_state:
    st.session_state.messages = load_history()

if "vectordb" not in st.session_state:
    st.session_state.vectordb = None

if "dataframes" not in st.session_state:
    st.session_state.dataframes = None

if "chain" not in st.session_state:
    st.session_state.chain = setup_agent(
        vectordb=None, 
        dataframes=None, 
        model_choice="Google Gemini"
    )

with st.sidebar:
    st.header("🧠 Brain Settings")
    model_choice = st.radio("Choose your Model:", ("Google Gemini", "OpenAI GPT-4o"))
    
    if st.button("Apply Model Change"):
        #current_db = st.session_state.get("vectordb", None)
        st.session_state.chain = setup_agent(
            vectordb=st.session_state.vectordb, 
            dataframes=st.session_state.dataframes, 
            model_choice=model_choice
        )
        st.success(f"Switched to {model_choice}!")

    st.divider()

    st.header("📂 Feed the Brain")
    uploaded_files = st.file_uploader("Upload PDF, TXT, CSV, or Excel", accept_multiple_files=True)
    
    if st.button("Process & Add to Brain"):
        if uploaded_files:
            with st.spinner("Digesting information..."):
                if not os.path.exists("temp_data"):
                    os.makedirs("temp_data")
                
                temp_paths = []
                for uploaded_file in uploaded_files:
                    file_path = os.path.join("temp_data", uploaded_file.name)
                    with open(file_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    temp_paths.append(file_path)

                try:
                    docs, dataframes = load_documents(temp_paths)
                    
                    vectordb = None
                    if docs:
                        splits = split_documents(docs)
                        vectordb = create_vector_db(splits)
                    
                    st.session_state.vectordb = vectordb
                    st.session_state.dataframes = dataframes
                    
                    st.session_state.chain = setup_agent(
                        vectordb=vectordb, 
                        dataframes=dataframes, 
                        model_choice=model_choice
                    )
                    
                    msg = "Brain Updated!"
                    if docs: msg += f" 📄 Read {len(docs)} text pages."
                    if dataframes: msg += f" 📊 Loaded {len(dataframes)} data tables."
                    st.success(msg)
                    
                except Exception as e:
                    st.error(f"An error occurred: {e}")
                finally:
                    try:
                        shutil.rmtree("temp_data")
                    except PermissionError:
                        st.warning("Windows is holding a file. It will be cleared later.")
        else:
            st.warning("Please upload documents first.")

    if st.button("🗑️ Reset / Clear Brain"):
        with st.spinner("Performing lobotomy..."):
            st.session_state.vectordb = None
            st.session_state.dataframes = None
            st.session_state.chain = None
            st.session_state.messages = []
            
            clear_history()
            
            import gc
            gc.collect()
            
            if os.path.exists("./chroma_db"):
                try:
                    shutil.rmtree("./chroma_db")
                except PermissionError:
                    st.error("Windows Locked the File! Manual delete required.")
            
            st.session_state.chain = setup_agent(None, None, model_choice)
            st.rerun()

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask me anything (YouTube, Web, or Docs)..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    save_message("user", prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                response = st.session_state.chain.run(input=prompt)
                
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
                save_message("assistant", response)
                
                if os.path.exists("visual.png"):

                    st.image("visual.png", caption="Generated Visualization")
                    
                    os.remove("visual.png")
                    
            except Exception as e:
                st.error(f"Error: {e}")