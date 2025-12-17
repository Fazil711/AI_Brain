import pysqlite3
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
import streamlit as st
import os
import shutil
from brain import load_documents, split_documents, create_vector_db, setup_agent

st.set_page_config(page_title="Personal Brain 🧠", layout="wide")
st.title("🤖 AI Knowledge Agent (Personal Brain)")
if "chain" not in st.session_state:
    st.session_state.chain = setup_agent(vectordb=None, model_choice="Google Gemini")

if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.header("🧠 Brain Settings")
    
    model_choice = st.radio("Choose your Model:", ("Google Gemini", "OpenAI GPT-4o"))
    
    if st.button("Apply Model Change"):
        current_db = st.session_state.get("vectordb", None)
        st.session_state.chain = setup_agent(current_db, model_choice)
        st.success(f"Switched to {model_choice}!")

    st.divider()

    st.header("📂 Feed the Brain")
    uploaded_files = st.file_uploader("Upload PDFs or Text", accept_multiple_files=True)
    
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
                    docs = load_documents(temp_paths)
                    splits = split_documents(docs)
                    vectordb = create_vector_db(splits)
                    
                    st.session_state.vectordb = vectordb
                    st.session_state.chain = setup_agent(vectordb, model_choice)
                    
                    st.success("Brain Updated! Now I can read your files.")
                except Exception as e:
                    st.error(f"An error occurred: {e}")
                finally:
                    try:
                        shutil.rmtree("temp_data")
                    except PermissionError:
                        pass
        else:
            st.warning("Please upload documents first.")

    if st.button("🗑️ Reset / Clear Brain"):
        with st.spinner("Performing lobotomy..."):
            
            st.session_state.vectordb = None
            st.session_state.chain = None
            st.session_state.messages = [] 
            
            import gc
            gc.collect()
            
            if os.path.exists("./chroma_db"):
                try:
                    shutil.rmtree("./chroma_db")
                    st.warning("Brain wipe successful! Memory deleted.")
                except PermissionError:
                    st.error("⚠️ Windows Locked the File! Please stop the app (Ctrl+C) and manually delete the 'chroma_db' folder.")
            else:
                st.info("Brain was already empty.")
            
            st.session_state.chain = setup_agent(None, model_choice)
            st.rerun() 

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask me anything (Web or Docs)..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                response = st.session_state.chain.run(input=prompt)
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
            except Exception as e:
                st.error(f"Error: {e}")