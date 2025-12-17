import pysqlite3
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
import streamlit as st
import os
import shutil
from brain import load_documents, split_documents, create_vector_db, setup_chain

st.set_page_config(page_title="Personal Brain 🧠", layout="wide")
st.title("🤖 AI Knowledge Agent (Personal Brain)")

with st.sidebar:
    st.header("🧠 Feed the Brain")
    uploaded_files = st.file_uploader("Upload PDFs or Text", accept_multiple_files=True)
    
    if st.button("Process & Ingest"):
        if uploaded_files:
            with st.spinner("Digesting information..."):
                temp_paths = []
                if not os.path.exists("temp_data"):
                    os.makedirs("temp_data")
                
                for uploaded_file in uploaded_files:
                    file_path = os.path.join("temp_data", uploaded_file.name)
                    with open(file_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    temp_paths.append(file_path)

                docs = load_documents(temp_paths)
                splits = split_documents(docs)
                vectordb = create_vector_db(splits)
                
                st.session_state.chain = setup_chain(vectordb)
                
                try:
                    shutil.rmtree("temp_data")
                except PermissionError:
                    st.warning("⚠️ Note: Windows is holding onto the file. It will be cleared automatically later.")
                
                st.success("Brain Updated!")
        else:
            st.warning("Please upload documents first.")

if "messages" not in st.session_state:
    st.session_state.messages = []

if "chain" not in st.session_state:
    st.info("👈 Please upload documents in the sidebar to start.")
else:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Ask your Personal Brain..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = st.session_state.chain({"question": prompt})
                answer = response['answer']
                sources = response['source_documents']
                
                st.markdown(answer)
                
                with st.expander("📚 View Sources / Citations"):
                    for i, doc in enumerate(sources):
                        st.markdown(f"**Source {i+1}:** {doc.metadata.get('source', 'Unknown')}")
                        st.caption(doc.page_content[:300] + "...")

        st.session_state.messages.append({"role": "assistant", "content": answer})