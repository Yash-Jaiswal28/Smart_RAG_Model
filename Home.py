import io
import requests
import json
from PyPDF2 import PdfReader
import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Modularized function for processing multiple PDFs into text
def extract_text_from_pdfs(pdf_files):
    all_text = ""
    for pdf in pdf_files:
        try:
            pdf_reader = PdfReader(io.BytesIO(pdf.read()))
            text = "".join([page.extract_text() or "" for page in pdf_reader.pages])
            all_text += text
        except Exception as e:
            st.error(f"Error reading PDF {pdf.name}: {e}")
    return all_text

# Chunk text into smaller parts
def split_text_into_chunks(text, chunk_size=1000, overlap=100):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
    return text_splitter.split_text(text)

# Create FAISS vector store
def create_vector_store(text_chunks, embedding_model):
    try:
        vector_store = FAISS.from_texts(text_chunks, embedding=embedding_model)
        vector_store.save_local("faiss_index")
        return vector_store
    except Exception as e:
        st.error(f"Failed to create vector store: {e}")

# Extract page content from a response
def extract_page_content(response):
    return [item["Document"].get("page_content", "") for item in response if "Document" in item]

def process_pdfs(uploaded_pdfs):
    raw_text = extract_text_from_pdfs(uploaded_pdfs)
    if not raw_text:
        st.error("No text could be extracted from the PDFs.")
        return None

    # Split text into chunks
    text_chunks = split_text_into_chunks(raw_text)
    embeddings_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-V2")
    vector_store = create_vector_store(text_chunks, embeddings_model)

    if vector_store:
        st.session_state['vector_store'] = vector_store
        st.success("Vector store created and loaded successfully!")
    else:
        st.error("Vector store creation failed.")

def query_answering(query, endpoint):
    # Perform query answering via FastAPI
    payload = {"query": query, "context": st.session_state['raw_text']}
    response = requests.post(f"http://127.0.0.1:8000/{endpoint}/", json=payload)
    
    if response.status_code == 200:
        return response.json().get("message", "No response received.")
    else:
        st.error(f"Failed to send query. Status code: {response.status_code}")
        return None

def main():
    st.title("PDF Question Answering App")

    # PDF Upload
    uploaded_pdfs = st.file_uploader("Upload PDF Files", type="pdf", accept_multiple_files=True)

    if uploaded_pdfs and st.button("Process PDFs"):
        process_pdfs(uploaded_pdfs)

    if 'raw_text' in st.session_state:
        query = st.text_input("Ask a question related to the PDF content")
        
        if query:
            st.write(f"Query: {query}")

            if st.button("AI Answer"):
                result = query_answering(query, "AI_Answer")
                if result:
                    st.write(result)

            if st.button("Smart AI Agent"):
                result = query_answering(query, "Smart_AI_Answer")
                if result:
                    st.write(result)

if __name__ == "__main__":
    main()
