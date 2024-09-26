import io
from PyPDF2 import PdfReader
import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import requests
import json

def extract_text_from_pdfs(pdf_files):
    """Extracts and concatenates text from multiple PDF files."""
    all_text = ""
    for pdf in pdf_files:
        try:
            pdf_file = io.BytesIO(pdf.read())
            pdf_reader = PdfReader(pdf_file)
            for page in pdf_reader.pages:
                page_text = page.extract_text()
                if page_text:
                    all_text += page_text
        except Exception as e:
            st.error(f"Error reading PDF file: {pdf.name}. Reason: {str(e)}")
    return all_text


def split_text_into_chunks(text, chunk_size=1000, overlap=100):
    """Splits a given text into smaller chunks."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
    return text_splitter.split_text(text)


def create_vector_store_from_chunks(text_chunks, embedding_model):
    """Creates a FAISS vector store from the given text chunks and embeddings."""
    try:
        vector_store = FAISS.from_texts(text_chunks, embedding=embedding_model)
        vector_store.save_local("faiss_index")
        return vector_store
    except Exception as e:
        st.error(f"Failed to create vector store. Reason: {str(e)}")

def extract_page_content(response):
    """
    Extracts 'page_content' from each document in the response.
    """
    content_list = []
    
    for item in response:
        if "Document" in item and "page_content" in item["Document"]:
            # Extract the page content
            content = item["Document"]["page_content"]
            content_list.append(content)
    
    return content_list

def main():
    st.title("PDF Question Answering App")

    # Upload PDF files
    uploaded_pdfs = st.file_uploader("Upload PDF Files", type="pdf", accept_multiple_files=True)

    # Check if PDFs are uploaded
    if uploaded_pdfs:
        # When the button is clicked, process PDFs
        if st.button("Process PDFs"):
            raw_text = extract_text_from_pdfs(uploaded_pdfs)
            
            if raw_text:
                # Store raw text in session state to retain it between reruns
                st.session_state['raw_text'] = raw_text
                # st.write(f"Extracted text length: {len(raw_text)} characters")
                
                # Split text into chunks (if needed for any processing)
                text_chunks = split_text_into_chunks(raw_text)
                
                if text_chunks:
                    st.session_state['text_chunks'] = text_chunks  # Store chunks in session state
                    # st.write("Sample Text Chunk 1:", text_chunks[0])
                
                # Create embeddings model and store vector store (if still needed)
                embeddings_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-V2")
                vector_store = create_vector_store_from_chunks(text_chunks, embeddings_model)

                if vector_store:
                    st.session_state['vector_store'] = vector_store  # Store vector store in session state
                    st.success("Vector store created successfully!")
                    try:
                        vector_store = FAISS.load_local("faiss_index", embeddings_model, allow_dangerous_deserialization=True)
                        st.session_state['vector_store'] = vector_store  # Update in session state
                        st.write("Vector store loaded successfully.")
                    except Exception as e:
                        st.error(f"Failed to load vector store: {str(e)}")
            else:
                st.error("No text could be extracted from the uploaded PDFs.")
    
    # Ensure that we display text input only after PDFs have been processed
    if 'raw_text' in st.session_state:
        query = st.text_input("Ask a question related to the PDF content")

        # Process the query when available
        if query:
            st.write(f"Query entered: {query}")

            # Instead of similarity search, send the full text
            complete_text = st.session_state['raw_text']
            payload = {
                "query": query,
                "context": complete_text  # Pass complete text to the API
            }

            if st.button("AI Answer"):
                response = requests.post("http://127.0.0.1:8000/AI_Answer/", json=payload)

                if response.status_code == 200:
                    st.success(f"Response from FastAPI: {response.json()['message']}")
                else:
                    st.error(f"Failed to send data. Status code: {response.status_code}")
    

if __name__ == "__main__":
    main()
