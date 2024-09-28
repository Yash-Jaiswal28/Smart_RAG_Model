# Smart_RAG_Model

**Smart_RAG_Model** is an intelligent PDF question-answering application that leverages advanced AI and Retrieval-Augmented Generation (RAG) capabilities to provide responses based on user queries. The app allows you to upload PDFs and query their content using two different modes:

1. **AI + RAG Response**: Generates answers using both AI models and a vector store of the PDF content.
2. **Smart Routing**: Routes the query to either:
   - AI Response
   - AI + RAG Response
   - Wikipedia Search for additional contextual information.

This project uses **Streamlit** for the frontend and **FastAPI** for the backend.

## Features

### 1. PDF Upload and Question Answering
After uploading PDFs, users can ask questions based on the content of the uploaded PDFs. The app supports multiple PDFs and processes them to create a searchable vector store.

### 2. Modes of Answering
- **AI + RAG Response**: Combines AI-powered answers with retrieval from a pre-built vector store of PDF content.
- **Smart AI Agent**: Routes the query to either:
  - **AI Response**: Direct response from the AI model (e.g., using Groq's Llama-3.1).
  - **AI + RAG Response**: Combines AI with retrieval from vectorstore.
  - **Wikipedia Search**: If the query is outside the scope of the vectorstore, searches Wikipedia for relevant information.

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/YOUR-USERNAME/Smart_RAG_Model.git
   cd Smart_RAG_Model
   ```

2. Install the required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Set up environment variables:

   Ensure you have your **Groq API key** set in the environment:

   ```bash
   export groq_api_key="YOUR_GROQ_API_KEY"
   ```

## Running the Application

### 1. Start the FastAPI Backend

The backend handles AI and RAG-based responses, and Wikipedia queries.

```bash
uvicorn fastapi_app:app --reload
```

This will start the FastAPI server locally at `http://127.0.0.1:8000/`.

### 2. Start the Streamlit Frontend

The frontend is a simple web interface to upload PDFs and submit queries.

```bash
streamlit run Home.py
```

Access the frontend at `http://localhost:8501/`.

## Usage

1. **Upload PDFs**: You can upload multiple PDF files using the interface on the home page.
2. **Ask Questions**: Once the PDFs are processed, you can ask questions about the content.
3. **Choose a Response Mode**:
   - **AI + RAG Response**: Uses a combination of AI and a vector store of the uploaded PDFs to provide a contextual answer.
   - **Smart AI Agent**: Routes the query to either AI, AI + RAG, or Wikipedia, based on the context of the query.

### Backend Routes

The backend supports the following API routes:

1. **AI Answer**:
   - **Endpoint**: `/AI_Answer/`
   - **Method**: POST
   - **Payload**:
     ```json
     {
       "query": "Your query",
       "context": "PDF content or other text"
     }
     ```
   - **Description**: Provides a direct AI-generated answer based on the query and the PDF context.

2. **Smart AI Agent**:
   - **Endpoint**: `/Smart_AI_Answer/`
   - **Method**: POST
   - **Payload**:
     ```json
     {
       "query": "Your query",
       "context": "PDF content or other text"
     }
     ```
   - **Description**: Routes the query to either an AI response, AI + RAG response, or Wikipedia search depending on the context.

## Project Structure

```
Smart_RAG_Model/
│
├── Home.py                 # Streamlit frontend
├── fastapi_app.py           # FastAPI backend
├── requirements.txt         # Python dependencies
├── README.md                # Project documentation
└── faiss_index/             # FAISS vector store files
```

## Technology Stack

- **Frontend**: Streamlit (for uploading PDFs, and asking questions)
- **Backend**: FastAPI (for handling AI and RAG-based responses)
- **LLM**: Groq's Llama-3.1-70b model (for AI-based answering)
- **Vector Store**: FAISS (for managing and retrieving from the PDF content)
- **Embeddings**: HuggingFace `all-MiniLM-L6-V2`
- **Wikipedia Search**: Wikipedia API Wrapper

## Example

1. **Upload PDFs**: Upload multiple PDF files, and the system will extract and process the text.
2. **Ask a Question**: Enter a question related to the content of the PDF.
3. **Get AI or RAG Response**: Based on the selected option, either an AI model or a combination of AI + RAG will provide an answer. You can also use the Smart Agent, which may route the query to Wikipedia for broader context.

## Contributing

Feel free to contribute by creating a pull request, opening an issue, or improving the documentation.
