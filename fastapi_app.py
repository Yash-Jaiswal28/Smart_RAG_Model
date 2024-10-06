from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
from typing import Literal
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_community.tools import WikipediaQueryRun

# Setup Wikipedia API Wrapper
api_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=400)
wiki = WikipediaQueryRun(api_wrapper=api_wrapper)

# Base model for request body
class QueryRequest(BaseModel):
    query: str
    context: str

# Response router model
class RouterQuery(BaseModel):
    datasource: Literal["ai_response", "vectorstore", "wiki_search"]

# FastAPI initialization
app = FastAPI()

# Load Groq API key and model
groq_api_key = os.getenv("groq_api_key")
if not groq_api_key:
    raise ValueError("Groq API key is missing from the environment variables")

llm = ChatGroq(groq_api_key=groq_api_key, model_name="llama-3.1-70b-versatile")
structured_llm_router = llm.with_structured_output(RouterQuery)

# Function for routing queries
def route_question():
    system_prompt = """
        You are an expert at routing user questions to vectorstore, AI, or Wikipedia.
        Use vectorstore for sound-related topics, AI for salutations, and Wikipedia for other topics.
    """
    
    prompt = ChatPromptTemplate.from_messages([("system", system_prompt), ("human", "{question}")])
    return prompt | structured_llm_router

# LLM answer generation
def generate_llm_answer(query, context):
    prompt = f"Context: {context}\nQuestion: {query}\nAnswer:"
    try:
        return llm.invoke(prompt).content
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM error: {e}")

# Wikipedia search function
def search_wikipedia(query):
    return wiki.run(query)

# Route to handle AI Answer
@app.post("/AI_Answer/")
def ai_answer(request: QueryRequest):
    try:
        return {"message": generate_llm_answer(request.query, request.context)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Route to handle Smart AI Agent
@app.post("/Smart_AI_Answer/")
def smart_ai_answer(request: QueryRequest):
    try:
        question_router = route_question()
        routed_answer = question_router.invoke({"question": request.query})

        if routed_answer.datasource == "ai_response":
            return {"path": "AI_Response", "message": llm.invoke(request.query).content}
        elif routed_answer.datasource == "vectorstore":
            return {"path": "Vectorstore", "message": generate_llm_answer(request.query, request.context)}
        else:
            return {"path": "Wiki Search", "message": search_wikipedia(request.query)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Routing error: {e}")
