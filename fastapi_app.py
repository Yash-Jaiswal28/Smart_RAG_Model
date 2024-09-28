from fastapi import FastAPI, HTTPException
from langchain_groq import ChatGroq
import os
from typing import Literal
from langchain.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_community.tools import WikipediaQueryRun

# Setup Wikipedia API Wrapper
api_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=400)
wiki = WikipediaQueryRun(api_wrapper=api_wrapper)


class RouterQuery(BaseModel):
    datasource: Literal["ai_response", "vectorstore", "wiki_search"] = Field(
        ..., description="Route to Wikipedia or vectorstore."
    )


# Initialize FastAPI app
app = FastAPI()

# Load Groq API key and model
groq_api_key = os.environ.get("groq_api_key")
if not groq_api_key:
    raise ValueError("Groq API key is missing in the environment variables")


llm = ChatGroq(groq_api_key=groq_api_key, model_name="llama-3.1-70b-versatile")
structured_llm_router = llm.with_structured_output(RouterQuery)

# Function to handle query routing
def route():
    system = """
        You are an expert at routing a user question to vectorstore, ai_response, or Wikipedia.
        Vectorstore has documents related to sound, its characteristics, and properties. 
        For questions on these topics, use the vectorstore.
        For salutations, use ai_response.
        Use Wikipedia for other topics.
    """
    
    router_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            ("human", "{question}")
        ]
    )
    question_router = router_prompt | structured_llm_router
    return question_router

# Function to generate an LLM answer
def llm_answer(query, context):
    prompt = f"""
        Context: {context}

        You are an expert assistant designed to answer questions based on the provided context. 
        Please answer the following question based on this context:

        Question: {query}

        Answer:
    """
    try:
        response = llm.invoke(prompt)
        return response.content
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error invoking LLM: {str(e)}")


def wiki_search(query):
      docs = wiki.run(
          query
      )
      wiki_results= docs
      return wiki_results

class New(BaseModel):
    query: str
    context: str  


@app.post("/AI_Answer/")
def response_1(new: New):
    query=new.query
    context = new.context
    return {"message": llm_answer(new.query, new.context)}


@app.post("/Smart_AI_Answer/")
def smart_ai_answer(new: New):
    question_router = route()
    try:
        routed_answer = question_router.invoke({"question": new.query})
        path=routed_answer.datasource
        if path =="ai_response":
            return {
                "path":"AI_Response",
                "message":llm.invoke(new.query).content
                }
        elif path=="vectorstore":
            return {
                "path":"Vectorstore",
                "message": llm_answer(new.query, new.context)
                }
        else :
            return {
                "path":"Wiki Search",
                "message":wiki_search(new.query)
                }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error routing question: {str(e)}")
