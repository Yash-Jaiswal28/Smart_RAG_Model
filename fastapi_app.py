from fastapi import FastAPI
from pydantic import BaseModel
from langchain_groq import ChatGroq
import os
# Create a FastAPI instance
app = FastAPI()
groq_api_key="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"

print(groq_api_key)
llm =ChatGroq(groq_api_key=groq_api_key, model_name="llama-3.1-70b-versatile")

print(llm)
def llm_answer(query, context):
    prompt = f"""
        Context: {context}

        You are a knowledgeable assistant designed to answer questions based on the provided context. The context consists of extracted text from a PDF document. Please provide a detailed and informative answer to the following question based on this context. 

        If the context does not contain relevant information to answer the question, clearly state that you cannot provide an answer instead of giving irrelevant or inaccurate information.

        Question: {query}

        Answer:
    """
    response = llm.invoke(prompt)
    return response.content
    
# Define a data model for the request body
class Item(BaseModel):
    name: str
    description: str


# POST endpoint that receives data from the Streamlit frontend
@app.post("/items/")
def create_item(item: Item):
    return {
        "message": f"Item '{item.name}' created with description: '{item.description}'"
    }

class New(BaseModel):
    query: str
    context: str  

@app.post("/AI_Answer/")
def check(new: New):
    return {"message": llm_answer(new.query,new.context)}