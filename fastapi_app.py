from fastapi import FastAPI
from pydantic import BaseModel

# Create a FastAPI instance
app = FastAPI()

# Define a data model for the request body
class Item(BaseModel):
    name: str
    description: str

# POST endpoint that receives data from the Streamlit frontend
@app.post("/items/")
def create_item(item: Item):
    return {
        "message": f"Item '{item.name}' created with description: '{item.description}'",
        "current":"kya haal"
        }
