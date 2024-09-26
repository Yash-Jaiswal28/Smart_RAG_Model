import streamlit as st
import requests

def app():
    st.title("Page 1")
    
    # Example of calling another FastAPI endpoint
    response = requests.get("http://127.0.0.1:8000/get_data")
    if response.status_code == 200:
        data = response.json()
        st.write("Received data from FastAPI:", data["data"])
    else:
        st.write("Error fetching data from FastAPI.")
