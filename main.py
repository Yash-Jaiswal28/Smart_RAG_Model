import streamlit as st
import requests
import json

# Streamlit app for sending data to FastAPI
def app():
    st.title("Send Data to FastAPI")

    # Create a simple form in Streamlit
    name = st.text_input("Enter item name:")
    description = st.text_area("Enter item description:")

    if st.button("Send to FastAPI"):
        # Prepare the payload to send to FastAPI
        payload = {
            "name": name,
            "description": description
        }

        # Send the POST request to FastAPI
        try:
            response = requests.post("http://127.0.0.1:8000/items/", json=payload)  # Use json=payload
        except requests.exceptions.RequestException as e:
            st.error(f"Request failed: {str(e)}")
            return
        
        # Display response from FastAPI
        if response.status_code == 200:
            st.success(f"Response from FastAPI: {response.json()['message']}")
        else:
            st.error(f"Failed to send data. Status code: {response.status_code}")

# Run the Streamlit app
if __name__ == "__main__":
    app()
