import os
import streamlit as st
from dotenv import load_dotenv
import requests

# Load environment variables from .env file
load_dotenv()

# Set your Hugging Face API token
api_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
API_URL = "https://api-inference.huggingface.co/models/EleutherAI/gpt-neox-20b"
headers = {"Authorization": f"Bearer {api_token}"}

def query(payload):
    response = requests.post(API_URL, headers=headers, json=payload)
    return response.json()

# Streamlit app layout
st.title("GPT-NeoX-20B Text Generation")
st.sidebar.header("Instructions")
st.sidebar.write("""
1. Enter a prompt in the text area.
2. Use the slider to set the maximum length of the generated text.
3. Click the "Generate" button to see the output.
""")

prompt = st.text_area("Prompt", "Once upon a time...")
max_length = st.slider("Max Length", 10, 100, 50)

if st.button("Generate"):
    try:
        output = query({
            "inputs": prompt,
            "parameters": {"max_length": max_length}
        })
        
        # Check if the output is a list and handle accordingly
        if isinstance(output, list) and len(output) > 0:
            generated_text = output[0].get("generated_text", "No output generated.")
        else:
            generated_text = "No output generated."
        
        st.write(generated_text)
    except Exception as e:
        st.error(f"Error generating text: {e}")
