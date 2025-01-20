import os
import streamlit as st
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModelForCausalLM

# Load environment variables from .env file
load_dotenv()

# Set your Hugging Face API token
api_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")

# Load the model and tokenizer
model_name = "EleutherAI/gpt-neox-20b"
tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=api_token)
model = AutoModelForCausalLM.from_pretrained(model_name, use_auth_token=api_token)

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
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(**inputs, max_length=max_length)
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    st.write(generated_text)
