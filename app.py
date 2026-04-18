"""
app.py
A Streamlit interface for the Mental Health Support Chatbot.
Run with: streamlit run app.py
"""

import streamlit as st
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# Title and Description
st.set_page_config(page_title="EmpathyBot", page_icon="💙", layout="centered")
st.title("💙 EmpathyBot - Mental Health Support")
st.markdown("A supportive, non-judgmental AI fine-tuned on empathetic dialogues. *Not a substitute for professional mental healthcare.*")

# Load model from the Hugging Face Hub!
@st.cache_resource
def load_bot():
    # TODO: Replace 'YourUsername' with your actual Hugging Face profile name!
    model_id = "shabanaftab01/mental-health-support-bot" 
    
    st.sidebar.info("Downloading model from Hugging Face... this takes a minute on first boot!")
    
    # We use CPU by default for the free tier on Streamlit Cloud
    pipe = pipeline("text-generation", model=model_id, tokenizer=model_id, device=-1)
    
    st.sidebar.success("Model loaded successfully!")
    return pipe, pipe.tokenizer

pipe, tokenizer = load_bot()

# Initialize session state for conversation history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Quick safety filter
crisis_keywords = ["suicide", "kill myself", "want to die", "end my life"]

# User input
if prompt := st.chat_input("I'm here for you. How are you feeling today?"):
    # Show user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Check for crisis
    if any(k in prompt.lower() for k in crisis_keywords):
        response = "I hear how much pain you're in, and I want you to know you're not alone. Please reach out to emergency services or a crisis lifeline immediately. They can offer real help right now."
        with st.chat_message("assistant"):
            st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})
    else:
        # Generate response using our fine-tuned prompt format
        # Format: <|user|> user_text <|bot|>
        formatted_prompt = f"<|user|> {prompt} <|bot|>"
        
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                # We tell the model to stop generating when it hits eos_token or next user token
                outputs = pipe(
                    formatted_prompt, 
                    max_new_tokens=60, 
                    do_sample=True, 
                    temperature=0.7, 
                    top_p=0.9,
                    repetition_penalty=1.3,
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    truncation=True
                )
                
                generated_text = outputs[0]['generated_text']
                # Extract only the bot's response
                if "<|bot|>" in generated_text:
                    response = generated_text.split("<|bot|>")[-1].split("<|user|>")[0].strip()
                else:
                    response = generated_text
                
                st.markdown(response)
        
        st.session_state.messages.append({"role": "assistant", "content": response})
