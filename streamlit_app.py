import streamlit as st
import requests

BACKEND_URL = "http://localhost:3000"


st.title("Legal Chatbot v1")
# st.header("You can ask questions about the Indian Contract Law ⚖")

# Initialize chat history
if "messages " not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("What is the indian contract law?"):
    # Display user message in chat message container
    with st.chat_message("hi"):
        st.markdown(prompt)
    res = requests.post(f"{BACKEND_URL}/api/chat",
                        params={"request": prompt})
    with st.chat_message("assistant"):
        st.markdown(res.json()["answer"])

    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})