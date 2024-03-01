import random
import requests
import streamlit as st
from dataclasses import dataclass


BACKEND_URL = "http://localhost:3000"


def test_func():
    return random.choice(("hello", "Ola", "hi"))


@dataclass
class Message:
    actor: str
    payload: str


USER = "user"
ASSISTANT = "ai"
MESSAGES = "messages"

st.title("Legal Chatbot_v1 ⚖")
if MESSAGES not in st.session_state:
    st.session_state[MESSAGES] = [Message(actor=ASSISTANT, payload="Hi! How can I help you?")]

msg: Message
for msg in st.session_state[MESSAGES]:
    st.chat_message(msg.actor).write(msg.payload)

prompt: str = st.chat_input("Enter a prompt here")

if prompt:
    st.session_state[MESSAGES].append(Message(actor=USER, payload=prompt))
    st.chat_message(USER).write(prompt)
    res = requests.post(f"{BACKEND_URL}/api/chat",
                        params={"request": prompt})
    # res = test_func()
    st.session_state[MESSAGES].append(Message(actor=ASSISTANT, payload=res.json()["answer"]))
    st.chat_message(ASSISTANT).write(res.json()["answer"])