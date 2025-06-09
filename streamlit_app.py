import streamlit as st
from atc_agent import atc_agent  # Make sure this is importable

st.title("🛫 ATC Agent Chat")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

user_input = st.chat_input("Ask the ATC Agent...")

if user_input:
    st.session_state.chat_history.append({"role": "user", "text": user_input})
    with st.spinner("Thinking..."):
        response = atc_agent.run(user_input)
    st.session_state.chat_history.append({"role": "agent", "text": response})

for msg in st.session_state.chat_history:
    if msg["role"] == "user":
        st.chat_message("user").markdown(msg["text"])
    else:
        st.chat_message("assistant").markdown(msg["text"])
