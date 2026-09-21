import streamlit as st
import torch
# Import your existing pipeline functions
from src.retrieval.rag_pipeline import build_hr_agent 

st.set_page_config(page_title="Enterprise HR Agent", page_icon="🤖")

st.title("Enterprise GenAI HR Agent 🏢")
st.markdown("Ask me questions about Bengaluru leave policies, IT asset requisitions, or try to ask for PII to test my DPO privacy guardrails!")

# Initialize the model only once using Streamlit's cache
@st.cache_resource
def load_agent():
    return build_hr_agent()

with st.spinner("Loading Phi-3 Model and FAISS Database (This takes a moment)..."):
    agent = load_agent()

# Chat Interface
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

prompt = st.chat_input("E.g., What is the policy for Privilege Leave?")

if prompt:
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Generate and display agent response
    with st.chat_message("assistant"):
        with st.spinner("Searching policies..."):
            result = agent.invoke({"input": prompt})
            answer = result.get("answer", "Error generating response.")
            st.markdown(answer)
            
            # Optional: Show retrieved sources
            with st.expander("View Retrieved Context"):
                for i, doc in enumerate(result.get("context", [])):
                    st.write(f"**Chunk {i+1}:** {doc.page_content}")

    st.session_state.messages.append({"role": "assistant", "content": answer})