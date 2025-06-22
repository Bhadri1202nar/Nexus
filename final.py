import os

import tempfile

from dotenv import load_dotenv
import requests
import streamlit as st
from langchain_groq import ChatGroq
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from pyarrow import duration
from summary import api_key
from research import search_arxiv
from attachment import add_attachment
from audio_utils import record_audio , transcribe_audio

# Load environment variables
load_dotenv()
MODEL_OPTIONS = {
    "Groq" : {
    "LLaMA 4 Scout 17B": "meta-llama/llama-4-scout-17b-16e-instruct",
    "Mixtral 8x7B": "mistral-saba-24b",
    "Gemma 7B": "gemma2-9b-it"},
    "Gemini" : {"Gemini 1.5 Flash":"gemini-1.5-flash" ,
                "Gemini 2.0 Flash":"gemini-2.0-flash"

    }
    # Add more as supported by Groq
}


# Set Streamlit page config
st.set_page_config(page_title="Groq Langchain Chat", layout="centered")
st.title("🧠 Chat with Langchain and search 🌐")
provider=st.selectbox("🧠 Choose a provider",list(MODEL_OPTIONS.keys()))
model_labels=list(MODEL_OPTIONS[provider].keys())
selected_model_label=st.selectbox("🧠 Choose a model",model_labels)
selected_model=MODEL_OPTIONS[provider][selected_model_label]
print(selected_model)
#st.markdown(f"Current model:{selected_model}")

# Initialize session state
if "chat_memory" not in st.session_state:
    st.session_state.chat_memory = ConversationBufferMemory(return_messages=True)
if "conversation" not in st.session_state:
    groq_api_key = os.getenv("GROQ_API_KEY")
    llm = ChatGroq(
        temperature=0.7,
        model_name=selected_model,
        api_key=groq_api_key
    )
    st.session_state.conversation = ConversationChain(
        llm=llm,
        memory=st.session_state.chat_memory,
        verbose=False
    )
if "conversations" not in st.session_state:
    st.session_state.conversations = []  # List of message lists
if "active_convo_index" not in st.session_state:
    st.session_state.active_convo_index = None
if "search_google_clicked" not in st.session_state:
    st.session_state.search_google_clicked = False
if "search_research_clicked" not in st.session_state:
    st.session_state.search_research_clicked = False

# Sidebar Navigation & History
with st.sidebar:
    st.markdown("### ☰ Navigation")
    show_history = st.checkbox("Show Prompt History", value=False)

    st.markdown("### 🧾 Past Conversations")
    for i, convo in enumerate(st.session_state.conversations):
        label = convo[0]["content"][:30] + "..." if convo else f"Conversation {i+1}"
        if st.button(label, key=f"convo_{i}"):
            st.session_state.active_convo_index = i
            st.session_state.chat_memory = ConversationBufferMemory(return_messages=True)
            for msg in convo:
                if msg["type"] == "user":
                    st.session_state.chat_memory.chat_memory.add_user_message(msg["content"])
                else:
                    st.session_state.chat_memory.chat_memory.add_ai_message(msg["content"])
            st.session_state.conversation = ConversationChain(
                llm=st.session_state.conversation.llm,
                memory=st.session_state.chat_memory,
                verbose=False
            )
            st.session_state.show_conversation_preview = True

        # Google Search Function
def search_google(query):
    serp_api_key = os.getenv("SERP_API_KEY")
    params = {"q": query, "api_key": serp_api_key, "engine": "google"}
    url = "https://serpapi.com/search"
    response = requests.get(url, params=params)
    if response.status_code == 200:
        results = response.json().get("organic_results", [])
        if not results:
            return "No results found."
        output = ""
        for i, result in enumerate(results[:3], start=1):
            title = result.get("title")
            link = result.get("link")
            snippet = result.get("snippet", "")
            output += f"**{i}. [{title}]({link})**\n\n{snippet}\n\n"
        return output
    return "❌ Error fetching results from SERP API"

# File Upload
uploaded_files = st.file_uploader(
    "📎 Upload document for Q&A (PDF, TXT, DOCX, CSV)",
    type=["pdf", "txt", "md", "docx", "doc", "csv","png", "jpg", "jpeg"],accept_multiple_files=True
)

if uploaded_files:
    too_large = [f.name for f in uploaded_files if f.size > 5 * 1024 * 1024]
    if too_large:
        st.error(f"❌ These files are too large (>5MB): {', '.join(too_large)}")
    else:
        try:
            st.session_state.qa_chain = add_attachment(uploaded_files, st.session_state.conversation.llm)
            st.success("✅ Files uploaded and processed!")
            for f in uploaded_files:
                st.info(f"📄 Uploaded: `{f.name}` ({f.type})")
        except ValueError as e:
            st.error(f"❌ {e}")

# ✅ Show selected conversation if user just clicked it
if st.session_state.get("show_conversation_preview", False):
    st.markdown("### 💬 Restored Conversation")
    for msg in st.session_state.conversations[st.session_state.active_convo_index]:
        st.chat_message(msg["type"]).write(msg["content"])
    # reset flag after showing
    st.session_state.show_conversation_preview = False

# Chat input
user_input = st.chat_input("Ask Something")
#Audio input

# Action Buttons
col1, col2, col3 = st.columns(3)
with col1:
    if st.button("🌐"):
        st.session_state.search_google_clicked = True
with col2:
    if st.button("💡"):
        st.session_state.search_research_clicked = True
with col3:
    if st.button("🎙️"):
        st.info("🎤 Recording... Speak now.")
        audio_path = record_audio(duration=10)
        st.success("✅ Recording complete! Transcribing...")

        user_input = transcribe_audio(audio_path)
        st.success(f"📄 You said: {user_input}")

# Execute Actions
if user_input:
    user_msg = {"type": "user", "content": user_input}

    if st.session_state.search_google_clicked:
        with st.spinner("Searching Google..."):
            response = search_google(user_input)
    elif st.session_state.search_research_clicked:
        with st.spinner("Searching arXiv..."):
            response = search_arxiv(user_input)
    elif "qa_chain" in st.session_state:
        with st.spinner("Answering from uploaded document..."):
            response = st.session_state.qa_chain.run(user_input)
    else:
        with st.spinner("Thinking..."):
            response = st.session_state.conversation.predict(input=user_input)

    assistant_msg = {"type": "assistant", "content": response}

    if st.session_state.active_convo_index is None:
        st.session_state.active_convo_index = len(st.session_state.conversations)
        st.session_state.conversations.append([])

    st.session_state.conversations[st.session_state.active_convo_index].append(user_msg)
    st.session_state.conversations[st.session_state.active_convo_index].append(assistant_msg)



    st.session_state.search_google_clicked = False
    st.session_state.search_research_clicked = False

# Show chat history of selected conversation
if st.session_state.active_convo_index is not None:
    for msg in st.session_state.conversations[st.session_state.active_convo_index]:
        st.chat_message(msg["type"]).write(msg["content"])

# Prompt History (if enabled)

