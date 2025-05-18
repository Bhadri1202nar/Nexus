import os
from dotenv import load_dotenv
import requests
import streamlit as st
from langchain_groq import ChatGroq
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from summary import api_key
from research import search_arxiv
from attachement import add_attachment

load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")
st.set_page_config(page_title="Groq Langchain Chat", layout="centered")
st.title("🧠 Chat with Langchain and search 🌐")

# Sidebar navigation
with st.sidebar:
    st.markdown("### ☰ Navigation")
    show_history = st.checkbox("Show Prompt History", value=False)

# Initialize chat memory
if "chat_memory" not in st.session_state:
    st.session_state.chat_memory = ConversationBufferMemory(return_messages=True)

# Initialize Groq LLM
llm = ChatGroq(
    temperature=0.7,
    model_name="meta-llama/llama-4-scout-17b-16e-instruct",
    api_key=groq_api_key
)

conversation = ConversationChain(
    llm=llm,
    memory=st.session_state.chat_memory,
    verbose=False
)

# Google search via SERP API
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

# State setup
if "search_google_clicked" not in st.session_state:
    st.session_state.search_google_clicked = False
if "search_research_clicked" not in st.session_state:
    st.session_state.search_research_clicked = False

# Upload any document (max 5 MB)
uploaded_file = st.file_uploader(
    "📎 Upload document for Q&A (PDF, TXT, DOCX, CSV)",
    type=["pdf", "txt", "md", "docx", "doc", "csv"]
)

if uploaded_file:
    if uploaded_file.size > 5 * 1024 * 1024:
        st.error("❌ File too large. Please upload a file under 5MB.")
    else:
        try:
            st.session_state.qa_chain = add_attachment(uploaded_file, llm)
            st.success("✅ File uploaded and processed!")
            st.info(f"📄 Uploaded: `{uploaded_file.name}` ({uploaded_file.type})")
        except ValueError as e:
            st.error(f"❌ {e}")

# Chat input
user_input = st.chat_input("Ask Something")

# Action buttons
col1, col2 = st.columns(2)
with col1:
    if st.button("🌐"):
        st.session_state.search_google_clicked = True
with col2:
    if st.button("💡"):
        st.session_state.search_research_clicked = True

# Perform actions
if st.session_state.search_google_clicked and user_input:
    with st.spinner("Searching Google..."):
        result = search_google(user_input)
        st.chat_message("user").write(user_input)
        st.chat_message("assistant").write(result)
    st.session_state.search_google_clicked = False

elif st.session_state.search_research_clicked and user_input:
    with st.spinner("Searching arXiv..."):
        result = search_arxiv(user_input)
        st.chat_message("user").write(user_input)
        st.chat_message("assistant").write(result)
    st.session_state.search_research_clicked = False

elif user_input:
    with st.spinner("Thinking..."):
        if "qa_chain" in st.session_state:
            response = st.session_state.qa_chain.run(user_input)
        else:
            response = conversation.predict(input=user_input)
        st.chat_message("user").write(user_input)
        st.chat_message("assistant").write(response)

# Prompt history in sidebar
if show_history:
    with st.sidebar:
        st.markdown("### 🕘 Prompt History")
        for msg in st.session_state.chat_memory.chat_memory.messages:
            st.markdown(f"**{msg.type.title()}**: {msg.content}")
