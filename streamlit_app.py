import os
import streamlit as st
from dotenv import load_dotenv

# Document loading and splitting
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Google GenAI Embeddings & LLM
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI

# Vector storage and chains
from langchain_community.vectorstores import FAISS
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# Page settings
st.set_page_config(
    page_title="Taran's AI Assistant",
    page_icon="🤖",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Adaptive Theme CSS (Works seamlessly in Light & Dark Mode)
st.markdown("""
    <style>
    /* Completely hide sidebar and collapse button */
    [data-testid="stSidebarCollapseButton"], section[data-testid="stSidebar"] {
        display: none !important;
    }

    /* Clean padding inside embedded iframe */
    .block-container {
        padding-top: 1rem !important;
        padding-bottom: 2rem !important;
        padding-left: 0.8rem !important;
        padding-right: 0.8rem !important;
    }

    /* Header text using theme variables */
    h2 {
        font-size: 1.15rem !important;
        font-weight: 700 !important;
        color: var(--text-color) !important;
        margin: 0 !important;
    }

    .stCaption {
        font-size: 0.8rem !important;
        opacity: 0.8 !important;
    }

    .stChatMessage p {
        font-size: 0.88rem !important;
        line-height: 1.5 !important;
        color: var(--text-color) !important;
    }

    /* Adaptive Info 'i' Button */
    div[data-testid="stColumn"]:nth-child(2) button {
        border-radius: 50% !important;
        width: 32px !important;
        height: 32px !important;
        padding: 0 !important;
        font-weight: 600 !important;
        border: 1px solid rgba(128, 128, 128, 0.3) !important;
        background-color: var(--secondary-background-color) !important;
        color: #2563eb !important;
        float: right !important;
        transition: transform 0.15s ease, border-color 0.15s ease;
    }

    div[data-testid="stColumn"]:nth-child(2) button:hover {
        transform: scale(1.08);
        border-color: #2563eb !important;
    }

    /* Chat message card styling using native surfaces */
    div[data-testid="stChatMessage"] {
        background-color: var(--secondary-background-color) !important;
        border: 1px solid rgba(128, 128, 128, 0.15) !important;
        border-radius: 12px !important;
        padding: 10px 14px !important;
        margin-bottom: 10px !important;
    }
    </style>
""", unsafe_allow_html=True)

load_dotenv()
if not os.getenv("GOOGLE_API_KEY"):
    st.error("Missing GOOGLE_API_KEY. Check your Streamlit Secrets.")
    st.stop()

# Cache RAG pipeline
@st.cache_resource(show_spinner="Loading Taran's Personal AI Assistant...")
def init_rag_pipeline(pdf_path: str = "portfolio.pdf"):
    if not os.path.exists(pdf_path):
        st.error(f"'{pdf_path}' not found! Place your portfolio PDF in this folder.")
        st.stop()

    loader = PyPDFLoader(pdf_path)
    docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    chunks = text_splitter.split_documents(docs)

    embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-2-preview")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.2
    )

    system_prompt = (
        "You are an AI assistant representing the owner of this portfolio. "
        "Answer questions from recruiters or visitors using only the retrieved context below. "
        "Be concise, professional, and factual. If an answer cannot be found in the context, "
        "state that the information is not in the portfolio and suggest contacting the candidate directly.\n\n"
        "Context:\n{context}"
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])

    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    return create_retrieval_chain(retriever, question_answer_chain)

rag_chain = init_rag_pipeline()

# Info Dialog overlay
@st.dialog("About This Assistant")
def show_about_dialog():
    st.markdown("""
    This bot is built using **LangChain**, **FAISS**, and **Google Gemini** by **Taran**.
    
    It scans the portfolio resume to answer queries regarding technical stack, professional background, and past projects.
    
    📧 Contact: [taranveer.in](https://taranveer.in)
    """)
    st.divider()
    if st.button("🧹 Clear Chat History", use_container_width=True):
        st.session_state.messages = [
            {"role": "assistant", "content": "Chat cleared! How can I help you today?"}
        ]
        st.rerun()

# Top Navigation: Title + Adaptive Info Button
col_title, col_btn = st.columns([0.85, 0.15])

with col_title:
    st.markdown("## 💼 Taran's AI Assistant")
    st.caption("Ask questions about my experience, skills, and projects.")

with col_btn:
    if st.button("ℹ️", key="info_btn", help="About & options"):
        show_about_dialog()

st.divider()

# Chat History setup
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! Ask me anything about my projects, background, or skills."}
    ]

# Render chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# User prompt
if prompt := st.chat_input("Ask a question about my work..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Typing..."):
            response = rag_chain.invoke({"input": prompt})
            answer = response["answer"]
            st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})