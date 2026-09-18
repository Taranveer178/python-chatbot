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
    page_title="Taranveer Singh | AI Assistant",
    page_icon="💼",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Robust, Adaptive CSS (Light & Dark Theme Compatible)
st.markdown("""
    <style>
    /* 1. Hide default Streamlit chrome, header, sidebar arrow, and footer */
    #MainMenu, header, footer {
        visibility: hidden !important;
        height: 0 !important;
    }
    [data-testid="stHeader"] {
        display: none !important;
    }
    [data-testid="stSidebarCollapseButton"], section[data-testid="stSidebar"] {
        display: none !important;
    }

    /* 2. Responsive layout container with zero iframe waste */
    .block-container {
        padding-top: 0.8rem !important;
        padding-bottom: 4rem !important;
        padding-left: 0.8rem !important;
        padding-right: 0.8rem !important;
        max-width: 100% !important;
    }

    /* 3. Header styling */
    .header-container {
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 0.5rem;
    }

    .title-text {
        font-size: 1.15rem !important;
        font-weight: 700 !important;
        color: var(--text-color) !important;
        margin: 0 !important;
        line-height: 1.2 !important;
    }

    .status-badge {
        display: inline-flex;
        align-items: center;
        gap: 5px;
        font-size: 0.72rem;
        color: #16a34a;
        font-weight: 500;
        margin-top: 2px;
    }

    .status-dot {
        width: 7px;
        height: 7px;
        background-color: #16a34a;
        border-radius: 50%;
        display: inline-block;
    }

    /* 4. Top-Right Info Button styling */
    div[data-testid="stColumn"]:last-child button {
        border-radius: 50% !important;
        width: 32px !important;
        height: 32px !important;
        padding: 0 !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        border: 1px solid rgba(128, 128, 128, 0.25) !important;
        background-color: var(--secondary-background-color) !important;
        color: #2563eb !important;
        float: right !important;
        transition: transform 0.15s ease, border-color 0.15s ease;
    }

    div[data-testid="stColumn"]:last-child button:hover {
        transform: scale(1.08);
        border-color: #2563eb !important;
    }

    /* 5. Chat message bubbles & typography */
    [data-testid="stChatMessage"] {
        background-color: var(--secondary-background-color) !important;
        border: 1px solid rgba(128, 128, 128, 0.15) !important;
        border-radius: 14px !important;
        padding: 10px 14px !important;
        margin-bottom: 8px !important;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.03) !important;
    }

    [data-testid="stChatMessage"] p {
        font-size: 0.88rem !important;
        line-height: 1.45 !important;
        color: var(--text-color) !important;
        margin-bottom: 0 !important;
    }

    /* 6. Clean Chat Input field */
    [data-testid="stChatInput"] {
        border-radius: 20px !important;
    }
    </style>
""", unsafe_allow_html=True)

load_dotenv()
if not os.getenv("GOOGLE_API_KEY"):
    st.error("Missing GOOGLE_API_KEY. Please set it in Streamlit Secrets or your local .env file.")
    st.stop()

# Cache pipeline so PDF parsing and embedding happen only once
@st.cache_resource(show_spinner="Connecting to Taran's Knowledge Base...")
def init_rag_pipeline(pdf_path: str = "portfolio.pdf"):
    if not os.path.exists(pdf_path):
        st.error(f"'{pdf_path}' not found! Please ensure portfolio.pdf is in the project root.")
        st.stop()

    loader = PyPDFLoader(pdf_path)
    docs = loader.load()

    # Smaller chunk size minimizes token load per request
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.split_documents(docs)

    embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-2-preview")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    
    # Retrieve top 2 most relevant chunks to preserve token quota
    retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

    llm = ChatGoogleGenerativeAI(
        model="gemini-3.5-flash-lite",
        temperature=0.2,
        max_output_tokens=150
    )

    system_prompt = (
        "You are the official portfolio AI assistant representing Taranveer Singh, "
        "a Full-Stack PHP Developer specializing in PHP, Laravel, React JS, and MySQL. "
        "Answer questions from recruiters and visitors strictly using the retrieved context. "
        "Keep answers short, direct, and concise (1 to 2 sentences maximum). "
        "If an answer is not in the context, state that briefly and recommend contacting Taran directly "
        "via email at staranveer178@gmail.com or via https://taranveer.in.\n\n"
        "Context:\n{context}"
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])

    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    return create_retrieval_chain(retriever, question_answer_chain)

rag_chain = init_rag_pipeline()

# Right-hand "ℹ️" Modal Dialog
@st.dialog("About Taran's Assistant")
def show_about_dialog():
    st.markdown("""
    **Taranveer Singh — Portfolio Assistant**
    
    Full-Stack PHP Developer building reliable, business-focused web applications with **Laravel, React JS, MySQL, and Modern AI**.
    
    * **Portfolio:** [taranveer.in](https://taranveer.in)
    * **Email:** [staranveer178@gmail.com](mailto:staranveer178@gmail.com)
    * **Stack:** LangChain • FAISS • Gemini API
    """)
    st.divider()
    if st.button("🧹 Clear Chat History", use_container_width=True):
        st.session_state.messages = [
            {"role": "assistant", "content": "Chat reset! Feel free to ask about Taran's projects, tech stack, or experience."}
        ]
        st.rerun()

# Header: Left side Title/Status, Right side Info Button
col_header, col_info = st.columns([0.86, 0.14])

with col_header:
    st.markdown("""
        <div class="title-text">💼 Taranveer Singh</div>
        <div class="status-badge"><span class="status-dot"></span> Online • Portfolio AI Assistant</div>
    """, unsafe_allow_html=True)

with col_info:
    if st.button("ℹ️", key="info_btn", help="About Taran & Options"):
        show_about_dialog()

st.write("")  # Lightweight spacer

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Hi! I'm Taran's portfolio assistant. Ask me anything about his technical stack, past projects, or background."
        }
    ]

# Render conversation history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# User query handling
if prompt := st.chat_input("Ask about skills, projects, or experience..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Answering..."):
            response = rag_chain.invoke({"input": prompt})
            answer = response["answer"]
            st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})