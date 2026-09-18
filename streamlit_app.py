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
    page_icon="🤖",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# 1. Custom CSS applied for the Header, Chat Bubbles, and Typing Animation
st.markdown("""
    <style>
    /* Hide Streamlit default chrome */
    #MainMenu, header, footer { visibility: hidden !important; height: 0 !important; }
    [data-testid="stHeader"] { display: none !important; }
    [data-testid="stSidebarCollapseButton"], section[data-testid="stSidebar"] { display: none !important; }

    /* Clean padding for iframe embed */
    .block-container {
        padding-top: 0.8rem !important;
        padding-bottom: 4rem !important;
        padding-left: 0.5rem !important;
        padding-right: 0.5rem !important;
        max-width: 100% !important;
    }

    /* ----------------------------------------------------
       HEADER: Force single row, #5166D8 background
    ---------------------------------------------------- */
    /* Target the st.columns container to prevent mobile stacking */
    div[data-testid="stHorizontalBlock"] {
        flex-wrap: nowrap !important; 
        background-color: #5166D8 !important;
        padding: 12px 16px !important;
        border-radius: 12px !important;
        align-items: center !important;
        margin-bottom: 15px !important;
        box-shadow: 0 4px 10px rgba(81, 102, 216, 0.25) !important;
    }

    .title-text {
        font-size: 1.15rem !important;
        font-weight: 700 !important;
        color: #ffffff !important;
        margin: 0 !important;
        line-height: 1.2 !important;
    }

    .status-badge {
        display: inline-flex;
        align-items: center;
        gap: 5px;
        font-size: 0.72rem;
        color: #e0e7ff !important;
        font-weight: 500;
        margin-top: 3px;
    }

    .status-dot {
        width: 7px;
        height: 7px;
        background-color: #4ade80; /* bright online green */
        border-radius: 50%;
        display: inline-block;
    }

    /* Info button styling (Top Right) */
    div[data-testid="stColumn"]:last-child {
        display: flex !important;
        justify-content: flex-end !important;
        min-width: 40px !important;
    }

    div[data-testid="stColumn"]:last-child button {
        border-radius: 50% !important;
        width: 34px !important;
        height: 34px !important;
        padding: 0 !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        border: 1px solid rgba(255, 255, 255, 0.4) !important;
        background-color: rgba(255, 255, 255, 0.1) !important;
        color: #ffffff !important;
        transition: all 0.2s ease !important;
    }

    div[data-testid="stColumn"]:last-child button:hover {
        background-color: rgba(255, 255, 255, 0.25) !important;
        transform: scale(1.05);
    }

    /* ----------------------------------------------------
       CHAT BUBBLES: Fix text bleed and height constraints
    ---------------------------------------------------- */
    [data-testid="stChatMessage"] {
        background-color: var(--secondary-background-color) !important;
        border: 1px solid rgba(128, 128, 128, 0.15) !important;
        border-radius: 14px !important;
        padding: 12px 14px !important;
        margin-bottom: 12px !important;
        height: auto !important;             /* Fixes the bounding box text spill */
        min-height: min-content !important; 
        display: flex !important;
        align-items: flex-start !important;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.04) !important;
    }

    [data-testid="stChatMessage"] > div {
        overflow: visible !important;
    }

    [data-testid="stChatMessage"] p {
        font-size: 0.9rem !important;
        line-height: 1.5 !important;
        color: var(--text-color) !important;
        margin: 0 !important;
        word-wrap: break-word !important;
        white-space: pre-wrap !important;
    }

    /* ----------------------------------------------------
       TYPING ANIMATION (3 Dots)
    ---------------------------------------------------- */
    .typing-indicator {
        display: inline-flex;
        align-items: center;
        gap: 5px;
        padding: 4px 8px;
    }
    .typing-indicator span {
        width: 6px;
        height: 6px;
        background-color: #5166D8;
        border-radius: 50%;
        animation: typingBounce 1.4s infinite ease-in-out both;
    }
    .typing-indicator span:nth-child(1) { animation-delay: -0.32s; }
    .typing-indicator span:nth-child(2) { animation-delay: -0.16s; }
    
    @keyframes typingBounce {
        0%, 80%, 100% { transform: scale(0); opacity: 0.5; }
        40% { transform: scale(1); opacity: 1; }
    }
    </style>
""", unsafe_allow_html=True)

load_dotenv()
if not os.getenv("GOOGLE_API_KEY"):
    st.error("Missing GOOGLE_API_KEY. Please verify your environment variables.")
    st.stop()

@st.cache_resource(show_spinner="Connecting to Taran's Knowledge Base...")
def init_rag_pipeline(pdf_path: str = "portfolio.pdf"):
    if not os.path.exists(pdf_path):
        st.error(f"'{pdf_path}' not found! Place portfolio.pdf in this folder.")
        st.stop()

    loader = PyPDFLoader(pdf_path)
    docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.split_documents(docs)

    embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-2-preview")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

    llm = ChatGoogleGenerativeAI(
        model="gemini-3.5-flash-lite",
        temperature=0.2,
        max_output_tokens=150
    )

    system_prompt = (
        "You are the official portfolio AI assistant representing Taranveer Singh. "
        "Answer questions strictly using the retrieved context below. "
        "Keep answers short and concise (1 to 2 sentences maximum). "
        "If an answer is not in the context, state that briefly and suggest contacting Taran via taranveer.in.\n\n"
        "Context:\n{context}"
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])

    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    return create_retrieval_chain(retriever, question_answer_chain)

rag_chain = init_rag_pipeline()

@st.dialog("About Taran's Assistant")
def show_about_dialog():
    st.markdown("""
    **Taranveer Singh — Portfolio Assistant**
    
    Full-Stack PHP Developer building reliable web applications with Laravel, React JS, and Modern AI.
    * **Portfolio:** [taranveer.in](https://taranveer.in)
    * **Email:** staranveer178@gmail.com
    """)
    st.divider()
    if st.button("🧹 Clear Chat History", use_container_width=True):
        st.session_state.messages = [
            {"role": "assistant", "content": "Chat reset! Feel free to ask about Taran's projects or experience."}
        ]
        st.rerun()

# 2. Header UI
col_header, col_info = st.columns([0.85, 0.15])

with col_header:
    st.markdown("""
        <div class="title-text">💼 Taranveer Singh</div>
        <div class="status-badge"><span class="status-dot"></span> Online • Portfolio AI Assistant</div>
    """, unsafe_allow_html=True)

with col_info:
    if st.button("ℹ️", key="info_btn", help="About Taran"):
        show_about_dialog()

# Initialize history
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant", 
            "content": "Hi! I'm Taran's portfolio assistant. Ask me anything about his technical stack, past projects, or background."
        }
    ]

# 3. Render previous chat history with Custom Avatars (🤖 and 👤)
for msg in st.session_state.messages:
    avatar_icon = "🤖" if msg["role"] == "assistant" else "👤"
    with st.chat_message(msg["role"], avatar=avatar_icon):
        st.markdown(msg["content"])

# Handle user input
if prompt := st.chat_input("Ask about skills, projects, or experience..."):
    # Append user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar="👤"):
        st.markdown(prompt)

    # 4. Generate Assistant Response with Typing Animation
    with st.chat_message("assistant", avatar="🤖"):
        # Create an empty placeholder to hold the typing animation
        message_placeholder = st.empty()
        
        # Inject HTML for the animated 3-dot pill
        message_placeholder.markdown("""
            <div class="typing-indicator">
              <span></span><span></span><span></span>
            </div>
        """, unsafe_allow_html=True)
        
        # Fetch the response
        response = rag_chain.invoke({"input": prompt})
        answer = response["answer"]
        
        # Replace the typing animation with the actual text response
        message_placeholder.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})