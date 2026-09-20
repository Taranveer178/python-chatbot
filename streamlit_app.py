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

# Page settings - Sidebar disabled entirely for embed mode
st.set_page_config(
    page_title="Taran's AI Assistant",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Avatar URLs
BOT_AVATAR_URL = "https://taranveer.in/img/chatbot.webp"
USER_AVATAR_URL = "https://taranveer.in/img/user.webp"

# Custom CSS for embed UI
st.markdown(f"""
    <style>
    /* 1. Hide default Streamlit elements */
    header[data-testid="stHeader"], 
    [data-testid="collapsedControl"], 
    footer, 
    .viewerBadge_container, 
    .stDeployButton, 
    [data-testid="stToolbar"], 
    [data-testid="stDecoration"] {{
        display: none !important; 
        visibility: hidden !important;
    }}

    /* 2. Remove default app background and borders */
    .stApp {{
        background: transparent !important;
        border: none !important;
    }}
    
    /* 3. Outer padding */
    .block-container {{
        padding: 15px 15px 100px 15px !important; 
        max-width: 100% !important;
    }}

    /* 4. Chat Input Focus Border */
    [data-testid="stChatInput"] textarea,
    [data-testid="stChatInput"] textarea:focus {{
        border: none !important;
        outline: none !important;
        box-shadow: none !important;
    }}

    [data-testid="stChatInput"] > div:focus-within {{
        border-color: #2563EB !important;
        box-shadow: 0 0 0 1px #2563EB !important;
    }}

    /* Base Chat Layout */
    [data-testid="stChatMessage"] {{
        background-color: transparent !important;
        border: none !important;
        padding: 0 !important;
        margin-bottom: 16px !important;
        display: flex !important;
        align-items: flex-start !important;
        width: 100% !important;
        gap: 12px !important;
    }}

    /* 5. Avatar Styling */
    /* Bot Avatar */
    [data-testid="stChatMessage"]:has(img[src*="chatbot.webp"]) img {{
        background-color: #2563EB !important;
        border-radius: 50% !important;
        padding: 6px !important; 
        object-fit: contain !important;
        width: 38px !important;
        height: 38px !important;
    }}

    /* User Avatar */
    [data-testid="stChatMessage"]:has(img[src*="user.webp"]) img,
    [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) div {{
        background-color: transparent !important;
        border-radius: 50% !important;
        padding: 0px !important;
        object-fit: cover !important;
        width: 38px !important;
        height: 38px !important;
    }}

    /* 6. Message Alignment - The structural fix */
    [data-testid="stChatMessage"]:has(img[src*="user.webp"]) {{
        flex-direction: row-reverse !important;
    }}
    
    /* Force the wrapper to act as a full-width flex column */
    [data-testid="stChatMessage"] div[data-testid="stChatMessageContent"] {{
        display: flex !important;
        flex-direction: column !important;
        width: 100% !important;
    }}

    /* 7. Bubble Styles - Snaps the bubble perfectly left or right */
    
    /* Bot Bubble */
    [data-testid="stChatMessage"]:has(img[src*="chatbot.webp"]) div[data-testid="stMarkdownContainer"] {{
        background-color: #F9FAFB !important;
        color: #1F2937 !important;
        padding: 12px 16px !important; 
        border-radius: 0px 16px 16px 16px !important;
        border: 1px solid #E5E7EB !important;
        box-shadow: 0 1px 2px rgba(0,0,0,0.02) !important;
        width: fit-content !important;
        max-width: 85% !important;
        align-self: flex-start !important; /* Snaps to left */
    }}

    /* User Bubble */
    [data-testid="stChatMessage"]:has(img[src*="user.webp"]) div[data-testid="stMarkdownContainer"] {{
        background-color: #2563EB !important;
        color: white !important;
        padding: 12px 16px !important; 
        border-radius: 16px 16px 0px 16px !important;
        box-shadow: 0 1px 2px rgba(0,0,0,0.05) !important;
        width: fit-content !important; /* Shrinks wrapping box to text size */
        max-width: 85% !important;
        align-self: flex-end !important; /* Snaps perfectly to the right */
        margin-left: auto !important;    /* Fallback push to the right */
    }}

    /* Zero out internal text margins to maintain exact padding balance */
    [data-testid="stChatMessage"] div[data-testid="stMarkdownContainer"] p {{
        margin: 0 !important;
        padding: 0 !important;
        line-height: 1.5 !important;
        font-size: 14.5px !important;
    }}
    [data-testid="stChatMessage"]:has(img[src*="user.webp"]) div[data-testid="stMarkdownContainer"] p {{
        color: white !important;
    }}

    /* 8. Fix for typing spinner */
    [data-testid="stChatMessage"] div[data-testid="stSpinner"] {{
        display: flex !important;
        align-items: center !important;
        gap: 8px !important;
        white-space: nowrap !important;
        width: fit-content !important;
        min-width: 120px !important;
        align-self: flex-start !important;
    }}
    [data-testid="stChatMessage"] div[data-testid="stSpinner"] > div {{
        white-space: nowrap !important;
        overflow: visible !important;
    }}
    </style>
""", unsafe_allow_html=True)

load_dotenv()
if not os.getenv("GOOGLE_API_KEY"):
    st.error("Missing GOOGLE_API_KEY. Please verify your environment variables or Streamlit Secrets.")
    st.stop()

@st.cache_resource(show_spinner="Loading Taran's Personal AI Assistant...")
def init_rag_pipeline(pdf_path: str = "portfolio.pdf"):
    if not os.path.exists(pdf_path):
        st.error(f"'{pdf_path}' not found! Place your portfolio PDF in this folder.")
        st.stop()

    loader = PyPDFLoader(pdf_path)
    docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=450, chunk_overlap=40)
    chunks = text_splitter.split_documents(docs)

    embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-2-preview")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

    llm = ChatGoogleGenerativeAI(
        model="gemini-3.5-flash-lite",
        temperature=0.2
    )

    system_prompt = (
        "You are an AI assistant representing Taranveer Singh's portfolio. "
        "Answer questions using ONLY the provided context. "
        "CRITICAL: Summarize your answer in 1 or 2 short sentences. Do not use bullet points or long lists. "
        "If the answer is not in the context, reply that it's unlisted and suggest contacting him directly.\n\n"
        "Context:\n{context}"
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])

    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    return create_retrieval_chain(retriever, question_answer_chain)

rag_chain = init_rag_pipeline()

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! Ask me anything about my projects, background, or skills."}
    ]

for msg in st.session_state.messages:
    avatar = USER_AVATAR_URL if msg["role"] == "user" else BOT_AVATAR_URL
    with st.chat_message(msg["role"], avatar=avatar):
        st.markdown(msg["content"])

if prompt := st.chat_input("Ask a question about my work..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar=USER_AVATAR_URL):
        st.markdown(prompt)

    with st.chat_message("assistant", avatar=BOT_AVATAR_URL):
        with st.spinner("Typing..."):
            response = rag_chain.invoke({"input": prompt})
            answer = response["answer"]
            st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})