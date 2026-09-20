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
    layout="centered"
)

# Avatar URL for the Assistant
AVATAR_URL = "https://taranveer.in/img/Taranveer_logo_light.webp"

# Custom CSS for Professional Chatbot UI (Drift/Intercom style)
st.markdown(f"""
    <style>
    /* Reduce top padding */
    .block-container {{
        padding-top: 2rem !important;
        max-width: 700px;
    }}
    
    /* Remove default background from chat messages */
    [data-testid="stChatMessage"] {{
        background-color: transparent !important;
        border: none !important;
        padding: 0 !important;
        margin-bottom: 20px !important;
    }}

    /* Bot Message Bubble (Left) */
    /* Targets the bot message since we inject an img tag for its avatar */
    [data-testid="stChatMessage"]:has(img) div[data-testid="stMarkdownContainer"] {{
        background-color: #F3F4F6;
        color: #1F2937;
        padding: 12px 18px;
        border-radius: 0px 18px 18px 18px;
        display: inline-block;
        max-width: 85%;
        box-shadow: 0 1px 2px rgba(0,0,0,0.05);
    }}

    /* User Message Bubble (Right) */
    /* Targets the user message which uses the default SVG avatar */
    [data-testid="stChatMessage"]:has(svg) {{
        flex-direction: row-reverse;
    }}
    
    [data-testid="stChatMessage"]:has(svg) div[data-testid="stMarkdownContainer"] {{
        background-color: #2563EB;
        color: white;
        padding: 12px 18px;
        border-radius: 18px 18px 0px 18px;
        display: inline-block;
        max-width: 85%;
        box-shadow: 0 1px 2px rgba(0,0,0,0.1);
    }}
    
    /* Ensure user text is white */
    [data-testid="stChatMessage"]:has(svg) div[data-testid="stMarkdownContainer"] p {{
        color: white !important;
    }}

    /* Hide the default user avatar to match professional UI style */
    [data-testid="stChatMessage"]:has(svg) div[data-testid="chatAvatarIcon-user"] {{
        display: none;
    }}
    
    /* General Chat Text Formatting */
    .stChatMessage p {{
        font-size: 15px !important;
        margin-bottom: 0 !important;
        line-height: 1.5;
    }}
    </style>
""", unsafe_allow_html=True)

load_dotenv()
if not os.getenv("GOOGLE_API_KEY"):
    st.error("Missing GOOGLE_API_KEY. Please verify your environment variables or Streamlit Secrets.")
    st.stop()

# Cache pipeline so PDF parsing and embedding only happen once
@st.cache_resource(show_spinner="Loading Taran's Personal AI Assistant...")
def init_rag_pipeline(pdf_path: str = "portfolio.pdf"):
    if not os.path.exists(pdf_path):
        st.error(f"'{pdf_path}' not found! Place your portfolio PDF in this folder.")
        st.stop()

    loader = PyPDFLoader(pdf_path)
    docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    chunks = text_splitter.split_documents(docs)

    # Use Google's native embedding model (No local compilation required)
    embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-2-preview")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    llm = ChatGoogleGenerativeAI(
        model="gemini-3.5-flash-lite",
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

# Custom Professional Header (Replaces st.title to match the reference image)
st.markdown(f"""
    <div style="background-color: #2563EB; padding: 15px 20px; border-radius: 12px 12px 0 0; display: flex; align-items: center; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1); margin-bottom: 30px;">
        <img src="{AVATAR_URL}" style="width: 45px; height: 45px; border-radius: 50%; margin-right: 15px; background-color: white; object-fit: contain; padding: 2px;">
        <div>
            <h2 style="margin: 0; color: white; font-size: 18px; font-weight: 600; padding-bottom: 2px;">Taran's AI Assistant</h2>
            <div style="display: flex; align-items: center; gap: 6px;">
                <div style="width: 8px; height: 8px; background-color: #22c55e; border-radius: 50%;"></div>
                <p style="margin: 0; font-size: 13px; color: #e2e8f0;">Online Now</p>
            </div>
        </div>
    </div>
""", unsafe_allow_html=True)

# Sidebar controls
with st.sidebar:
    st.header("About")
    st.write("This bot is developed using **LangChain**, **FAISS**, and **Google Gemini's API** By Taran. For more details kindly contact me through my mail given on https://taranveer.in")
    if st.button("Clear Conversation", type="primary"):
        st.session_state.messages = []
        st.rerun()

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! Ask me anything about my projects, background, or skills."}
    ]

# Render previous chat history
for msg in st.session_state.messages:
    # Use custom avatar for assistant, default for user
    avatar = AVATAR_URL if msg["role"] == "assistant" else None
    with st.chat_message(msg["role"], avatar=avatar):
        st.markdown(msg["content"])

# Handle incoming user queries
if prompt := st.chat_input("Ask a question about my work..."):
    
    # Render user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Render assistant message
    with st.chat_message("assistant", avatar=AVATAR_URL):
        with st.spinner("Typing..."):
            response = rag_chain.invoke({"input": prompt})
            answer = response["answer"]
            st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})