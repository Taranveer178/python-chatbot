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

AVATAR_URL = "https://taranveer.in/img/Taranveer_logo_light.webp"

# Custom CSS for a flush, embed-ready chat interface
st.markdown(f"""
    <style>
    /* Hide all default Streamlit top-level elements */
    header[data-testid="stHeader"] {{ display: none !important; }}
    [data-testid="collapsedControl"] {{ display: none !important; }}
    footer {{ display: none !important; }}
    
    /* Remove main padding to fit flush inside your website's iframe */
    .block-container {{
        padding: 15px 15px 100px 15px !important; 
        max-width: 100% !important;
    }}

    /* Base Chat Layout */
    [data-testid="stChatMessage"] {{
        background-color: transparent !important;
        border: none !important;
        padding: 0 !important;
        margin-bottom: 16px !important;
    }}

    /* Bot Message Bubble (Left) */
    [data-testid="stChatMessage"]:has(img) div[data-testid="stMarkdownContainer"] {{
        background-color: #F9FAFB;
        color: #1F2937;
        padding: 12px 16px;
        border-radius: 0px 16px 16px 16px;
        display: inline-block;
        max-width: 85%;
        border: 1px solid #E5E7EB;
        box-shadow: 0 1px 2px rgba(0,0,0,0.02);
    }}

    /* User Message Bubble (Right) */
    [data-testid="stChatMessage"]:has(svg) {{
        flex-direction: row-reverse;
    }}
    [data-testid="stChatMessage"]:has(svg) div[data-testid="stMarkdownContainer"] {{
        background-color: #2563EB;
        color: white;
        padding: 12px 16px;
        border-radius: 16px 16px 0px 16px;
        display: inline-block;
        max-width: 85%;
        box-shadow: 0 1px 2px rgba(0,0,0,0.05);
    }}
    [data-testid="stChatMessage"]:has(svg) div[data-testid="stMarkdownContainer"] p {{
        color: white !important;
    }}

    /* Hide the default user avatar */
    [data-testid="stChatMessage"]:has(svg) div[data-testid="chatAvatarIcon-user"] {{
        display: none;
    }}
    
    /* Chat Text Sizing */
    .stChatMessage p {{
        font-size: 14.5px !important;
        margin-bottom: 0 !important;
        line-height: 1.45;
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

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    chunks = text_splitter.split_documents(docs)

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

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! Ask me anything about my projects, background, or skills."}
    ]

for msg in st.session_state.messages:
    avatar = AVATAR_URL if msg["role"] == "assistant" else None
    with st.chat_message(msg["role"], avatar=avatar):
        st.markdown(msg["content"])

if prompt := st.chat_input("Ask a question about my work..."):
    
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant", avatar=AVATAR_URL):
        with st.spinner("Typing..."):
            response = rag_chain.invoke({"input": prompt})
            answer = response["answer"]
            st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})