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
# Add this right after st.set_page_config(...) in streamlit_app.py

st.markdown("""
    <style>
    /* Change size of the main title  */
    h1 {
        font-size: 22px !important;
    }
    
    /* Change size of the subtitle/caption */
    .stCaption {
        font-size: 13px !important;
    }
    
    /* Change size of the chat messages text */
    .stChatMessage p {
        font-size: 14px !important;
    }
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

# Header section
st.title("💼 Taran's AI Assistant")
st.caption("Ask questions about my experience, technical skills, and past projects.")

# Sidebar controls
with st.sidebar:
    st.header("About")
    st.write("This bot is developed using **LangChain**, **FAISS**, and **Google Gemini's API** By Taran. For more details kindly contact me through my mail given on the https://taranveer.in")
    if st.button("Clear Conversation"):
        st.session_state.messages = []
        st.rerun()

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! Ask me anything about my projects, background, or skills."}
    ]

# Render previous chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Handle incoming user queries
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
