import os
from dotenv import load_dotenv

# Document loading and splitting
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# FastEmbed for lightweight local CPU embeddings
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings

# Gemini LLM for answering questions
from langchain_google_genai import ChatGoogleGenerativeAI

# Vector storage and LangChain Classic chains
from langchain_community.vectorstores import FAISS
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# 1. Load API Key
load_dotenv()
if not os.getenv("GOOGLE_API_KEY"):
    raise ValueError("GOOGLE_API_KEY is not set. Check your .env file.")

def create_rag_pipeline(pdf_path: str = "portfolio.pdf"):
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"'{pdf_path}' not found. Place your PDF in the project folder.")

    # 2. Extract Document Content
    print("[1/4] Loading document...")
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()

    # 3. Chunk the Document
    print("[2/4] Splitting text into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100
    )
    chunks = text_splitter.split_documents(docs)

    # 4. Local Embeddings with FastEmbed (Runs on CPU, bypasses Google 404)
    print("[3/4] Creating local embeddings via FastEmbed...")
    embeddings = FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    # 5. Gemini Chat Model for generating responses
    print("[4/4] Configuring Gemini LLM...")
    llm = ChatGoogleGenerativeAI(
        model="gemini-3.6-flash",
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

    # 6. Assemble Retrieval Chain
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)
    
    return rag_chain

if __name__ == "__main__":
    print("Starting Portfolio Bot...")
    chain = create_rag_pipeline("portfolio.pdf")
    print("\nPortfolio Bot is ready! (Type 'exit' to stop)\n" + "-" * 50)
    
    while True:
        query = input("\nAsk something: ").strip()
        if query.lower() in ("exit", "quit"):
            print("Goodbye!")
            break
        if not query:
            continue
            
        try:
            result = chain.invoke({"input": query})
            print(f"\nResponse:\n{result['answer']}")
        except Exception as e:
            print(f"\nError: {str(e)}")


            
