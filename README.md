# 💼 Taran's AI Portfolio Assistant

An interactive, Retrieval-Augmented Generation (RAG) chatbot designed to answer visitor and recruiter questions about Taranveer Singh's technical stack, work experience, and projects.

Built with **Streamlit**, **LangChain**, **FAISS**, and **Google Gemini**, optimized for low-latency responses and embedded directly onto [taranveer.in](https://taranveer.in?).

---

## ⚡ Features

* **Context-Aware Retrieval:** Uses a vector database (FAISS) to ground answers strictly in `portfolio.pdf`.
* **Token-Optimized:** Capped chunk size (`500`) and low retriever counts (`k=2`) paired with strict prompt constraints ensure minimal token consumption under free-tier API quotas.
* **Adaptive UI:** Responsive layout themed to `#5166D8` with full light/dark mode support using dynamic CSS variables.
* **Embed Ready:** Designed to live inside an `iframe` widget without unwanted scrollbars, chrome, or mobile header collapsing.
* **Typing Indicator & Modals:** Built-in animated loading state and a collapsible modal dialog for portfolio metadata and chat resets.

---

## 🛠️ Tech Stack

* **Frontend & App Framework:** [Streamlit](https://streamlit.io/)
* **LLM Orchestration:** [LangChain](https://www.langchain.com/)
* **Embeddings:** Google Generative AI (`gemini-embedding-2-preview`)
* **LLM Engine:** Google Gemini Flash (`gemini-3.5-flash-lite`)
* **Vector Index:** [FAISS CPU](https://github.com/facebookresearch/faiss)
* **PDF Processing:** PyPDF

---

## 📁 Repository Structure

```text
├── .env                      # Local environment secrets (ignored by Git)
├── .gitignore                # Git ignore configuration
├── portfolio.pdf             # Portfolio context document parsed by the RAG pipeline
├── requirements.txt          # Python dependencies
├── streamlit_app.py          # Core application logic, RAG pipeline, and custom UI
└── README.md                 # Project documentation

```

---

## 🚀 Getting Started

### Prerequisites

* Python 3.10+
* A Google AI Studio API key ([Get one here](https://aistudio.google.com/))

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/portfolio-bot.git
cd portfolio-bot

```

### 2. Set Up a Virtual Environment

**Windows (PowerShell):**

```powershell
python -m venv venv
.\venv\Scripts\Activate

```

**macOS / Linux:**

```bash
python3 -m venv venv
source venv/bin/activate

```

### 3. Install Dependencies

```bash
pip install -r requirements.txt

```

### 4. Configure Environment Variables

Create a `.env` file in the root directory:

```env
GOOGLE_API_KEY="your_actual_gemini_api_key_here"

```

Ensure you have your resume or project overview placed in the root folder as `portfolio.pdf`.

### 5. Run the Application Locally

```bash
streamlit run streamlit_app.py

```

The app will launch at `http://localhost:8501`.

---

## ☁️ Deployment (Streamlit Cloud)

1. Push this repository to GitHub.
2. Log into [Streamlit Community Cloud](https://share.streamlit.io/).
3. Select **New App**, pick your repository, and set the **Main file path** to `streamlit_app.py`.
4. Go to **Advanced settings → Secrets** and configure your API key:
```toml
GOOGLE_API_KEY = "your_actual_gemini_api_key_here"

```


5. Deploy.

---

## 🌐 Embedding on Your Website

To embed the deployed chatbot into an iframe or floating widget on `taranveer.in`, use:

```html
<iframe
  src="https://your-streamlit-subdomain.streamlit.app/?embed=true"
  width="380"
  height="580"
  style="border: none; border-radius: 16px; box-shadow: 0 10px 30px rgba(0,0,0,0.15);"
></iframe>

```

---

## 📄 License

This project is open-source and available under the [MIT License](https://www.google.com/search?q=LICENSE&).
