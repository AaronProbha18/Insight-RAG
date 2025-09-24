
# 🚀 InsightRAG: Retrieval-Augmented Generation Framework

**InsightRAG** is a modern, modular system for intelligent question answering over your PDF documents using Retrieval-Augmented Generation (RAG). It combines state-of-the-art document processing, embeddings, vector search, and generative AI for accurate, cited answers.

---

## ✨ Features

- **PDF Upload & Processing**: Effortlessly upload and process your own documents
- **Semantic Search**: Find the most relevant content using advanced embeddings
- **AI-Powered Answers**: Get clear, cited answers from your documents
- **Session Logging**: Save and download your Q&A history
- **Web Interface**: Modern, responsive, and easy to use
- **Modular Codebase**: Easily extend or integrate with your own tools

---

## 📂 Repository Structure

```text
InsightRAG/
│
├── notebooks/
│   └── InsightRAG_Document_QA.ipynb      # Main notebook for RAG-based Q&A
│
├── src/
│   ├── pdf_processing.py                 # PDF extraction and chunking logic
│   ├── embeddings.py                     # Embedding generation utilities
│   ├── vector_store.py                   # Vector store and search logic
│   ├── language_model.py                 # Language model loading and response generation
│   └── web_app.py                        # Flask web interface
│
├── requirements.txt                      # Python dependencies
├── README.md                             # Project overview and instructions
├── LICENSE
└── data/
     └── sample_pdfs/                      # Example PDFs for testing
```

---

## 🚀 Quick Start

1. **Clone the repository**
    ```bash
    git clone https://github.com/AaronProbha18/Insight-RAG.git
    ```
2. **Install dependencies**
    ```bash
    pip install -r requirements.txt
    ```
3. **Add your PDFs** to `data/sample_pdfs/` or upload via the web interface.
4. **Open the notebook** in `notebooks/InsightRAG_Document_QA.ipynb` and follow the instructions.
5. **(Optional)** Modularize code into `src/` and run as a web app for production use.

---

## 📦 Requirements

- Python 3.8+
- Jupyter Notebook or VS Code
- See `requirements.txt` for all Python dependencies

---

## 🖥️ Usage

**Notebook:**
- Run all cells in order
- Upload your PDF(s) when prompted
- Ask questions and receive cited answers
- Download session logs as needed

**Web App:**
- Start the Flask app in `src/web_app.py` (coming soon)
- Interact via your browser

---

## ❓ FAQ

**Q: What types of documents are supported?**

A: Currently, only PDF files are supported.

**Q: Can I use my own LLM?**

A: Yes! The code is modular—swap in your preferred model in `src/language_model.py`.

**Q: Is this production-ready?**

A: This project is designed for research, prototyping, and demos. For production, further security and scaling work is recommended.

---

## 📝 License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
