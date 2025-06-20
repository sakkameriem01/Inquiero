# Inquiero

An AI-powered application to chat with your PDF documents.

Inquiero is a full-stack application that uses a Retrieval-Augmented Generation (RAG) pipeline to provide intelligent, context-aware answers from your documents.

![Inquiero Conversation Example](./.github/images/Inquiero-Smart-PDF-Chatbot-06-20-2025_11_50_PM.png)

## Core Features

-   **Interactive Q&A**: Ask questions in natural language and get answers from your PDFs.
-   **Multiple Document Support**: Upload and query multiple documents in a single session.
-   **Persistent Chat History**: Save, manage, and revisit your past conversations.
-   **Local First**: Runs with Ollama for privacy and offline capability.
-   **Modern UI**: A clean, responsive interface with dark/light modes and in-app PDF preview.

## Tech Stack

-   **Backend**: FastAPI, Python, LangChain, FAISS, PyMuPDF
-   **Frontend**: React, Tailwind CSS
-   **LLM**: Ollama (Mistral)
-   **Containerization**: Docker

## Getting Started

### Prerequisites

-   Python 3.9+
-   Node.js 16+
-   Ollama installed and running

### Installation & Launch

1.  **Clone the Repository**
   ```bash
    git clone https://github.com/yourusername/inquiero.git
   cd inquiero
   ```

2.  **Set Up & Run Backend**
   ```bash
    cd backend
    python -m venv .venv
    source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   pip install -r requirements.txt
    python run.py
   ```

3.  **Set Up & Run Frontend** (in a new terminal)
   ```bash
    cd frontend
    npm install
    npm start
   ```

4.  **Run Ollama** (in a new terminal, if not already running)
   ```bash
    ollama serve
    ```
    *Ensure you have pulled the Mistral model: `ollama pull mistral`*

5.  **Access the Application**
    Open your browser to `http://localhost:3000`.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details. 