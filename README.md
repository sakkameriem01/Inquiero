# Inquiero

An intelligent PDF chat application that allows users to upload PDF documents and ask questions about their content using a **locally hosted Mistral model via Ollama**.

## Features

- PDF document upload and processing
- Intelligent question answering using Mistral (via Ollama)
- Local text embedding for efficient document search
- Modern web interface
- Real-time chat interaction

## Prerequisites

- Python 3.8 or higher
- Node.js 14 or higher (for frontend)
- [Ollama](https://ollama.com) installed with Mistral model downloaded locally

## Installation

1. Clone the repository:
```bash
git clone https://github.com/sakkameriem01/inquiero.git
cd inquiero
```

2. Set up the backend:
```bash
cd backend
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

3. Set up the frontend:
```bash
cd frontend
npm install
```

4. Create a `.env` file in the backend directory:
```
GOOGLE_API_KEY=your_google_api_key_here
DATABASE_URL=sqlite:///./chat_history.db
```

## Running the Application

1. Start the backend server:
```bash
cd backend
python main.py
```

2. Start the frontend development server:
```bash
cd frontend
npm start
```

3. Open your browser and navigate to `http://localhost:3000`

## Usage

1. Easily upload single or multiple PDF documents.
2. Wait for the document to be processed
3. Start asking questions about the document's content
4. View the AI-generated answers

## Project Structure

```
inquiero/
├── backend/
│   ├── utils/
│   │   ├── pdf_loader.py    # PDF processing and text extraction
│   │   └── qa_chain.py      # Question answering system
│   ├── main.py             # FastAPI server
│   ├── requirements.txt    # Python dependencies
│   ├── database.py         # Database connection and session management
│   └── models.py           # Database models
├── frontend/
│   ├── src/
│   │   ├── components/     # React components
│   │   └── App.js          # Main application component
│   └── package.json       # Node.js dependencies
└── README.md
```

## Dependencies

### Backend
- FastAPI
- LangChain
- Ollama + Mistral (local model)
- FAISS
- Sentence Transformers
- PyPDF2

### Frontend
- React
- TypeScript
- Tailwind CSS
- Axios

