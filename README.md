# Inquiero - AI-Powered PDF Document Q&A System

Inquiero is an intelligent document Q&A system that allows users to upload PDF documents and ask questions about their content. The system uses advanced AI and natural language processing to provide accurate answers based on the document context.

## Features

- 📄 **PDF Document Processing**: Upload and process multiple PDF documents
- 💬 **Interactive Chat Interface**: Ask questions about your documents in natural language
- 🔍 **Context-Aware Responses**: Get answers based on the actual content of your documents
- 📚 **Chat History**: Save and manage your conversations
- 📱 **Modern UI**: Clean and responsive user interface
- 🔒 **Session Management**: Maintain context across conversations
- 📌 **Favorite Chats**: Pin important conversations for quick access

## Tech Stack

### Frontend
- React.js
- Tailwind CSS
- Modern UI components

### Backend
- FastAPI
- LangChain
- ChromaDB (Vector Store)
- HuggingFace Transformers
- SQLAlchemy
- PostgreSQL/SQLite

## Prerequisites

- Python 3.8+
- Node.js 14+
- PostgreSQL (optional, SQLite is used as fallback)

## Installation

### Backend Setup

1. Create and activate a virtual environment:
```bash
# Windows
python -m venv venv
.\venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

2. Install dependencies:
```bash
cd backend
pip install -r requirements.txt
```

3. Set up environment variables:
Create a `.env` file in the backend directory:
```env
DATABASE_URL=postgresql://user:password@localhost:5432/inquiero
# Or use SQLite (default)
# DATABASE_URL=sqlite:///./chat_history.db
```

4. Run database migrations:
```bash
alembic upgrade head
```

5. Start the backend server:
```bash
python main.py
```

### Frontend Setup

1. Install dependencies:
```bash
cd frontend
npm install
```

2. Start the development server:
```bash
npm start
```

## Usage

1. Open your browser and navigate to `http://localhost:3000`
2. Upload one or more PDF documents
3. Wait for the documents to be processed
4. Start asking questions about the content
5. View your chat history and manage conversations

## Project Structure

```
inquiero/
├── backend/
│   ├── main.py              # FastAPI application
│   ├── database.py          # Database configuration
│   ├── models.py            # SQLAlchemy models
│   ├── requirements.txt     # Python dependencies
│   └── utils/
│       ├── pdf_loader.py    # PDF processing utilities
│       └── qa_chain.py      # QA system implementation
├── frontend/
│   ├── src/
│   │   ├── components/      # React components
│   │   ├── App.js          # Main application
│   │   └── index.js        # Entry point
│   └── package.json        # Node.js dependencies
└── README.md
``

## Acknowledgments

- LangChain for the QA framework
- HuggingFace for the transformer models
- FastAPI for the backend framework
- React and Tailwind CSS for the frontend

