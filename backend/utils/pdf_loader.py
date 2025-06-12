from typing import List, Dict
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

class PDFProcessor:
    def __init__(self):
        # Initialize the local embeddings model
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",  
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        self.vector_stores = {}  # session_id -> vector_store
        self.processed_files = {}  # session_id -> {filename: {chunks, pages}}

    def get_session_vector_store(self, session_id: str):
        """Get or create a vector store for a specific session."""
        if session_id not in self.vector_stores:
            self.vector_stores[session_id] = None
        return self.vector_stores[session_id]

    def process_pdf(self, file_path: str, filename: str, session_id: str) -> dict:
        """Process a PDF file and return chunks and pages info for a specific session."""
        try:
            logger.info(f"Starting PDF processing for {filename} in session {session_id}")
            
            # Read PDF
            reader = PdfReader(file_path)
            if len(reader.pages) == 0:
                raise ValueError(f"PDF file '{filename}' is empty")
            
            logger.info(f"PDF loaded successfully: {len(reader.pages)} pages found")
            
            # Extract text from each page
            text = ""
            for i, page in enumerate(reader.pages):
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
                    logger.info(f"Extracted text from page {i+1}: {len(page_text)} characters")
                else:
                    logger.warning(f"No text extracted from page {i+1}")

            if not text.strip():
                raise ValueError(f"No text could be extracted from '{filename}'")
            
            logger.info(f"Total text extracted: {len(text)} characters")

            # Split text into chunks
            chunks = self.text_splitter.split_text(text)
            logger.info(f"Text split into {len(chunks)} chunks")
            
            # Initialize session's processed files if not exists
            if session_id not in self.processed_files:
                self.processed_files[session_id] = {}
            
            # Create or update vector store for this session
            if self.vector_stores.get(session_id) is None:
                # Create a new vector store for this session
                self.vector_stores[session_id] = Chroma.from_texts(
                    chunks,
                    self.embeddings,
                    metadatas=[{"source": filename, "chunk_index": i} for i in range(len(chunks))],
                    persist_directory=f"./chroma_db/session_{session_id}"
                )
                logger.info(f"Created new vector store for session {session_id} with {len(chunks)} chunks")
            else:
                # Update the existing vector store for this session
                self.vector_stores[session_id].add_texts(
                    chunks,
                    metadatas=[{"source": filename, "chunk_index": i} for i in range(len(chunks))]
                )
                logger.info(f"Added {len(chunks)} chunks to vector store for session {session_id}")

            # Update processed files info for this session
            self.processed_files[session_id][filename] = {
                "chunks": len(chunks),
                "pages": len(reader.pages)
            }

            # Verify vector store
            vector_store = self.vector_stores[session_id]
            if vector_store:
                # Test retrieval with a simple query
                test_results = vector_store.similarity_search("test", k=1)
                if test_results:
                    logger.info("Vector store verification successful: test query returned results")
                else:
                    logger.warning("Vector store verification: test query returned no results")

            return {
                "chunks": len(chunks),
                "pages": len(reader.pages),
                "text_length": len(text)
            }

        except Exception as e:
            logger.error(f"Error processing PDF {filename} for session {session_id}: {str(e)}")
            raise

    def get_relevant_chunks(self, query: str, session_id: str, k: int = 4) -> List[Dict[str, str]]:
        """Get relevant chunks for a query with metadata from a specific session."""
        vector_store = self.get_session_vector_store(session_id)
        if not vector_store:
            raise ValueError(f"No vector store available for session {session_id}. Please process PDFs first.")
        
        results = vector_store.similarity_search_with_score(query, k=k)
        return [{
            "text": doc.page_content,
            "source": doc.metadata["source"],
            "score": score
        } for doc, score in results]

    def get_processed_files(self, session_id: str) -> dict:
        """Get information about processed files for a specific session."""
        return self.processed_files.get(session_id, {})

    def clear_session(self, session_id: str):
        """Clear the vector store and processed files for a specific session."""
        if session_id in self.vector_stores:
            del self.vector_stores[session_id]
        if session_id in self.processed_files:
            del self.processed_files[session_id]
        logger.info(f"Cleared vector store and processed files for session {session_id}")

    def clear_all(self):
        """Clear all vector stores and processed files."""
        self.vector_stores = {}
        self.processed_files = {}
        logger.info("All vector stores and processed files tracking cleared")

def process_pdf(file_path: str) -> tuple:
    """Process a PDF file and return chunks and pages count."""
    try:
        # Read PDF
        reader = PdfReader(file_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"

        # Split text into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
        chunks = text_splitter.split_text(text)

        return len(chunks), len(reader.pages)

    except Exception as e:
        logger.error(f"Error processing PDF {file_path}: {str(e)}")
        raise 