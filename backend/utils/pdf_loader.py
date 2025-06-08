from typing import List, Dict
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS, Chroma
import os
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
        self.vector_store = None
        self.processed_files = {}  

    def extract_text_from_pdf(self, file_path: str, file_name: str) -> Dict[str, str]:
        """Extract text from PDF file and return with metadata."""
        try:
            reader = PdfReader(file_path)
            if len(reader.pages) == 0:
                raise ValueError(f"PDF file '{file_name}' is empty")
            
            text = ""
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
            
            if not text.strip():
                raise ValueError(f"No text could be extracted from '{file_name}'")
            
            return {
                "text": text,
                "name": file_name,
                "pages": len(reader.pages)
            }
        except Exception as e:
            raise ValueError(f"Error extracting text from '{file_name}': {str(e)}")

    def chunk_text(self, text: str, file_name: str) -> List[Dict[str, str]]:
        """Split text into chunks with metadata."""
        if not text.strip():
            raise ValueError(f"No text to chunk from '{file_name}'")
        
        chunks = self.text_splitter.split_text(text)
        if not chunks:
            raise ValueError(f"No chunks could be created from '{file_name}'")
        
        # Add metadata to each chunk
        return [{
            "text": chunk,
            "source": file_name,
            "chunk_index": i
        } for i, chunk in enumerate(chunks)]

    def create_vector_store(self, chunks_with_metadata: List[Dict[str, str]]):
        """Create or update FAISS vector store from text chunks with metadata."""
        if not chunks_with_metadata:
            raise ValueError("No chunks provided for vector store creation")
        
        try:
            # Extract just the text for FAISS
            texts = [chunk["text"] for chunk in chunks_with_metadata]
            metadatas = [{"source": chunk["source"], "chunk_index": chunk["chunk_index"]} 
                        for chunk in chunks_with_metadata]
            
            if self.vector_store is None:
                # Create new vector store
                self.vector_store = FAISS.from_texts(
                    texts, 
                    self.embeddings,
                    metadatas=metadatas
                )
            else:
                # Add to existing vector store
                self.vector_store.add_texts(
                    texts,
                    metadatas=metadatas
                )
            
            logger.info(f"Successfully updated vector store with {len(texts)} new chunks")
        except Exception as e:
            raise ValueError(f"Error creating/updating vector store: {str(e)}")

    def process_pdf(self, file_path: str, filename: str) -> dict:
        """Process a PDF file and return chunks and pages info."""
        try:
            # Read PDF
            reader = PdfReader(file_path)
            text = ""
            for page in reader.pages:
                text += page.extract_text() + "\n"

            # Split text into chunks
            chunks = self.text_splitter.split_text(text)
            
            # Create or update vector store
            if self.vector_store is None:
                # Create a new vector store
                self.vector_store = Chroma.from_texts(
                    chunks,
                    self.embeddings,
                    metadatas=[{"source": filename, "chunk_index": i} for i in range(len(chunks))]
                )
            else:
                # Update the existing vector store
                self.vector_store.add_texts(
                    chunks,
                    metadatas=[{"source": filename, "chunk_index": i} for i in range(len(chunks))]
                )

            # Update processed files info
            self.processed_files[filename] = {
                "chunks": len(chunks),
                "pages": len(reader.pages)
            }

            return {
                "chunks": len(chunks),
                "pages": len(reader.pages)
            }

        except Exception as e:
            logger.error(f"Error processing PDF {filename}: {str(e)}")
            raise

    def get_relevant_chunks(self, query: str, k: int = 4) -> List[Dict[str, str]]:
        """Get relevant chunks for a query with metadata."""
        if not self.vector_store:
            raise ValueError("No vector store available. Please process PDFs first.")
        
        results = self.vector_store.similarity_search_with_score(query, k=k)
        return [{
            "text": doc.page_content,
            "source": doc.metadata["source"],
            "score": score
        } for doc, score in results]

    def get_processed_files(self) -> dict:
        """Get information about processed files."""
        return self.processed_files

    def clear_vector_store(self):
        """Clear the vector store and processed files."""
        self.vector_store = None
        self.processed_files = {}
        logger.info("Vector store and processed files tracking cleared")

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