from typing import List
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import os
from dotenv import load_dotenv

load_dotenv()

class PDFProcessor:
    def __init__(self):
        # Initialize the local embeddings model
        self.embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2",  # A good balance of speed and quality
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        self.vector_store = None

    def extract_text_from_pdf(self, file_path: str) -> str:
        """Extract text from PDF file."""
        try:
            reader = PdfReader(file_path)
            if len(reader.pages) == 0:
                raise ValueError("PDF file is empty")
            
            text = ""
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
            
            if not text.strip():
                raise ValueError("No text could be extracted from the PDF")
            
            return text
        except Exception as e:
            raise ValueError(f"Error extracting text from PDF: {str(e)}")

    def chunk_text(self, text: str) -> List[str]:
        """Split text into chunks."""
        if not text.strip():
            raise ValueError("No text to chunk")
        
        chunks = self.text_splitter.split_text(text)
        if not chunks:
            raise ValueError("No chunks could be created from the text")
        
        return chunks

    def create_vector_store(self, chunks: List[str]):
        """Create FAISS vector store from text chunks."""
        if not chunks:
            raise ValueError("No chunks provided for vector store creation")
        
        try:
            self.vector_store = FAISS.from_texts(chunks, self.embeddings)
        except Exception as e:
            raise ValueError(f"Error creating vector store: {str(e)}")

    def process_pdf(self, file_path: str) -> int:
        """Process PDF file and create vector store."""
        try:
            # Extract text
            text = self.extract_text_from_pdf(file_path)
            
            # Chunk text
            chunks = self.chunk_text(text)
            
            # Create vector store
            self.create_vector_store(chunks)
            
            return len(chunks)
        except Exception as e:
            self.vector_store = None  # Reset vector store on error
            raise ValueError(f"Error processing PDF: {str(e)}")

    def get_relevant_chunks(self, query: str, k: int = 4) -> List[str]:
        """Get relevant chunks for a query."""
        if not self.vector_store:
            raise ValueError("No vector store available. Please process a PDF first.")
        return self.vector_store.similarity_search(query, k=k) 