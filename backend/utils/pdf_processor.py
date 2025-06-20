"""
PDF Processor Module

This module handles PDF processing operations including:
- PDF text extraction
- Text chunking
- Vector embedding creation
- Answer generation using AI models
"""

import os
from pathlib import Path
import PyPDF2
import fitz
import pytesseract
from PIL import Image
import io
import tempfile
import shutil
from typing import Tuple, List, Dict, Optional
import logging
from datetime import datetime
from langdetect import detect, LangDetectException
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_community.llms import Ollama

# Import the new RAG engine
from .rag_engine import AdvancedRAGEngine

# Configure logging
logger = logging.getLogger(__name__)

class PDFProcessor:
    """
    Handles PDF processing and AI interactions.
    
    Attributes:
        embeddings: HuggingFace embeddings model for creating embeddings
        vectorstore: ChromaDB instance for storing and retrieving embeddings
        rag_engine: Advanced RAG engine for improved question answering
    """

    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.pdfs_dir = self.data_dir / "pdfs"
        self.pdfs_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize text splitter for chunking
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1500,
            chunk_overlap=300,
            length_function=len,
        )
        
        # Initialize embeddings model
        self.embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'}
        )
        
        # Initialize vector store
        self.vector_store = None
        
        # Initialize Ollama LLM with local Mistral model
        self.llm = Ollama(
            model="mistral",
            temperature=0.7,
            num_ctx=4096,  # Context window size
            base_url="http://localhost:11434"  # Default Ollama URL
        )
        
        # Initialize advanced RAG engine
        self.rag_engine = AdvancedRAGEngine(data_dir)
        
        # Configure Tesseract path and check for its existence
        self.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
        self.tesseract_available = False
        if os.name == 'nt':
            if os.path.exists(self.tesseract_cmd):
                pytesseract.pytesseract.tesseract_cmd = self.tesseract_cmd
                self.tesseract_available = True
            else:
                logger.warning(
                    "Tesseract not found at '%s'. OCR for images will be disabled. "
                    "Install Tesseract and/or update the path in pdf_processor.py.",
                    self.tesseract_cmd
                )
        else:
            # On non-Windows systems, check if Tesseract is in the system's PATH
            if shutil.which("tesseract"):
                self.tesseract_available = True
            else:
                logger.warning(
                    "Tesseract not found in PATH. OCR for images will be disabled. "
                    "Please install Tesseract."
                )

    def process_pdf(self, pdf_path: str) -> Tuple[str, Dict[str, any]]:
        """
        Process a PDF file and extract its text content and metadata.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Tuple containing:
            - Extracted text content
            - Dictionary with metadata including:
              - page_count: Number of pages
              - has_images: Boolean indicating if PDF contains images
              - languages: List of detected languages
              - processing_time: Time taken to process
        """
        try:
            pdf_path = Path(pdf_path)
            if not pdf_path.exists():
                raise FileNotFoundError(f"PDF file not found: {pdf_path}")

            # Initialize metadata
            metadata = {
                "page_count": 0,
                "has_images": False,
                "languages": [],
                "processing_time": 0
            }

            # Extract text using PyMuPDF (faster and better for most PDFs)
            text_content = []
            with fitz.open(pdf_path) as doc:
                metadata["page_count"] = len(doc)
                
                for page_num in range(len(doc)):
                    page = doc[page_num]
                    
                    # Extract text
                    text = page.get_text()
                    if text.strip():
                        text_content.append(text)
                    
                    # Check for images
                    if not metadata["has_images"]:
                        image_list = page.get_images()
                        if image_list:
                            metadata["has_images"] = True
                            
                            # Process images with OCR only if Tesseract is available
                            if self.tesseract_available:
                                for img_index, img in enumerate(image_list):
                                    try:
                                        xref = img[0]
                                        base_image = doc.extract_image(xref)
                                        image_bytes = base_image["image"]
                                        
                                        # Convert to PIL Image
                                        image = Image.open(io.BytesIO(image_bytes))
                                        
                                        # Extract text from image using OCR
                                        image_text = pytesseract.image_to_string(image)
                                        if image_text.strip():
                                            text_content.append(f"\n[Image {img_index + 1} on page {page_num + 1}]:\n{image_text}\n")
                                    except Exception as e:
                                        logger.warning(f"Failed to process image on page {page_num + 1}: {str(e)}")
                            else:
                                logger.info("Skipping OCR because Tesseract is not available.")

            # Combine all text content
            full_text = "\n\n".join(text_content)
            
            # If no text was extracted, try PyPDF2 as fallback
            if not full_text.strip():
                with open(pdf_path, 'rb') as file:
                    reader = PyPDF2.PdfReader(file)
                    text_content = []
                    for page in reader.pages:
                        text = page.extract_text()
                        if text:
                            text_content.append(text)
                    full_text = "\n\n".join(text_content)

            # Update RAG engine with new content
            self._update_rag_engine(full_text, pdf_path.name)

            return full_text, metadata

        except Exception as e:
            logger.error(f"Error processing PDF {pdf_path}: {str(e)}")
            raise

    def _update_rag_engine(self, text: str, pdf_name: str) -> None:
        """Update the RAG engine with new text content."""
        try:
            # Add document to RAG engine with metadata
            metadata = {
                "source": pdf_name,
                "timestamp": datetime.now().isoformat(),
                "document_type": "pdf"
            }
            
            self.rag_engine.add_documents([text], [metadata])
            logger.info(f"Updated RAG engine with content from {pdf_name}")
                
        except Exception as e:
            logger.error(f"Error updating RAG engine: {str(e)}")
            raise

    def _update_vector_store(self, text: str) -> None:
        """Update the vector store with new text content (legacy method)."""
        try:
            # Split text into chunks
            chunks = self.text_splitter.split_text(text)
            
            # Create or update vector store
            if self.vector_store is None:
                self.vector_store = FAISS.from_texts(chunks, self.embeddings)
            else:
                self.vector_store.add_texts(chunks)
                
        except Exception as e:
            logger.error(f"Error updating vector store: {str(e)}")
            raise

    def save_pdf(self, file_content: bytes, original_filename: str) -> str:
        """
        Save an uploaded PDF file with a unique name.
        
        Args:
            file_content: The PDF file content as bytes
            original_filename: Original filename of the PDF
            
        Returns:
            Path where the PDF was saved
        """
        try:
            # Create a unique filename
            unique_filename = f"{os.urandom(8).hex()}_{original_filename}"
            pdf_path = self.pdfs_dir / unique_filename
            
            # Save the file
            with open(pdf_path, 'wb') as f:
                f.write(file_content)
            
            return str(pdf_path)
            
        except Exception as e:
            logger.error(f"Error saving PDF {original_filename}: {str(e)}")
            raise

    def delete_pdf(self, pdf_path: str) -> bool:
        """
        Delete a PDF file and its associated files.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            True if deletion was successful, False otherwise
        """
        try:
            pdf_path = Path(pdf_path)
            if pdf_path.exists():
                pdf_path.unlink()
                return True
            return False
        except Exception as e:
            logger.error(f"Error deleting PDF {pdf_path}: {str(e)}")
            return False

    def get_pdf_info(self, pdf_path: str) -> Dict[str, any]:
        """
        Get information about a PDF file.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Dictionary containing PDF information
        """
        try:
            pdf_path = Path(pdf_path)
            if not pdf_path.exists():
                raise FileNotFoundError(f"PDF file not found: {pdf_path}")

            info = {
                "filename": pdf_path.name,
                "size": pdf_path.stat().st_size,
                "created": pdf_path.stat().st_ctime,
                "modified": pdf_path.stat().st_mtime,
                "page_count": 0,
                "has_images": False
            }

            with fitz.open(pdf_path) as doc:
                info["page_count"] = len(doc)
                
                # Check for images in first few pages
                for page_num in range(min(5, len(doc))):
                    if doc[page_num].get_images():
                        info["has_images"] = True
                        break

            return info

        except Exception as e:
            logger.error(f"Error getting PDF info for {pdf_path}: {str(e)}")
            raise

    def extract_images(self, pdf_path: str, output_dir: Optional[str] = None) -> List[str]:
        """
        Extract images from a PDF file.
        
        Args:
            pdf_path: Path to the PDF file
            output_dir: Optional directory to save extracted images
            
        Returns:
            List of paths to extracted images
        """
        try:
            pdf_path = Path(pdf_path)
            if not pdf_path.exists():
                raise FileNotFoundError(f"PDF file not found: {pdf_path}")

            # Create temporary directory if output_dir not specified
            if output_dir is None:
                temp_dir = tempfile.mkdtemp()
                output_dir = temp_dir
            else:
                output_dir = Path(output_dir)
                output_dir.mkdir(parents=True, exist_ok=True)

            image_paths = []
            with fitz.open(pdf_path) as doc:
                for page_num in range(len(doc)):
                    page = doc[page_num]
                    image_list = page.get_images()
                    
                    for img_index, img in enumerate(image_list):
                        try:
                            xref = img[0]
                            base_image = doc.extract_image(xref)
                            image_bytes = base_image["image"]
                            
                            # Save image
                            image_path = Path(output_dir) / f"page_{page_num + 1}_img_{img_index + 1}.png"
                            with open(image_path, 'wb') as img_file:
                                img_file.write(image_bytes)
                            image_paths.append(str(image_path))
                            
                        except Exception as e:
                            logger.warning(f"Failed to extract image on page {page_num + 1}: {str(e)}")

            return image_paths

        except Exception as e:
            logger.error(f"Error extracting images from {pdf_path}: {str(e)}")
            raise
        finally:
            # Clean up temporary directory if we created one
            if output_dir is None and 'temp_dir' in locals():
                shutil.rmtree(temp_dir)

    def load_pdf(self, pdf_path: str) -> str:
        """
        Loads a PDF and returns its text content.
        This is a convenience wrapper around process_pdf.
        """
        full_text, _ = self.process_pdf(pdf_path)
        return full_text

    def detect_language(self, text: str) -> str:
        """
        Detects the language of a given text.
        
        Args:
            text: The text to analyze.
            
        Returns:
            The detected language code (e.g., 'en', 'es'). Defaults to 'en'.
        """
        if not text or not text.strip():
            return "en" # Default language for empty text
        try:
            # Detect language, return 'en' if detection fails
            return detect(text)
        except LangDetectException:
            logger.warning("Language detection failed. Defaulting to 'en'.")
            return "en"

    def get_llm_response(self, query: str, pdf_paths: List[str], chat_history: List[Dict], language: str = "en") -> str:
        """
        Get LLM response, augmented with document context.
        
        Args:
            query (str): The user's question.
            pdf_paths (List[str]): List of PDF paths for context.
            chat_history (List[Dict]): The conversation history.
            language (str): The language for the response.
            
        Returns:
            str: The generated response from the LLM.
        """
        if not pdf_paths:
            logger.warning("No PDF documents are available for context.")
            return "I can't answer questions without a document. Please upload a PDF file."

        # Make sure RAG engine is aware of the document content
        # In a real application, you'd manage this more efficiently
        for pdf_path in pdf_paths:
            if pdf_path not in [doc.metadata.get("source", "") for doc in self.rag_engine.documents]:
                logger.info(f"Loading {pdf_path} into RAG engine...")
                text, _ = self.process_pdf(pdf_path)
                # self._update_rag_engine(text, Path(pdf_path).name) - process_pdf does this

        try:
            logger.info(f"Generating response in language: {language}")
            rag_response = self.rag_engine.generate_response(query, chat_history, language=language)
            
            # Log the full RAG response for debugging
            logger.info(
                f"RAG Response - Confidence: {rag_response.get('confidence', 0.0):.2f}, "
                f"Sources: {rag_response.get('sources', [])}, "
                f"Context chunks: {rag_response.get('context_chunks', 0)}"
            )
            
            return rag_response.get("answer", "No answer could be generated.")
            
        except Exception as e:
            logger.error(f"Error getting LLM response: {e}")
            return "An error occurred while generating the response."

    def get_rag_stats(self) -> Dict[str, any]:
        """
        Get RAG engine statistics and performance metrics.
        
        Returns:
            Dictionary containing RAG system stats
        """
        try:
            return self.rag_engine.get_system_stats()
        except Exception as e:
            logger.error(f"Error getting RAG stats: {str(e)}")
            return {}
    
    def clear_rag_memory(self) -> None:
        """Clear the RAG engine's conversation memory."""
        try:
            self.rag_engine.clear_memory()
            logger.info("Cleared RAG conversation memory")
        except Exception as e:
            logger.error(f"Error clearing RAG memory: {str(e)}") 