"""
PDF processing and AI interaction utilities for Inquiero backend.
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
from langchain_community.llms import Ollama

from .rag_engine import AdvancedRAGEngine

logger = logging.getLogger(__name__)

class PDFProcessor:
    """
    Handles PDF processing, text extraction, OCR, embedding, and AI-powered Q&A.
    """

    def __init__(self, data_dir: str = "data"):
        """
        Initialize the PDFProcessor with directories, models, and OCR settings.
        """
        self.data_dir = Path(data_dir)
        self.pdfs_dir = self.data_dir / "pdfs"
        self.pdfs_dir.mkdir(parents=True, exist_ok=True)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1500,
            chunk_overlap=300,
            length_function=len,
        )
        self.embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'}
        )
        self.vector_store = None
        self.llm = Ollama(
            model="mistral",
            temperature=0.7,
            num_ctx=4096,
            base_url="http://localhost:11434"
        )
        self.rag_engine = AdvancedRAGEngine(data_dir)
        self.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
        self.tesseract_available = False
        if os.name == 'nt':
            if os.path.exists(self.tesseract_cmd):
                pytesseract.pytesseract.tesseract_cmd = self.tesseract_cmd
                self.tesseract_available = True
            else:
                logger.warning("Tesseract not found at '%s'. OCR for images will be disabled.", self.tesseract_cmd)
        else:
            if shutil.which("tesseract"):
                self.tesseract_available = True
            else:
                logger.warning("Tesseract not found in PATH. OCR for images will be disabled.")

    def process_pdf(self, pdf_path: str) -> Tuple[str, Dict[str, any]]:
        """
        Extract text and metadata from a PDF file. Uses OCR for images if available.
        Args:
            pdf_path: Path to the PDF file.
        Returns:
            Tuple of (extracted text, metadata dict).
        """
        try:
            pdf_path = Path(pdf_path)
            if not pdf_path.exists():
                raise FileNotFoundError(f"PDF file not found: {pdf_path}")
            metadata = {
                "page_count": 0,
                "has_images": False,
                "languages": [],
                "processing_time": 0
            }
            text_content = []
            with fitz.open(pdf_path) as doc:
                metadata["page_count"] = len(doc)
                for page_num in range(len(doc)):
                    page = doc[page_num]
                    text = page.get_text()
                    if text.strip():
                        text_content.append(text)
                    if not metadata["has_images"]:
                        image_list = page.get_images()
                        if image_list:
                            metadata["has_images"] = True
                            if self.tesseract_available:
                                for img_index, img in enumerate(image_list):
                                    try:
                                        xref = img[0]
                                        base_image = doc.extract_image(xref)
                                        image_bytes = base_image["image"]
                                        image = Image.open(io.BytesIO(image_bytes))
                                        image_text = pytesseract.image_to_string(image)
                                        if image_text.strip():
                                            text_content.append(f"\n[Image {img_index + 1} on page {page_num + 1}]:\n{image_text}\n")
                                    except Exception as e:
                                        logger.warning(f"Failed to process image on page {page_num + 1}: {str(e)}")
                            else:
                                logger.info("Skipping OCR because Tesseract is not available.")
            full_text = "\n\n".join(text_content)
            # Fallback to PyPDF2 if no text was extracted
            if not full_text.strip():
                with open(pdf_path, 'rb') as file:
                    reader = PyPDF2.PdfReader(file)
                    text_content = []
                    for page in reader.pages:
                        text = page.extract_text()
                        if text:
                            text_content.append(text)
                    full_text = "\n\n".join(text_content)
            self._update_rag_engine(full_text, pdf_path.name)
            return full_text, metadata
        except Exception as e:
            logger.error(f"Error processing PDF {pdf_path}: {str(e)}")
            raise

    def _update_rag_engine(self, text: str, pdf_name: str) -> None:
        """
        Update the RAG engine with new text content from a PDF.
        """
        try:
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

    def save_pdf(self, file_content: bytes, original_filename: str) -> str:
        """
        Save an uploaded PDF file with a unique name.
        Args:
            file_content: PDF file content as bytes.
            original_filename: Original filename of the PDF.
        Returns:
            Path where the PDF was saved (as string).
        """
        try:
            unique_filename = f"{os.urandom(8).hex()}_{original_filename}"
            pdf_path = self.pdfs_dir / unique_filename
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
            pdf_path: Path to the PDF file.
        Returns:
            True if deletion was successful, False otherwise.
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
        Get basic information about a PDF file.
        Args:
            pdf_path: Path to the PDF file.
        Returns:
            Dictionary with PDF info (filename, size, created, modified, page_count, has_images).
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
        Extract images from a PDF file and save them to disk.
        Args:
            pdf_path: Path to the PDF file.
            output_dir: Optional directory to save extracted images.
        Returns:
            List of paths to extracted images.
        """
        try:
            pdf_path = Path(pdf_path)
            if not pdf_path.exists():
                raise FileNotFoundError(f"PDF file not found: {pdf_path}")
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
            if output_dir is None and 'temp_dir' in locals():
                shutil.rmtree(temp_dir)

    def load_pdf(self, pdf_path: str) -> str:
        """
        Load a PDF and return its text content (wrapper for process_pdf).
        Args:
            pdf_path: Path to the PDF file.
        Returns:
            Extracted text content.
        """
        full_text, _ = self.process_pdf(pdf_path)
        return full_text

    def detect_language(self, text: str) -> str:
        """
        Detect the language of a given text. Defaults to 'en' if detection fails.
        Args:
            text: The text to analyze.
        Returns:
            The detected language code (e.g., 'en', 'es').
        """
        if not text or not text.strip():
            return "en"
        try:
            return detect(text)
        except LangDetectException:
            logger.warning("Language detection failed. Defaulting to 'en'.")
            return "en"

    def get_llm_response(self, query: str, pdf_paths: List[str], chat_history: List[Dict], language: str = "en") -> str:
        """
        Generate an LLM response, augmented with document context.
        Args:
            query: The user's question.
            pdf_paths: List of PDF paths for context.
            chat_history: The conversation history.
            language: The language for the response.
        Returns:
            The generated response from the LLM.
        """
        if not pdf_paths:
            logger.warning("No PDF documents are available for context.")
            return "I can't answer questions without a document. Please upload a PDF file."
        for pdf_path in pdf_paths:
            if pdf_path not in [doc.metadata.get("source", "") for doc in self.rag_engine.documents]:
                logger.info(f"Loading {pdf_path} into RAG engine...")
                text, _ = self.process_pdf(pdf_path)
        try:
            logger.info(f"Generating response in language: {language}")
            rag_response = self.rag_engine.generate_response(query, chat_history, language=language)
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
            Dictionary containing RAG system stats.
        """
        try:
            return self.rag_engine.get_system_stats()
        except Exception as e:
            logger.error(f"Error getting RAG stats: {str(e)}")
            return {}
    
    def clear_rag_memory(self) -> None:
        """
        Clear the RAG engine's conversation memory.
        """
        try:
            self.rag_engine.clear_memory()
            logger.info("Cleared RAG conversation memory")
        except Exception as e:
            logger.error(f"Error clearing RAG memory: {str(e)}") 