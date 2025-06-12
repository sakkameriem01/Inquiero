from langchain.chains import RetrievalQA
from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate
from typing import Dict, List, Optional
import logging
import time
import json
import re
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ChatMessage:
    def __init__(self, content: str, role: str, timestamp: Optional[datetime] = None):
        self.content = content
        self.role = role  
        self.timestamp = timestamp or datetime.now()

    def to_dict(self) -> Dict:
        return {
            'content': self.content,
            'role': self.role,
            'timestamp': self.timestamp.isoformat()
        }

class QASystem:
    def __init__(self):
        self.qa_chains = {}  # session_id -> qa_chain
        self.processed_files = {}  # session_id -> set of processed files
        self.last_request_time = 0
        self.min_request_interval = 1
        self.chat_history = {}  # session_id -> List[ChatMessage]
        self.max_history_messages = 5  # Number of recent messages to include in context
        self.vector_store = None

    def preprocess_question(self, question: str) -> str:
        """Preprocess the question to make it more robust and forgiving."""
        # Remove extra whitespace
        question = question.strip()
        
        # Fix common typos and patterns
        common_typos = {
            r'\?c': '?',  # Fix "?c" typo
            r'\?s': '?',  # Fix "?s" typo
            r'\?q': '?',  # Fix "?q" typo
            r'\.\.\.': '...',  # Fix multiple dots
            r'!!+': '!',  # Fix multiple exclamation marks
            r'\?\?+': '?',  # Fix multiple question marks
            r'\s+': ' ',  # Fix multiple spaces
            r'wat': 'what',  # Common typos
            r'wut': 'what',
            r'pls': 'please',
            r'thx': 'thanks',
            r'u': 'you',
            r'ur': 'your',
            r'r': 'are',
            r'btw': 'by the way',
            r'afaik': 'as far as I know',
            r'imo': 'in my opinion',
            r'fyi': 'for your information',
            r'idk': 'I don\'t know',
            r'ty': 'thank you',
            r'np': 'no problem',
            r'omg': 'oh my god',
            r'brb': 'be right back',
            r'asap': 'as soon as possible',
            r'atm': 'at the moment',
            r'afaik': 'as far as I know',
            r'imo': 'in my opinion',
            r'fyi': 'for your information',
            r'idk': 'I don\'t know',
            r'ty': 'thank you',
            r'np': 'no problem',
            r'omg': 'oh my god',
            r'brb': 'be right back',
            r'asap': 'as soon as possible',
            r'atm': 'at the moment',
        }
        
        # Apply common typos fixes
        for pattern, replacement in common_typos.items():
            question = re.sub(r'\b' + pattern + r'\b', replacement, question, flags=re.IGNORECASE)
        
        # Fix punctuation
        question = re.sub(r'([.!?])\1+', r'\1', question)  # Remove duplicate punctuation
        if not question[-1] in '.!?':
            question += '?'
        
        # Remove any remaining special characters that might interfere
        question = re.sub(r'[^\w\s\?\.\!\-\'\"]', '', question)
        
        # Capitalize first letter
        question = question[0].upper() + question[1:]
        
        # Fix common sentence structure issues
        question = re.sub(r'\s+([.,!?])', r'\1', question)  # Remove spaces before punctuation
        question = re.sub(r'([.,!?])(?=[^\s])', r'\1 ', question)  # Add space after punctuation if missing
        
        # Fix common word spacing issues
        question = re.sub(r'\s+', ' ', question)  # Normalize spaces
        
        return question

    def create_qa_chain(self, vector_store, session_id: str):
        """Create a QA chain for a specific session."""
        if not vector_store:
            raise ValueError("Vector store is required to create QA chain")

        # Create a custom prompt template
        template = """Use the following pieces of context to answer the question at the end. 
        If you don't know the answer, just say that you don't know, don't try to make up an answer.
        Use three sentences maximum and keep the answer concise.

        Context: {context}

        Question: {question}

        Answer:"""

        QA_CHAIN_PROMPT = PromptTemplate(
            template=template,
            input_variables=["context", "question"]
        )

        # Initialize Ollama with Mistral model
        llm = Ollama(
            model="mistral",
            temperature=0.1,
            num_ctx=4096,  # Context window size
            num_thread=4,  # Number of CPU threads to use
            repeat_penalty=1.1,  # Penalty for repeating tokens
            top_k=40,  # Number of tokens to consider for each prediction
            top_p=0.9  # Nucleus sampling parameter
        )

        # Create a new QA chain for this session
        self.qa_chains[session_id] = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vector_store.as_retriever(search_kwargs={"k": 4}),
            return_source_documents=True,
            chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}
        )
        logger.info(f"Created new QA chain for session {session_id} using Mistral model")

    def get_answer(self, question: str, session_id: str) -> dict:
        """Get answer for a question using session-specific QA chain."""
        if not question.strip():
            raise ValueError("Question cannot be empty")

        if session_id not in self.qa_chains:
            raise ValueError(f"No QA chain available for session {session_id}. Please process PDFs first.")

        try:
            # Get answer from session-specific QA chain
            result = self.qa_chains[session_id]({"query": question})
            
            # Extract answer and source documents
            answer = result.get("result", "")
            source_docs = result.get("source_documents", [])

            # Process source documents
            sources = []
            for doc in source_docs:
                if hasattr(doc, "metadata"):
                    sources.append({
                        "text": doc.page_content,
                        "source": doc.metadata.get("source", "Unknown"),
                        "chunk_index": doc.metadata.get("chunk_index", -1)
                    })

            return {
                "answer": answer,
                "sources": sources
            }

        except Exception as e:
            logger.error(f"Error getting answer for session {session_id}: {str(e)}")
            raise ValueError(f"Failed to get answer: {str(e)}")

    def is_initialized(self, session_id: str) -> bool:
        """Check if QA chain is initialized for a session."""
        return session_id in self.qa_chains

    def clear_session(self, session_id: str):
        """Clear QA chain for a specific session."""
        if session_id in self.qa_chains:
            del self.qa_chains[session_id]
        if session_id in self.processed_files:
            del self.processed_files[session_id]
        logger.info(f"Cleared QA chain for session {session_id}")

    def clear_all(self):
        """Clear all QA chains."""
        self.qa_chains = {}
        self.processed_files = {}
        logger.info("All QA chains cleared")

    def get_chat_history_context(self, session_id: str) -> str:
        """Get formatted chat history context for the prompt."""
        if session_id not in self.chat_history:
            return ""
        
        messages = self.chat_history[session_id][-self.max_history_messages:]
        context = "\nRecent conversation:\n"
        
        for msg in messages:
            role = "User" if msg.role == "user" else "Assistant"
            context += f"{role}: {msg.content}\n"
        
        return context

    def add_to_chat_history(self, session_id: str, message: ChatMessage):
        """Add a message to the chat history."""
        if session_id not in self.chat_history:
            self.chat_history[session_id] = []
        self.chat_history[session_id].append(message)

    def update_processed_files(self, files_info: Dict[str, Dict[str, int]]):
        """Update the list of processed files."""
        self.processed_files.update(files_info)
        logger.info(f"Updated processed files: {len(self.processed_files)} files")

    def get_processed_files(self) -> Dict[str, Dict[str, int]]:
        """Get information about processed files."""
        return self.processed_files 