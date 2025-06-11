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
        self.role = role  # 'user' or 'assistant'
        self.timestamp = timestamp or datetime.now()

    def to_dict(self) -> Dict:
        return {
            'content': self.content,
            'role': self.role,
            'timestamp': self.timestamp.isoformat()
        }

class QASystem:
    def __init__(self):
        try:
            self.llm = Ollama(
                model="mistral",
                temperature=0.1,  
                timeout=30,
                num_ctx=4096,
                num_thread=4
            )
            logger.info("Successfully initialized Mistral model through Ollama")
        except Exception as e:
            logger.error(f"Error initializing Mistral model: {str(e)}")
            raise ValueError(f"Failed to initialize Mistral model: {str(e)}")
        
        self.qa_chain = None
        self.last_request_time = 0
        self.min_request_interval = 1
        self.processed_files = {}
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

    def create_qa_chain(self, vector_store):
        """Create a QA chain with the given vector store."""
        if not vector_store:
            raise ValueError("Vector store is required to create QA chain")
        
        try:
            self.vector_store = vector_store
            from langchain.chains import ConversationalRetrievalChain
            from langchain.memory import ConversationBufferMemory

            # Create memory for chat history
            memory = ConversationBufferMemory(
                memory_key="chat_history",
                return_messages=True,
                output_key="answer"  # Explicitly set which key to store in memory
            )

            # Create the chain
            self.qa_chain = ConversationalRetrievalChain.from_llm(
                llm=self.llm,
                retriever=vector_store.as_retriever(
                    search_kwargs={"k": 6}
                ),
                memory=memory,
                return_source_documents=True,
                verbose=True
            )
            
            logger.info("Successfully created QA chain")
        except Exception as e:
            self.qa_chain = None
            logger.error(f"Error creating QA chain: {str(e)}")
            raise ValueError(f"Error creating QA chain: {str(e)}")

    def is_initialized(self):
        """Check if the QA system is properly initialized."""
        return self.qa_chain is not None and self.vector_store is not None

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

    def clear_chat_history(self, session_id: str):
        """Clear chat history for a specific session."""
        if session_id in self.chat_history:
            del self.chat_history[session_id]

    def get_answer(self, question: str, session_id: Optional[str] = None) -> Dict[str, str]:
        """Get answer for a question using combined knowledge from all documents."""
        if not question.strip():
            raise ValueError("Question cannot be empty")
            
        if not self.is_initialized():
            raise ValueError("QA chain not initialized. Please process PDFs first.")
        
        # Preprocess the question
        processed_question = self.preprocess_question(question)
        logger.info(f"Original question: {question}")
        logger.info(f"Processed question: {processed_question}")
        
        # Add user message to chat history
        if session_id:
            self.add_to_chat_history(session_id, ChatMessage(processed_question, "user"))
        
        # Rate limiting
        current_time = time.time()
        time_since_last_request = current_time - self.last_request_time
        if time_since_last_request < self.min_request_interval:
            sleep_time = self.min_request_interval - time_since_last_request
            logger.info(f"Rate limiting: waiting {sleep_time:.2f} seconds")
            time.sleep(sleep_time)
        
        try:
            logger.info(f"Processing question: {processed_question}")
            self.last_request_time = time.time()
            
            # Get chat history context
            chat_history = self.get_chat_history_context(session_id) if session_id else ""
            
            # Invoke the QA chain with just the question
            result = self.qa_chain({"question": processed_question})
            
            # Extract and process the answer
            answer = result.get("answer", "")
            if not answer:
                answer = "I couldn't find a specific answer to your question in the provided documents. Could you please rephrase your question or provide more context?"
            
            # Ensure answer is a string
            if isinstance(answer, dict):
                answer = json.dumps(answer, ensure_ascii=False, indent=2)
            elif not isinstance(answer, str):
                answer = str(answer)
            
            answer = answer.strip()
            
            # Add assistant's response to chat history
            if session_id:
                self.add_to_chat_history(session_id, ChatMessage(answer, "assistant"))
            
            # Process source documents to include file information
            sources = []
            for doc in result.get("source_documents", []):
                source_info = {
                    "text": doc.page_content,
                    "source": doc.metadata.get("source", "Unknown document"),
                    "chunk_index": doc.metadata.get("chunk_index", 0)
                }
                sources.append(source_info)
            
            logger.info("Successfully generated answer")
            return {
                "answer": answer,
                "sources": sources
            }
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Error getting answer: {error_msg}")
            raise ValueError(f"Error getting answer: {error_msg}")

    def update_processed_files(self, files_info: Dict[str, Dict[str, int]]):
        """Update the list of processed files."""
        self.processed_files.update(files_info)
        logger.info(f"Updated processed files: {len(self.processed_files)} files")

    def get_processed_files(self) -> Dict[str, Dict[str, int]]:
        """Get information about processed files."""
        return self.processed_files 