"""
Advanced RAG (Retrieval-Augmented Generation) Engine

This module implements sophisticated RAG techniques including:
- Hybrid search (dense + sparse retrieval)
- Context reranking
- Advanced prompt engineering
- Conversation memory management
- Multi-document context fusion
"""

import logging
from typing import List, Dict, Optional, Any
from pathlib import Path
import re
from datetime import datetime

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferWindowMemory
from langchain_community.llms import Ollama
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)

class AdvancedRAGEngine:
    """
    Advanced RAG engine with hybrid retrieval, reranking, and context management.
    """
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize text splitter with optimized parameters
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1500,  # Smaller chunks for better precision
            chunk_overlap=300,  # Reduced overlap
            length_function=len,
            separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""]
        )
        
        # Initialize embeddings model
        self.embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        
        # Initialize vector store
        self.vector_store = None
        self.documents = []
        
        # Initialize LLM
        self.llm = Ollama(
            model="mistral",
            temperature=0.3,  # Lower temperature for more focused responses
            num_ctx=4096,
            base_url="http://localhost:11434"
        )
        
        # Initialize retrievers
        self.dense_retriever = None
        self.sparse_retriever = None
        self.ensemble_retriever = None
        
        # Conversation memory
        self.conversation_memory = ConversationBufferWindowMemory(
            k=10,  # Keep last 10 exchanges
            return_messages=True,
            memory_key="chat_history"
        )
        
        # Reranking model (simplified - using embedding similarity)
        self.rerank_threshold = 0.3
        
    def add_documents(self, texts: List[str], metadata: Optional[List[Dict]] = None) -> None:
        """
        Add documents to the RAG system with proper chunking and indexing.
        
        Args:
            texts: List of text documents
            metadata: Optional metadata for each document
        """
        try:
            # Create documents with metadata
            if metadata is None:
                metadata = [{"source": f"doc_{i}", "timestamp": datetime.now().isoformat()} 
                           for i in range(len(texts))]
            
            documents = []
            for text, meta in zip(texts, metadata):
                # Split text into chunks
                chunks = self.text_splitter.split_text(text)
                
                # Create Document objects with metadata
                for i, chunk in enumerate(chunks):
                    chunk_meta = meta.copy()
                    chunk_meta.update({
                        "chunk_id": i,
                        "chunk_size": len(chunk),
                        "word_count": len(chunk.split())
                    })
                    documents.append(Document(page_content=chunk, metadata=chunk_meta))
            
            # Update vector store
            if self.vector_store is None:
                self.vector_store = FAISS.from_documents(documents, self.embeddings)
            else:
                self.vector_store.add_documents(documents)
            
            # Store documents for sparse retrieval
            self.documents.extend(documents)
            
            # Initialize retrievers
            self._initialize_retrievers()
            
            logger.info(f"Added {len(documents)} document chunks to RAG system")
            
        except Exception as e:
            logger.error(f"Error adding documents: {str(e)}")
            raise
    
    def _initialize_retrievers(self) -> None:
        """Initialize dense and sparse retrievers for hybrid search."""
        try:
            if self.vector_store and self.documents:
                # Dense retriever (vector similarity)
                self.dense_retriever = self.vector_store.as_retriever(
                    search_type="similarity",
                    search_kwargs={"k": 10}
                )
                
                # Sparse retriever (BM25)
                self.sparse_retriever = BM25Retriever.from_documents(self.documents)
                self.sparse_retriever.k = 10
                
                # Ensemble retriever (combines both)
                self.ensemble_retriever = EnsembleRetriever(
                    retrievers=[self.dense_retriever, self.sparse_retriever],
                    weights=[0.8, 0.2]  # Weight dense retrieval more heavily
                )
                
                logger.info("Initialized hybrid retrievers")
                
        except Exception as e:
            logger.error(f"Error initializing retrievers: {str(e)}")
    
    def retrieve_context(self, query: str, k: int = 8) -> List[Document]:
        """
        Retrieve relevant context using hybrid search and reranking.
        
        Args:
            query: User query
            k: Number of documents to retrieve
            
        Returns:
            List of relevant documents
        """
        logger.info(f"Retrieving context for query: '{query}'")
        try:
            if not self.ensemble_retriever:
                logger.warning("Ensemble retriever not initialized. Returning empty context.")
                return []
            
            # Get initial candidates from ensemble retriever
            logger.info("Getting candidates from ensemble retriever...")
            candidates = self.ensemble_retriever.get_relevant_documents(query)
            logger.info(f"Retrieved {len(candidates)} initial candidates.")
            
            # Rerank candidates using query-document similarity
            reranked_candidates = self._rerank_documents(query, candidates)
            logger.info(f"Reranked to {len(reranked_candidates)} candidates.")
            
            # Filter by relevance threshold
            filtered_candidates = [
                doc for doc in reranked_candidates 
                if self._calculate_relevance_score(query, doc.page_content) > self.rerank_threshold
            ]
            logger.info(f"Filtered to {len(filtered_candidates)} candidates after applying threshold of {self.rerank_threshold}.")
            
            # Return top k documents
            return filtered_candidates[:k]
            
        except Exception as e:
            logger.error(f"Error retrieving context: {str(e)}")
            return []
    
    def _rerank_documents(self, query: str, documents: List[Document]) -> List[Document]:
        """
        Rerank documents based on multiple criteria.
        
        Args:
            query: User query
            documents: List of documents to rerank
            
        Returns:
            Reranked list of documents
        """
        try:
            # Calculate relevance scores
            scored_docs = []
            for doc in documents:
                score = self._calculate_relevance_score(query, doc.page_content)
                scored_docs.append((doc, score))
            
            # Sort by score (descending)
            scored_docs.sort(key=lambda x: x[1], reverse=True)
            
            # Return reranked documents
            return [doc for doc, score in scored_docs]
            
        except Exception as e:
            logger.error(f"Error reranking documents: {str(e)}")
            return documents
    
    def _calculate_relevance_score(self, query: str, document: str) -> float:
        """
        Calculate relevance score between query and document.
        
        Args:
            query: User query
            document: Document content
            
        Returns:
            Relevance score (0-1)
        """
        try:
            # Get embeddings
            query_embedding = self.embeddings.embed_query(query)
            doc_embedding = self.embeddings.embed_query(document)
            
            # Calculate cosine similarity
            similarity = cosine_similarity(
                [query_embedding], [doc_embedding]
            )[0][0]
            
            # Additional scoring factors
            length_factor = min(len(document.split()) / 100, 1.0)  # Prefer longer documents
            keyword_overlap = self._calculate_keyword_overlap(query, document)
            
            # Combined score
            final_score = (similarity * 0.6 + length_factor * 0.2 + keyword_overlap * 0.2)
            
            logger.info(f"Relevance score for doc: sim={similarity:.2f}, len={length_factor:.2f}, overlap={keyword_overlap:.2f} -> final={final_score:.2f}")

            return max(0.0, min(1.0, final_score))
            
        except Exception as e:
            logger.error(f"Error calculating relevance score: {str(e)}")
            return 0.0
    
    def _calculate_keyword_overlap(self, query: str, document: str) -> float:
        """
        Calculate keyword overlap between query and document.
        
        Args:
            query: User query
            document: Document content
            
        Returns:
            Keyword overlap score (0-1)
        """
        try:
            # Extract keywords (simple approach)
            query_words = set(re.findall(r'\b\w+\b', query.lower()))
            doc_words = set(re.findall(r'\b\w+\b', document.lower()))
            
            if not query_words:
                return 0.0
            
            # Calculate overlap
            overlap = len(query_words.intersection(doc_words))
            return min(overlap / len(query_words), 1.0)
            
        except Exception as e:
            logger.error(f"Error calculating keyword overlap: {str(e)}")
            return 0.0
    
    def generate_response(self, query: str, chat_history: List[Dict] = None, language: str = "en") -> Dict[str, Any]:
        """
        Generate a response using advanced RAG techniques.
        
        Args:
            query: User query
            chat_history: Previous conversation history
            language: The language for the response
            
        Returns:
            Dictionary containing response and metadata
        """
        try:
            logger.info(f"Generating response for query: '{query[:100]}...' in language: {language}")
            
            # Retrieve relevant context
            context_docs = self.retrieve_context(query)
            
            if not context_docs:
                return {
                    "answer": "I couldn't find relevant information to answer your question. Please try rephrasing or upload a document that might contain the information you're looking for.",
                    "sources": [],
                    "confidence": 0.0
                }
            
            # Format context
            context_text = self._format_context(context_docs)
            
            # Update conversation memory
            if chat_history:
                for msg in chat_history[-5:]:  # Keep last 5 messages
                    if msg.get("sender") == "user":
                        self.conversation_memory.chat_memory.add_user_message(msg.get("text", ""))
                    else:
                        self.conversation_memory.chat_memory.add_ai_message(msg.get("text", ""))
            
            # Generate response using advanced prompt
            response = self._generate_with_context(query, context_text, language)
            
            # Calculate confidence score
            confidence = self._calculate_confidence(query, response, context_docs)
            
            # Extract sources
            sources = [doc.metadata.get("source", "Unknown") for doc in context_docs]
            
            return {
                "answer": response,
                "sources": list(set(sources)),  # Remove duplicates
                "confidence": confidence,
                "context_chunks": len(context_docs)
            }
            
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            return {
                "answer": f"I encountered an error while processing your question: {str(e)}",
                "sources": [],
                "confidence": 0.0
            }
    
    def _format_context(self, documents: List[Document]) -> str:
        """
        Format retrieved documents into context string.
        
        Args:
            documents: List of retrieved documents
            
        Returns:
            Formatted context string
        """
        try:
            context_parts = []
            
            for i, doc in enumerate(documents, 1):
                # Truncate long documents
                content = doc.page_content
                if len(content) > 500:
                    content = content[:500] + "..."
                
                # Add source information
                source = doc.metadata.get("source", "Unknown")
                context_parts.append(f"Source {i} ({source}):\n{content}\n")
            
            return "\n".join(context_parts)
            
        except Exception as e:
            logger.error(f"Error formatting context: {str(e)}")
            return ""
    
    def _generate_with_context(self, query: str, context: str, language: str = "en") -> str:
        """
        Generate response using advanced prompt engineering.
        
        Args:
            query: User query
            context: Retrieved context
            language: The language for the response
            
        Returns:
            Generated response
        """
        try:
            logger.info(f"Generating response with context in language: {language}")
            
            # Advanced prompt template
            prompt_template = PromptTemplate(
                input_variables=["context", "question", "chat_history", "language"],
                template="""
You are an intelligent assistant that provides clean, direct answers based on the provided context.

CONTEXT:
{context}

CHAT HISTORY:
{chat_history}

QUESTION: {question}

LANGUAGE REQUIREMENT: You MUST respond in {language}. This is mandatory.

INSTRUCTIONS:
1. Synthesize an answer from the provided context.
2. Do NOT mention the documents, the context, or sources. Do not use phrases like "According to the document...", "The context states...", etc.
3. If the context doesn't contain enough information, state that you cannot answer the question based on the available information.
4. Your response should be concise, yet comprehensive.
5. CRITICAL: You MUST respond in {language} only. Do not use any other language.

LANGUAGE MAPPING:
- If language is "fr", respond in French
- If language is "es", respond in Spanish  
- If language is "de", respond in German
- If language is "it", respond in Italian
- If language is "pt", respond in Portuguese
- If language is "en", respond in English
- For any other language code, respond in that language

ANSWER (in {language}):
"""
            )
            
            # Get chat history
            chat_history = self.conversation_memory.buffer_as_str
            
            # Generate response
            chain = LLMChain(llm=self.llm, prompt=prompt_template)
            response = chain.run({
                "context": context,
                "question": query,
                "chat_history": chat_history,
                "language": language
            })
            
            return response.strip()
            
        except Exception as e:
            logger.error(f"Error generating with context: {str(e)}")
            return f"I apologize, but I encountered an error while generating a response: {str(e)}"
    
    def _calculate_confidence(self, query: str, response: str, context_docs: List[Document]) -> float:
        """
        Calculate confidence score for the generated response.
        
        Args:
            query: Original query
            response: Generated response
            context_docs: Retrieved context documents
            
        Returns:
            Confidence score (0-1)
        """
        try:
            # Base confidence from context relevance
            context_scores = [
                self._calculate_relevance_score(query, doc.page_content)
                for doc in context_docs
            ]
            avg_context_score = np.mean(context_scores) if context_scores else 0.0
            
            # Response length factor (prefer substantial responses)
            length_factor = min(len(response.split()) / 50, 1.0)
            
            # Response relevance to query
            response_relevance = self._calculate_relevance_score(query, response)
            
            # Combined confidence
            confidence = (
                avg_context_score * 0.5 +
                length_factor * 0.2 +
                response_relevance * 0.3
            )
            
            return max(0.0, min(1.0, confidence))
            
        except Exception as e:
            logger.error(f"Error calculating confidence: {str(e)}")
            return 0.5
    
    def clear_memory(self) -> None:
        """Clear conversation memory."""
        try:
            self.conversation_memory.clear()
            logger.info("Cleared conversation memory")
        except Exception as e:
            logger.error(f"Error clearing memory: {str(e)}")
    
    def get_system_stats(self) -> Dict[str, Any]:
        """
        Get system statistics and performance metrics.
        
        Returns:
            Dictionary containing system stats
        """
        try:
            stats = {
                "total_documents": len(self.documents),
                "vector_store_size": len(self.vector_store.index_to_docstore_id) if self.vector_store else 0,
                "memory_size": len(self.conversation_memory.chat_memory.messages),
                "rerank_threshold": self.rerank_threshold,
                "retrievers_initialized": bool(self.ensemble_retriever)
            }
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting system stats: {str(e)}")
            return {} 