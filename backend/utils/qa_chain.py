from langchain.chains import RetrievalQA
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from typing import List, Dict
import os
from dotenv import load_dotenv
import logging
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

class QASystem:
    def __init__(self):
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY environment variable is not set")
        
        try:
            self.llm = ChatGoogleGenerativeAI(
                model="gemini-1.5-flash-latest",
                google_api_key=api_key,
                temperature=0,
                convert_system_message_to_human=True,
                max_retries=2,
                timeout=15,
                max_output_tokens=1024,
                top_p=0.8,
                top_k=40
            )
            logger.info("Successfully initialized Gemini model")
        except Exception as e:
            logger.error(f"Error initializing Gemini model: {str(e)}")
            raise ValueError(f"Failed to initialize Gemini model: {str(e)}")
        
        self.qa_chain = None
        self.last_request_time = 0
        self.min_request_interval = 2
        self.processed_files = {}

    def create_qa_chain(self, vector_store):
        """Create a QA chain with the given vector store."""
        if not vector_store:
            raise ValueError("Vector store is required to create QA chain")
        
        try:
            prompt_template = """You are an intelligent assistant that can analyze and compare information from multiple documents. 
            Use the following pieces of context from different documents to answer the question at the end.
            
            If the question requires comparing information across documents, analyze the relationships and differences carefully.
            If you don't know the answer or can't find relevant information, just say that you don't know.
            Keep your answers concise and to the point.
            
            When referencing information, mention which document it comes from.
            
            Context from documents:
            {context}
            
            Question: {question}
            Answer:"""

            PROMPT = PromptTemplate(
                template=prompt_template, input_variables=["context", "question"]
            )

            self.qa_chain = RetrievalQA.from_chain_type(
                llm=self.llm,
                chain_type="stuff",
                retriever=vector_store.as_retriever(
                    search_kwargs={
                        "k": 6,  # Increased to get more context from different documents
                        "fetch_k": 20,  # Fetch more documents initially
                        "maximal_marginal_relevance": True,  # Ensure diverse document selection
                        "filter": None  # No filtering, get from all documents
                    }
                ),
                return_source_documents=True,
                chain_type_kwargs={"prompt": PROMPT}
            )
            logger.info("Successfully created QA chain")
        except Exception as e:
            self.qa_chain = None
            logger.error(f"Error creating QA chain: {str(e)}")
            raise ValueError(f"Error creating QA chain: {str(e)}")

    def get_answer(self, question: str) -> Dict[str, str]:
        """Get answer for a question using combined knowledge from all documents."""
        if not question.strip():
            raise ValueError("Question cannot be empty")
            
        if not self.qa_chain:
            raise ValueError("QA chain not initialized. Please process PDFs first.")
        
        # Rate limiting
        current_time = time.time()
        time_since_last_request = current_time - self.last_request_time
        if time_since_last_request < self.min_request_interval:
            sleep_time = self.min_request_interval - time_since_last_request
            logger.info(f"Rate limiting: waiting {sleep_time:.2f} seconds")
            time.sleep(sleep_time)
        
        try:
            logger.info(f"Processing question: {question}")
            self.last_request_time = time.time()
            result = self.qa_chain.invoke({"query": question})
            
            # Process source documents to include file information
            sources = []
            for doc in result["source_documents"]:
                source_info = {
                    "text": doc.page_content,
                    "source": doc.metadata.get("source", "Unknown document"),
                    "chunk_index": doc.metadata.get("chunk_index", 0)
                }
                sources.append(source_info)
            
            logger.info("Successfully generated answer")
            return {
                "answer": result["result"],
                "sources": sources
            }
        except Exception as e:
            error_msg = str(e)
            if "429" in error_msg:
                logger.error("Rate limit exceeded. Please try again in a few minutes.")
                raise ValueError("Rate limit exceeded. Please try again in a few minutes.")
            logger.error(f"Error getting answer: {error_msg}")
            raise ValueError(f"Error getting answer: {error_msg}")

    def update_processed_files(self, files_info: Dict[str, Dict[str, int]]):
        """Update the list of processed files."""
        self.processed_files.update(files_info)
        logger.info(f"Updated processed files: {len(self.processed_files)} files")

    def get_processed_files(self) -> Dict[str, Dict[str, int]]:
        """Get information about processed files."""
        return self.processed_files 