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
                model="gemini-1.5-flash-latest",  # Using the stable version
                google_api_key=api_key,
                temperature=0,
                convert_system_message_to_human=True,
                max_retries=2,  # Reduced retries to fail faster
                timeout=15,     # Reduced timeout
                max_output_tokens=1024,  # Reduced token limit
                top_p=0.8,
                top_k=40
            )
            logger.info("Successfully initialized Gemini model")
        except Exception as e:
            logger.error(f"Error initializing Gemini model: {str(e)}")
            raise ValueError(f"Failed to initialize Gemini model: {str(e)}")
        
        self.qa_chain = None
        self.last_request_time = 0
        self.min_request_interval = 2  # Minimum seconds between requests

    def create_qa_chain(self, vector_store):
        """Create a QA chain with the given vector store."""
        if not vector_store:
            raise ValueError("Vector store is required to create QA chain")
        
        try:
            prompt_template = """Use the following pieces of context to answer the question at the end. 
            If you don't know the answer, just say that you don't know, don't try to make up an answer.
            Keep your answers concise and to the point.

            Context: {context}

            Question: {question}
            Answer:"""

            PROMPT = PromptTemplate(
                template=prompt_template, input_variables=["context", "question"]
            )

            self.qa_chain = RetrievalQA.from_chain_type(
                llm=self.llm,
                chain_type="stuff",
                retriever=vector_store.as_retriever(search_kwargs={"k": 3}),  # Reduced context chunks
                return_source_documents=True,
                chain_type_kwargs={"prompt": PROMPT}
            )
            logger.info("Successfully created QA chain")
        except Exception as e:
            self.qa_chain = None
            logger.error(f"Error creating QA chain: {str(e)}")
            raise ValueError(f"Error creating QA chain: {str(e)}")

    def get_answer(self, question: str) -> Dict[str, str]:
        """Get answer for a question."""
        if not question.strip():
            raise ValueError("Question cannot be empty")
            
        if not self.qa_chain:
            raise ValueError("QA chain not initialized. Please process a PDF first.")
        
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
            logger.info("Successfully generated answer")
            return {
                "answer": result["result"],
                "sources": [doc.page_content for doc in result["source_documents"]]
            }
        except Exception as e:
            error_msg = str(e)
            if "429" in error_msg:
                logger.error("Rate limit exceeded. Please try again in a few minutes.")
                raise ValueError("Rate limit exceeded. Please try again in a few minutes.")
            logger.error(f"Error getting answer: {error_msg}")
            raise ValueError(f"Error getting answer: {error_msg}") 