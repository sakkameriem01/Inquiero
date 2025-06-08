from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
import os
from dotenv import load_dotenv
from utils.pdf_loader import PDFProcessor
from utils.qa_chain import QASystem
import tempfile
import shutil
import logging
from typing import Dict, Any

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize PDF processor and QA system
pdf_processor = PDFProcessor()
qa_system = QASystem()

@app.get("/")
async def root():
    return {"message": "Welcome to Smart PDF Chat API"}

@app.post("/upload/")
async def upload_file(file: UploadFile = File(...)) -> Dict[str, Any]:
    try:
        if not file.filename.endswith('.pdf'):
            raise HTTPException(status_code=400, detail="Only PDF files are allowed")
        
        # Save the uploaded file temporarily
        contents = await file.read()
        with open("temp.pdf", "wb") as f:
            f.write(contents)
        
        # Process the PDF
        num_chunks = pdf_processor.process_pdf("temp.pdf")
        
        # Initialize QA chain with the vector store
        qa_system.create_qa_chain(pdf_processor.vector_store)
        
        return {"message": "File processed successfully", "chunks": num_chunks}
    except Exception as e:
        logger.error(f"Error processing file: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat/")
async def chat(question: Dict[str, str]) -> Dict[str, Any]:
    try:
        response = qa_system.get_answer(question["question"])
        return response
    except Exception as e:
        logger.error(f"Error in chat: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True) 