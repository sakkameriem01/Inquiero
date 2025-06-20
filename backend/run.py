import uvicorn
from pathlib import Path
import sys
import logging

# Add the parent directory to Python path
backend_dir = Path(__file__).parent
sys.path.insert(0, str(backend_dir))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    logger.info("Starting Inquiero backend server...")
    uvicorn.run(
        "utils.api:app",  # Updated path to the FastAPI app
        host="127.0.0.1",
        port=8000,
        reload=True,
        reload_dirs=[str(backend_dir)]
    )
    logger.info("Inquiero backend server stopped.") 