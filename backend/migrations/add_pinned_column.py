from sqlalchemy import create_engine, Boolean
from sqlalchemy.sql import text
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get database URL from environment variable or use default SQLite
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./inquiero.db")

def upgrade():
    """Add pinned column to sessions table."""
    engine = create_engine(DATABASE_URL)
    
    with engine.connect() as connection:
        # Add pinned column with default value False
        connection.execute(text("""
            ALTER TABLE sessions 
            ADD COLUMN pinned BOOLEAN DEFAULT FALSE
        """))
        connection.commit()

if __name__ == "__main__":
    print("Adding pinned column to sessions table...")
    upgrade()
    print("Migration completed successfully!") 