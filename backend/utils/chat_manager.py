"""
Chat Manager Module

This module handles the management of chat sessions, including:
- Creating and managing chat sessions
- Storing and retrieving chat history
- Managing PDF files associated with chats
- Generating meaningful chat titles
"""

import json
from datetime import datetime
import uuid
from faker import Faker
from pathlib import Path
from typing import Dict, List, Optional
import logging

from config.settings import CHAT_HISTORY_DIR

# Configure logging
logger = logging.getLogger(__name__)

class ChatManager:
    """
    Manages chat operations including creating, saving, and retrieving chats.
    Each chat is stored as a separate JSON file in the chat history directory.
    """

    def __init__(self, chat_history_dir: str = "chat_history"):
        self.chat_history_dir = Path(chat_history_dir)
        self.chat_history_dir.mkdir(parents=True, exist_ok=True)
        self.faker = Faker()
        # Clean up any old format files
        self._cleanup_old_format_files()

    def _cleanup_old_format_files(self) -> None:
        """
        Clean up any old format chat files (like chats.json) and ensure all chats
        are in the correct format.
        """
        try:
            # Remove chats.json if it exists
            chats_json = self.chat_history_dir / "chats.json"
            if chats_json.exists():
                logger.info("Removing old format chats.json file")
                chats_json.unlink()
            
            # Ensure all chat files are in the correct format
            for chat_file in self.chat_history_dir.glob("*.json"):
                if chat_file.name == "chats.json":
                    continue
                    
                try:
                    with open(chat_file, "r", encoding="utf-8") as f:
                        chat_data = json.load(f)
                        
                    # If the file contains a list, convert it to proper format
                    if isinstance(chat_data, list):
                        logger.info(f"Converting list format chat file: {chat_file}")
                        for chat in chat_data:
                            if isinstance(chat, dict) and "id" in chat:
                                # Create a new file for each chat in the list
                                new_chat_file = self.chat_history_dir / f"{chat['id']}.json"
                                if not new_chat_file.exists():
                                    self._save_chat_data(new_chat_file, chat)
                        # Remove the old file
                        chat_file.unlink()
                except Exception as e:
                    logger.error(f"Error processing chat file {chat_file}: {str(e)}")
        except Exception as e:
            logger.error(f"Error during cleanup: {str(e)}")

    def _save_chat_data(self, chat_file: Path, chat_data: Dict) -> None:
        """
        Save chat data to a file with proper validation.
        
        Args:
            chat_file: Path to the chat file
            chat_data: Dictionary containing chat data
        """
        try:
            # Validate and ensure proper structure
            chat_data = self._validate_chat_data(chat_data)
            
            # Ensure chat_id matches filename
            chat_id = chat_file.stem
            chat_data["chat_id"] = chat_id
            
            # Save the file
            with open(chat_file, "w", encoding="utf-8") as f:
                json.dump(chat_data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"Error saving chat data to {chat_file}: {str(e)}")
            raise

    def _validate_chat_data(self, chat_data: Dict) -> Dict:
        """
        Validate and ensure chat data has required fields.
        
        Args:
            chat_data: Dictionary containing chat information
            
        Returns:
            Dictionary with validated chat data
        """
        # Ensure chat data is a dictionary
        if not isinstance(chat_data, dict):
            logger.warning(f"Invalid chat data type: {type(chat_data)}, creating new chat structure")
            chat_data = {
                "messages": [],
                "created_at": datetime.now().isoformat(),
                "name": "New Chat",
                "pdf_paths": [],
                "pdf_names": []
            }
        
        # Ensure required fields exist with proper types
        if "messages" not in chat_data or not isinstance(chat_data["messages"], list):
            chat_data["messages"] = []
            
        if "created_at" not in chat_data or not isinstance(chat_data["created_at"], str):
            chat_data["created_at"] = datetime.now().isoformat()
            
        if "name" not in chat_data or not isinstance(chat_data["name"], str):
            chat_data["name"] = "New Chat"
            
        if "pdf_paths" not in chat_data or not isinstance(chat_data["pdf_paths"], list):
            chat_data["pdf_paths"] = []
            
        # Ensure pdf_names exists and matches pdf_paths length
        if "pdf_names" not in chat_data or not isinstance(chat_data["pdf_names"], list):
            chat_data["pdf_names"] = []
            
        # If pdf_names is shorter than pdf_paths, generate names from paths
        if len(chat_data["pdf_names"]) < len(chat_data["pdf_paths"]):
            for i in range(len(chat_data["pdf_names"]), len(chat_data["pdf_paths"])):
                pdf_path = chat_data["pdf_paths"][i]
                pdf_name = Path(pdf_path).name if pdf_path else f"document_{i+1}.pdf"
                chat_data["pdf_names"].append(pdf_name)
            
        return chat_data

    def create_chat(self, chat_id: str, chat_data: Dict) -> None:
        """
        Create a new chat with the given ID and data.
        
        Args:
            chat_id: Unique identifier for the chat
            chat_data: Dictionary containing chat information
        """
        try:
            # Ensure chat_data is a dictionary
            if not isinstance(chat_data, dict):
                logger.warning(f"Invalid chat_data type: {type(chat_data)}, creating new chat structure")
                chat_data = {"messages": []}
            
            # Validate and ensure proper structure
            chat_data = self._validate_chat_data(chat_data)
            
            # Add creation timestamp if not present
            if "created_at" not in chat_data:
                chat_data["created_at"] = datetime.now().isoformat()
            
            # Add chat_id to the data
            chat_data["chat_id"] = chat_id
            
            chat_file = self.chat_history_dir / f"{chat_id}.json"
            with open(chat_file, "w", encoding="utf-8") as f:
                json.dump(chat_data, f, ensure_ascii=False, indent=2)
            logger.info(f"Created new chat: {chat_id}")
        except Exception as e:
            logger.error(f"Failed to create chat {chat_id}: {str(e)}")
            raise

    def get_chat(self, chat_id: str) -> Optional[Dict]:
        """
        Retrieve a chat by its ID.
        
        Args:
            chat_id: Unique identifier for the chat
            
        Returns:
            Dictionary containing chat data, or None if not found
        """
        try:
            chat_file = self.chat_history_dir / f"{chat_id}.json"
            if not chat_file.exists():
                return None
            with open(chat_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to get chat {chat_id}: {str(e)}")
            return None

    def save_chat(self, chat_id: str, chat_data: Dict) -> None:
        """
        Save or update an existing chat.
        
        Args:
            chat_id: Unique identifier for the chat
            chat_data: Dictionary containing updated chat information
        """
        try:
            # Update timestamp
            chat_data["updated_at"] = datetime.now().isoformat()
            
            chat_file = self.chat_history_dir / f"{chat_id}.json"
            with open(chat_file, "w", encoding="utf-8") as f:
                json.dump(chat_data, f, ensure_ascii=False, indent=2)
            logger.info(f"Saved chat: {chat_id}")
        except Exception as e:
            logger.error(f"Failed to save chat {chat_id}: {str(e)}")
            raise

    def delete_chat(self, chat_id: str) -> bool:
        """
        Delete a chat by its ID.
        
        Args:
            chat_id: Unique identifier for the chat
            
        Returns:
            True if deletion was successful, False otherwise
        """
        try:
            chat_file = self.chat_history_dir / f"{chat_id}.json"
            if chat_file.exists():
                chat_file.unlink()
                logger.info(f"Deleted chat: {chat_id}")
                return True
            return False
        except Exception as e:
            logger.error(f"Failed to delete chat {chat_id}: {str(e)}")
            return False

    def delete_all_chats(self) -> bool:
        """
        Delete all chats from the chat history directory.
        
        Returns:
            True if deletion was successful, False otherwise
        """
        try:
            deleted_count = 0
            for chat_file in self.chat_history_dir.glob("*.json"):
                try:
                    chat_file.unlink()
                    deleted_count += 1
                except Exception as e:
                    logger.error(f"Failed to delete chat file {chat_file}: {str(e)}")
            
            logger.info(f"Deleted {deleted_count} chat files")
            return True
        except Exception as e:
            logger.error(f"Failed to delete all chats: {str(e)}")
            return False

    def get_all_chats(self) -> List[Dict]:
        """
        Retrieve all chats.
        
        Returns:
            List of dictionaries containing chat data, sorted by creation date
        """
        try:
            chats = []
            for chat_file in self.chat_history_dir.glob("*.json"):
                try:
                    with open(chat_file, "r", encoding="utf-8") as f:
                        chat_data = json.load(f)
                        logger.debug(f"Raw chat data from {chat_file}: {chat_data}")
                        
                        # Handle case where chat_data might be a list
                        if isinstance(chat_data, list):
                            # Convert list to proper chat structure
                            chat_data = {
                                "messages": chat_data,
                                "created_at": datetime.now().isoformat(),
                                "name": f"Chat {chat_file.stem}",
                                "pdf_paths": [],
                                "pdf_names": [],
                                "chat_id": chat_file.stem
                            }
                        elif not isinstance(chat_data, dict):
                            logger.warning(f"Invalid chat data format in {chat_file}, skipping")
                            continue
                            
                        # Validate and ensure proper structure
                        chat_data = self._validate_chat_data(chat_data)
                        # Add chat_id from filename if not present
                        if "chat_id" not in chat_data:
                            chat_data["chat_id"] = chat_file.stem
                            
                        logger.debug(f"Processed chat data: {chat_data}")
                        chats.append(chat_data)
                except json.JSONDecodeError as e:
                    logger.error(f"Invalid JSON in chat file {chat_file}: {str(e)}")
                    continue
                except Exception as e:
                    logger.error(f"Failed to read chat file {chat_file}: {str(e)}")
                    continue
            
            # Sort by creation date, with fallback for missing dates
            sorted_chats = sorted(
                chats,
                key=lambda x: x.get("created_at", datetime.min.isoformat()),
                reverse=True
            )
            logger.debug(f"Returning {len(sorted_chats)} chats")
            return sorted_chats
        except Exception as e:
            logger.error(f"Failed to get all chats: {str(e)}")
            return []

    def rename_chat(self, chat_id: str, new_name: str) -> bool:
        """
        Rename a chat.
        
        Args:
            chat_id: Unique identifier for the chat
            new_name: New name for the chat
            
        Returns:
            True if rename was successful, False otherwise
        """
        try:
            chat = self.get_chat(chat_id)
            if chat is None:
                return False
            
            chat["name"] = new_name
            chat["updated_at"] = datetime.now().isoformat()
            self.save_chat(chat_id, chat)
            logger.info(f"Renamed chat {chat_id} to: {new_name}")
            return True
        except Exception as e:
            logger.error(f"Failed to rename chat {chat_id}: {str(e)}")
            return False

    def generate_initial_chat_name(self, pdf_name: str) -> str:
        """
        Generate an initial chat name based on the PDF filename.
        
        Args:
            pdf_name (str): Name of the uploaded PDF file
            
        Returns:
            str: Generated chat name
        """
        # Remove file extension and common words
        base_name = Path(pdf_name).stem
        # Remove common words and special characters
        words = [w for w in base_name.replace('_', ' ').replace('-', ' ').split() 
                if w.lower() not in ['document', 'file', 'pdf', 'copy', 'draft', 'final']]
        
        if words:
            # Use the first few meaningful words
            title = ' '.join(words[:3]).title()
            return f"{title} Chat"
        else:
            # Fallback to a generic name if no meaningful words found
            return f"{self.faker.word().capitalize()} {self.faker.word().capitalize()} Chat"

    def update_chat_title(self, chat_id: str, new_title: str) -> bool:
        """
        Update the title of a chat based on its content.
        
        Args:
            chat_id (str): ID of the chat to update
            new_title (str): New title for the chat
            
        Returns:
            bool: True if title was updated successfully, False otherwise
        """
        chat = self.get_chat(chat_id)
        if chat is None:
            return False
            
        chat["name"] = new_title
        self.save_chat(chat_id, chat)
        return True

    def create_new_chat(self, pdf_path: Optional[str] = None) -> Dict:
        """
        Create a new, empty chat session.
        
        Args:
            pdf_path (Optional[str]): Path to an initial PDF file
            
        Returns:
            Dict: Chat data for the new session
        """
        chat_id = str(uuid.uuid4())
        chat_name = "New Chat"
        pdf_paths = []
        pdf_names = []

        if pdf_path:
            pdf_filename = Path(pdf_path).name
            chat_name = self.generate_initial_chat_name(pdf_filename)
            pdf_paths.append(pdf_path)
            pdf_names.append(pdf_filename)

        chat_data = {
            "id": chat_id,
            "name": chat_name,
            "created_at": datetime.now().isoformat(),
            "pdf_paths": pdf_paths,
            "pdf_names": pdf_names,
            "messages": []
        }
        
        self.create_chat(chat_id, chat_data)
        return chat_data

    def append_pdf_to_chat(self, chat_id: str, pdf_path: str) -> Optional[Dict]:
        """
        Append a PDF to an existing chat.
        
        Args:
            chat_id (str): ID of the chat
            pdf_path (str): Path to the PDF file
            
        Returns:
            Optional[Dict]: Updated chat data if successful, None otherwise
        """
        chat = self.get_chat(chat_id)
        if chat is None:
            return None
            
        pdf_filename = Path(pdf_path).name
        
        # Ensure pdf_paths and pdf_names lists exist
        if "pdf_paths" not in chat:
            chat["pdf_paths"] = []
        if "pdf_names" not in chat:
            chat["pdf_names"] = []
            
        # Store the original PDF path instead of copying
        chat["pdf_paths"].append(pdf_path)  # Store original path
        chat["pdf_names"].append(pdf_filename)
        
        # Save the updated chat
        self.save_chat(chat_id, chat)
        return chat

    def add_message(self, chat_id: str, role: str, content: str) -> bool:
        """
        Add a message to a specific chat and potentially update the chat title.
        
        Args:
            chat_id (str): ID of the chat
            role (str): Role of the message sender ('user' or 'assistant')
            content (str): Message content
            
        Returns:
            bool: True if message was added successfully, False otherwise
        """
        chat = self.get_chat(chat_id)
        if chat is None:
            return False
            
        # Ensure messages list exists
        if "messages" not in chat:
            chat["messages"] = []
            
        # Add the message
        chat["messages"].append({
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat()
        })
        
        # Update title if this is the first user question
        if role == 'user' and len(chat["messages"]) == 1:
            # Create a title based on the first question
            words = content.split()
            if len(words) > 3:
                # Use first few words of the question as title
                new_title = ' '.join(words[:4]).title()
                if not new_title.endswith('?'):
                    new_title += '?'
                chat["name"] = f"{new_title} Chat"
        
        # Save the updated chat
        self.save_chat(chat_id, chat)
        return True

    def get_pdf_path(self, chat_id: str) -> Optional[str]:
        """
        Get the PDF path for a specific chat.
        
        Args:
            chat_id: ID of the chat
            
        Returns:
            Optional[str]: Path to the PDF file if found, None otherwise
        """
        chat = self.get_chat(chat_id)
        if chat and chat.get("pdf_paths"):
            return chat["pdf_paths"][0]
        return None

    def remove_pdf_from_chat(self, chat_id: str, pdf_path_to_remove: str) -> bool:
        """
        Remove a PDF from a specific chat.
        
        Args:
            chat_id: The ID of the chat.
            pdf_path_to_remove: The path of the PDF to remove.
            
        Returns:
            True if the PDF was removed successfully, False otherwise.
        """
        logger.info(f"ChatManager: Removing '{pdf_path_to_remove}' from chat '{chat_id}'")
        chat_data = self.get_chat(chat_id)
        if not chat_data:
            logger.warning(f"ChatManager: Chat '{chat_id}' not found.")
            return False

        pdf_paths = chat_data.get("pdf_paths", [])
        logger.debug(f"ChatManager: Existing PDF paths: {pdf_paths}")

        if pdf_path_to_remove not in pdf_paths:
            logger.warning(f"ChatManager: Path '{pdf_path_to_remove}' not found in chat's pdf_paths.")
            return False

        try:
            index_to_remove = pdf_paths.index(pdf_path_to_remove)
            
            pdf_paths.pop(index_to_remove)
            
            pdf_names = chat_data.get("pdf_names", [])
            if index_to_remove < len(pdf_names):
                pdf_names.pop(index_to_remove)

            chat_data["pdf_paths"] = pdf_paths
            chat_data["pdf_names"] = pdf_names

            self.save_chat(chat_id, chat_data)
            logger.info(f"ChatManager: Successfully removed PDF and saved chat '{chat_id}'.")
            return True
        except ValueError:
            logger.error(f"ChatManager: ValueError while trying to find index of '{pdf_path_to_remove}'. This should not happen.")
            return False 