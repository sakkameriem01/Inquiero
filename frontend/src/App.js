import React, { useState, useEffect, useRef } from 'react';
import { ToastContainer, toast, Slide } from 'react-toastify';
import 'react-toastify/dist/ReactToastify.css';
import { Document, Page, pdfjs } from 'react-pdf';
import {
  Plus,
  Trash2,
  FileText,
  Send,
  Moon,
  Sun,
  ChevronLeft,
  ChevronRight,
  X,
  FileUp,
  Star,
  Edit,
  Menu,
  Copy,
  Check,
} from 'lucide-react';
import './styles/globals.css';
import { Tooltip } from 'react-tooltip';
import 'react-tooltip/dist/react-tooltip.css';

// Configure react-pdf worker
pdfjs.GlobalWorkerOptions.workerSrc = '/pdf.worker.min.js';

const API_BASE_URL = 'http://localhost:8000'; // Assuming FastAPI runs on port 8000

function getToastIcon(type) {
  switch (type) {
    case 'success':
      return '✅';
    case 'error':
      return '❌';
    case 'info':
      return 'ℹ️';
    case 'warning':
      return '⚠️';
    default:
      return '🔔';
  }
}

function TypingBubble() {
  return (
    <div className="chat-bubble chat-bubble-ai typing-bubble">
      <div className="typing-dots">
        <span></span><span></span><span></span>
      </div>
    </div>
  );
}

function ChatItem({ 
  chat, 
  isSelected, 
  isFavorited, 
  onSelect, 
  onRename, 
  onFavorite, 
  onDelete, 
  renamingChatId, 
  newChatName, 
  setNewChatName,
  setRenamingChatId
}) {
  const isRenaming = renamingChatId === chat.id;

  return (
    <div 
      className={`
        group relative flex flex-col transition-all duration-200
        ${isSelected 
          ? 'bg-blue-50 dark:bg-blue-900/20 border-blue-200 dark:border-blue-800' 
          : isFavorited
            ? 'bg-amber-50/50 dark:bg-amber-900/10 border-amber-200 dark:border-amber-800/50 hover:bg-amber-50 dark:hover:bg-amber-900/20'
            : 'bg-white dark:bg-gray-800 border-gray-100 dark:border-gray-700 hover:bg-gray-50 dark:hover:bg-gray-700/50'
        }
        border rounded-xl shadow-sm hover:shadow-md
        cursor-pointer select-none
        ${isFavorited ? 'ring-1 ring-amber-200 dark:ring-amber-800/50' : ''}
      `}
    >
      <div 
        className="flex items-center justify-between p-3"
        onClick={() => onSelect(chat.id)}
      >
        <div className="flex items-center gap-3 min-w-0 flex-1">
          <div className="flex-shrink-0 relative">
            {isFavorited && (
              <div className="absolute -top-1 -right-1">
                <Star 
                  size={12} 
                  className="text-amber-400 fill-amber-400" 
                />
              </div>
            )}
            <FileText 
              size={18} 
              className={`
                ${isSelected 
                  ? 'text-blue-600 dark:text-blue-400' 
                  : isFavorited
                    ? 'text-amber-600 dark:text-amber-400'
                    : 'text-gray-400 dark:text-gray-500'
                }
                transition-colors duration-200
              `} 
            />
          </div>
          <div className="min-w-0 flex-1">
            {isRenaming ? (
              <input
                type="text"
                value={newChatName}
                onChange={(e) => setNewChatName(e.target.value)}
                onBlur={() => onRename(chat.id, newChatName)}
                onKeyPress={(e) => {
                  if (e.key === 'Enter') {
                    onRename(chat.id, newChatName);
                  }
                }}
                className={`
                  w-full px-2 py-1 text-sm rounded-md 
                  ${isFavorited 
                    ? 'border-amber-200 dark:border-amber-800 focus:ring-amber-500' 
                    : 'border-blue-200 dark:border-blue-800 focus:ring-blue-500'
                  }
                  bg-white dark:bg-gray-800 focus:ring-2 focus:border-transparent
                  transition-all duration-200
                `}
                autoFocus
              />
            ) : (
              <p className={`
                text-sm font-medium truncate
                ${isFavorited 
                  ? 'text-amber-700 dark:text-amber-300' 
                  : 'text-gray-700 dark:text-gray-200'
                }
              `}>
                {chat.name}
              </p>
            )}
          </div>
        </div>
        
        <div className="flex items-center gap-1 opacity-0 group-hover:opacity-100 transition-opacity duration-200">
          <button
            data-tooltip-id="rename-tooltip"
            data-tooltip-content="Rename chat"
            onClick={(e) => {
              e.stopPropagation();
              setRenamingChatId(chat.id);
              setNewChatName(chat.name);
            }}
            className={`
              p-1.5 rounded-lg transition-colors duration-200
              ${isFavorited
                ? 'hover:bg-amber-100 dark:hover:bg-amber-900/30 text-amber-600 dark:text-amber-400'
                : 'hover:bg-gray-100 dark:hover:bg-gray-700 text-gray-500 hover:text-gray-700 dark:text-gray-400 dark:hover:text-gray-200'
              }
            `}
          >
            <Edit size={14} />
          </button>
          
          <button
            data-tooltip-id="favorite-tooltip"
            data-tooltip-content={isFavorited ? "Remove from favorites" : "Add to favorites"}
            onClick={(e) => {
              e.stopPropagation();
              onFavorite(chat.id);
            }}
            className={`
              p-1.5 rounded-lg transition-colors duration-200
              ${isFavorited
                ? 'hover:bg-amber-100 dark:hover:bg-amber-900/30'
                : 'hover:bg-gray-100 dark:hover:bg-gray-700'
              }
            `}
          >
            <Star 
              size={14} 
              className={isFavorited 
                ? 'text-amber-400 fill-amber-400' 
                : 'text-gray-400 dark:text-gray-500 hover:text-amber-400 dark:hover:text-amber-400'
              } 
            />
          </button>
          
          <button
            data-tooltip-id="delete-tooltip"
            data-tooltip-content="Delete chat"
            onClick={(e) => {
              e.stopPropagation();
              onDelete(chat.id);
            }}
            className={`
              p-1.5 rounded-lg transition-colors duration-200
              ${isFavorited
                ? 'hover:bg-red-50 dark:hover:bg-red-900/30 text-amber-600/70 hover:text-red-500 dark:text-amber-400/70 dark:hover:text-red-400'
                : 'hover:bg-red-50 dark:hover:bg-red-900/30 text-gray-400 hover:text-red-500 dark:text-gray-500 dark:hover:text-red-400'
              }
            `}
          >
            <Trash2 size={14} />
          </button>
        </div>
      </div>
    </div>
  );
}

// Custom Confirmation Dialog Component
function ConfirmationDialog({ 
  isOpen, 
  onClose, 
  onConfirm, 
  title, 
  message, 
  confirmText = "Delete", 
  cancelText = "Cancel",
  type = "danger" // "danger", "warning", "info"
}) {
  if (!isOpen) return null;

  const handleConfirm = () => {
    onConfirm();
    onClose();
  };

  const handleBackdropClick = (e) => {
    if (e.target === e.currentTarget) {
      onClose();
    }
  };

  const getTypeStyles = () => {
    switch (type) {
      case "danger":
        return {
          icon: "🗑️",
          confirmButton: "bg-red-600 hover:bg-red-700 focus:ring-red-500",
          iconBg: "bg-red-100 dark:bg-red-900/20",
          iconColor: "text-red-600 dark:text-red-400"
        };
      case "warning":
        return {
          icon: "⚠️",
          confirmButton: "bg-yellow-600 hover:bg-yellow-700 focus:ring-yellow-500",
          iconBg: "bg-yellow-100 dark:bg-yellow-900/20",
          iconColor: "text-yellow-600 dark:text-yellow-400"
        };
      case "info":
        return {
          icon: "ℹ️",
          confirmButton: "bg-blue-600 hover:bg-blue-700 focus:ring-blue-500",
          iconBg: "bg-blue-100 dark:bg-blue-900/20",
          iconColor: "text-blue-600 dark:text-blue-400"
        };
      default:
        return {
          icon: "❓",
          confirmButton: "bg-gray-600 hover:bg-gray-700 focus:ring-gray-500",
          iconBg: "bg-gray-100 dark:bg-gray-900/20",
          iconColor: "text-gray-600 dark:text-gray-400"
        };
    }
  };

  const styles = getTypeStyles();

  return (
    <div 
      className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center p-4 z-50 confirmation-dialog-backdrop"
      onClick={handleBackdropClick}
    >
      <div className="bg-white dark:bg-gray-800 rounded-2xl shadow-2xl max-w-md w-full transform transition-all duration-200 scale-100 confirmation-dialog-content confirmation-dialog">
        {/* Header */}
        <div className="flex items-center gap-4 p-6 border-b border-gray-200 dark:border-gray-700">
          <div className={`p-3 rounded-full ${styles.iconBg}`}>
            <span className="text-2xl">{styles.icon}</span>
          </div>
          <div className="flex-1">
            <h3 className="text-xl font-semibold text-gray-900 dark:text-gray-100">
              {title}
            </h3>
          </div>
          <button
            onClick={onClose}
            className="p-2 rounded-lg text-gray-400 hover:text-gray-600 dark:hover:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-700 transition-colors"
          >
            <X size={20} />
          </button>
        </div>

        {/* Content */}
        <div className="p-6">
          <p className="text-gray-600 dark:text-gray-300 leading-relaxed">
            {message}
          </p>
        </div>

        {/* Actions */}
        <div className="flex gap-3 p-6 pt-0">
          <button
            onClick={onClose}
            className="flex-1"
          >
            {cancelText}
          </button>
          <button
            onClick={handleConfirm}
            className="flex-1"
          >
            {confirmText}
          </button>
        </div>
      </div>
    </div>
  );
}

function App() {
  const [chats, setChats] = useState([]);
  const [currentChatId, setCurrentChatId] = useState(null);
  const [messages, setMessages] = useState([]);
  const [question, setQuestion] = useState('');
  const [loading, setLoading] = useState(false);
  const [showPdfModal, setShowPdfModal] = useState(false);
  const [pdfUrlToPreview, setPdfUrlToPreview] = useState(null);
  const [numPages, setNumPages] = useState(null);
  const [pageNumber, setPageNumber] = useState(1);
  const [isDarkMode, setIsDarkMode] = useState(false);
  const [showTyping, setShowTyping] = useState(false);
  const typingTimeoutRef = useRef(null);

  const messagesEndRef = useRef(null);
  const mainFileInputRef = useRef(null);

  const [favoritedChats, setFavoritedChats] = useState(new Set());
  const [attachedPDFs, setAttachedPDFs] = useState([]);
  const [userLanguage, setUserLanguage] = useState('en');

  // Add a new state for the chat being renamed
  const [renamingChatId, setRenamingChatId] = useState(null);
  const [newChatName, setNewChatName] = useState('');

  // Add state for copy button confirmation
  const [copiedMessageIndex, setCopiedMessageIndex] = useState(null);

  // Add state for message editing
  const [editingMessageIndex, setEditingMessageIndex] = useState(null);
  const [editingMessageText, setEditingMessageText] = useState('');

  // Add state for confirmation dialogs
  const [showDeleteChatDialog, setShowDeleteChatDialog] = useState(false);
  const [showDeleteAllChatsDialog, setShowDeleteAllChatsDialog] = useState(false);
  const [chatToDelete, setChatToDelete] = useState(null);

  // Add state to track if this is a new chat for welcome message
  const [isNewChat, setIsNewChat] = useState(false);

  const [isSidebarOpen, setIsSidebarOpen] = useState(() => {
    const saved = localStorage.getItem('sidebarOpen');
    return saved !== null ? JSON.parse(saved) : true;
  });

  const onDocumentLoadSuccess = ({ numPages }) => {
    setNumPages(numPages);
    setPageNumber(1); // Reset to first page when new PDF loads
  };

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  const handleFileUpload = async (event) => {
    const files = Array.from(event.target.files);
    if (files.length === 0) return;

    // If we have a current chat, append to it; otherwise create new chat
    await uploadFilesToChat(files, currentChatId);
  };

  const handleDragOver = (e) => {
    e.preventDefault();
    e.stopPropagation();
  };

  const handleDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    
    const files = Array.from(e.dataTransfer.files).filter(file => file.type === 'application/pdf');
    if (files.length > 0) {
      const event = { target: { files } };
      handleFileUpload(event);
    }
  };

  const handleSendMessage = async (e) => {
    e.preventDefault();
    if (!question.trim() || !currentChatId) return;

    // Detect user's language
    try {
      const response = await fetch(`${API_BASE_URL}/detect-language/`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ text: question }),
      });
      
      if (response.ok) {
        const data = await response.json();
        setUserLanguage(data.language);
      }
    } catch (e) {
      console.error('Language detection failed:', e);
    }

    const userMessage = { role: 'user', content: question };
    setMessages((prevMessages) => [...prevMessages, userMessage]);
    setQuestion('');
    setIsNewChat(false); // Set to false once user sends first message
    setLoading(true);
    setShowTyping(false);
    if (typingTimeoutRef.current) clearTimeout(typingTimeoutRef.current);
    typingTimeoutRef.current = setTimeout(() => setShowTyping(true), 300);

    try {
      const response = await fetch(`${API_BASE_URL}/chats/${currentChatId}/message/`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ 
          text: userMessage.content,
          sender: 'user',
          language: userLanguage
        }),
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(`HTTP error! status: ${response.status} - ${errorData.detail || response.statusText}`);
      }

      const data = await response.json();
      setMessages((prevMessages) => [...prevMessages, { role: 'assistant', content: data.response }]);
      fetchChats();
      toast.success("Answer generated! 💬");

    } catch (e) {
      setMessages((prevMessages) => prevMessages.slice(0, prevMessages.length - 1)); // Remove user message if failed
      toast.error(`Failed to generate answer: ${e.message} ⚠️`);
    } finally {
      setLoading(false);
      setShowTyping(false); // Hide typing animation when response is complete
      if (typingTimeoutRef.current) {
        clearTimeout(typingTimeoutRef.current);
        typingTimeoutRef.current = null;
      }
    }
  };

  const handleDeleteChat = async (chatId) => {
    // Find the chat to get its name for the dialog
    const chat = chats.find(c => c.id === chatId);
    setChatToDelete(chat);
    setShowDeleteChatDialog(true);
  };

  const confirmDeleteChat = async () => {
    if (!chatToDelete) return;
    
    setLoading(true);
    try {
      const response = await fetch(`${API_BASE_URL}/chats/${chatToDelete.id}/`, {
        method: 'DELETE',
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(`HTTP error! status: ${response.status} - ${errorData.detail || response.statusText}`);
      }

      if (currentChatId === chatToDelete.id) {
        setCurrentChatId(null);
        setMessages([]);
        setAttachedPDFs([]);
      }
      setChats(prevChats => prevChats.filter(chat => chat.id !== chatToDelete.id));
      toast.success("Chat deleted successfully! 🗑️");
    } catch (e) {
      toast.error(`Failed to delete chat: ${e.message} ⚠️`);
    } finally {
      setLoading(false);
      setChatToDelete(null);
    }
  };

  const handleDeleteAllChats = async () => {
    setShowDeleteAllChatsDialog(true);
  };

  const handleRenameChat = async (chatId, newName) => {
    if (!newName.trim()) {
      setRenamingChatId(null);
      setNewChatName('');
      return;
    }

    try {
      const response = await fetch(`${API_BASE_URL}/chats/${chatId}/rename/`, {
        method: 'PUT',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ name: newName.trim() }),
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(`HTTP error! status: ${response.status} - ${errorData.detail || response.statusText}`);
      }

      // Update the chat name in the local state
      setChats(prevChats => 
        prevChats.map(chat => 
          chat.id === chatId 
            ? { ...chat, name: newName.trim() }
            : chat
        )
      );

      toast.success("Chat renamed successfully! ✏️");
    } catch (error) {
      toast.error(`Failed to rename chat: ${error.message} ⚠️`);
      // Reset to original name on error
      const originalChat = chats.find(c => c.id === chatId);
      if (originalChat) {
        setNewChatName(originalChat.name);
      }
    } finally {
      setRenamingChatId(null);
      setNewChatName('');
    }
  };

  const confirmDeleteAllChats = async () => {
    setLoading(true);
    try {
      console.log("Sending delete all chats request...");
      const response = await fetch(`${API_BASE_URL}/chats/all/`, {
        method: 'DELETE',
      });

      console.log("Delete all chats response status:", response.status);
      
      if (!response.ok) {
        const errorData = await response.json();
        console.error("Delete all chats error:", errorData);
        throw new Error(`HTTP error! status: ${response.status} - ${errorData.detail || response.statusText}`);
      }

      const result = await response.json();
      console.log("Delete all chats success:", result);

      setCurrentChatId(null);
      setChats([]);
      setMessages([]);
      setAttachedPDFs([]);
      toast.success("All chats deleted! 🗑️");
      
      // Refresh the chat list to ensure UI is updated
      await fetchChats();
    } catch (e) {
      console.error("Delete all chats failed:", e);
      toast.error(`Failed to delete all chats: ${e.message} ⚠️`);
    } finally {
      setLoading(false);
    }
  };

  const handlePdfPreview = (pdfPath) => {
    if (!pdfPath) {
      toast.error("PDF path is not available.");
      return;
    }
    
    try {
      // Extract the filename from the full path
      const filename = pdfPath.split('/').pop() || pdfPath.split('\\').pop();
      if (!filename) {
        toast.error("Invalid PDF path.");
        return;
      }
      
      const pdfUrl = `${API_BASE_URL}/pdf/${filename}`;
      setPdfUrlToPreview(pdfUrl);
      setShowPdfModal(true);
    } catch (error) {
      console.error("Error constructing PDF URL:", error);
      toast.error("Failed to preview PDF.");
    }
  };

  const toggleDarkMode = () => {
    setIsDarkMode(!isDarkMode);
    document.documentElement.classList.toggle('dark');
  };

  const toggleSidebar = () => {
    setIsSidebarOpen(!isSidebarOpen);
    localStorage.setItem('sidebarOpen', JSON.stringify(!isSidebarOpen));
  };

  const handleRemoveFile = async (filePathToRemove) => {
    if (!currentChatId) return;

    const originalFiles = [...attachedPDFs];
    setAttachedPDFs(prev => prev.filter(file => file.path !== filePathToRemove));

    try {
      const response = await fetch(`${API_BASE_URL}/chats/${currentChatId}/pdfs`, {
        method: 'DELETE',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ pdf_path: filePathToRemove }),
      });

      if (!response.ok) {
        toast.error("Failed to remove file. Reverting changes.");
        setAttachedPDFs(originalFiles);
      } else {
        toast.success("File removed successfully.");
      }
    } catch (error) {
      toast.error("An error occurred. Reverting changes.");
      setAttachedPDFs(originalFiles);
    }
  };

  const handleCopy = (text, messageIndex) => {
    navigator.clipboard.writeText(text).then(() => {
      // Set the confirmation state for this specific message
      setCopiedMessageIndex(messageIndex);
      
      // Reset the confirmation state after 2.5 seconds
      setTimeout(() => {
        setCopiedMessageIndex(null);
      }, 2500);
      
      toast.success("Copied to clipboard!");
    }, (err) => {
      toast.error("Failed to copy text.");
      console.error('Could not copy text: ', err);
    });
  };

  const handleEditMessage = (messageIndex, currentText) => {
    setEditingMessageIndex(messageIndex);
    setEditingMessageText(currentText);
  };

  const handleSaveEdit = async (messageIndex) => {
    if (!editingMessageText.trim()) {
      toast.error("Message cannot be empty.");
      return;
    }

    try {
      // Update the message in local state
      setMessages(prevMessages => 
        prevMessages.map((msg, index) => 
          index === messageIndex 
            ? { ...msg, content: editingMessageText.trim() }
            : msg
        )
      );

      // TODO: Send update to backend if needed
      // For now, we'll just update the local state
      
      toast.success("Message updated successfully! ✏️");
    } catch (error) {
      toast.error("Failed to update message.");
      console.error("Error updating message:", error);
    } finally {
      setEditingMessageIndex(null);
      setEditingMessageText('');
    }
  };

  const handleCancelEdit = () => {
    setEditingMessageIndex(null);
    setEditingMessageText('');
  };

  const handleKeyDown = (event) => {
    if (event.key === 'Enter' && !event.shiftKey) {
      event.preventDefault();
      handleSendMessage(event);
    }
  };

  // Keyboard shortcut to toggle sidebar (Ctrl/Cmd + B)
  useEffect(() => {
    const handleGlobalKeyDown = (event) => {
      if ((event.ctrlKey || event.metaKey) && event.key === 'b') {
        event.preventDefault();
        toggleSidebar();
      }
    };

    document.addEventListener('keydown', handleGlobalKeyDown);
    return () => document.removeEventListener('keydown', handleGlobalKeyDown);
  }, [isSidebarOpen, toggleSidebar]);

  const toggleFavorite = (chatId) => {
    setFavoritedChats(prev => {
      const newFavorited = new Set(prev);
      if (newFavorited.has(chatId)) {
        newFavorited.delete(chatId);
      } else {
        newFavorited.add(chatId);
      }
      return newFavorited;
    });
  };

  const handleSelectChat = (chatId) => {
    if (chatId === currentChatId) return;
    setCurrentChatId(chatId);
    setIsNewChat(false); // Set to false for existing chat
    fetchChatMessages(chatId);
  };

  const uploadFilesToChat = async (files, targetChatId = null) => {
    let finalChatId = targetChatId;

    // If no targetChatId is provided, and there's no active chat, create a new one.
    if (!finalChatId && !currentChatId) {
      const newChatData = await createNewChat();
      if (newChatData) {
        finalChatId = newChatData.id;
        setCurrentChatId(newChatData.id); // Set the new chat as active
        setIsNewChat(true); // Set to true for new chat
      } else {
        toast.error("Could not create a new chat session.");
        return;
      }
    } else if (!finalChatId && currentChatId) {
      // If there is an active chat, use its ID.
      finalChatId = currentChatId;
    }

    if (files.length === 0) {
      // If no files are being uploaded, we just wanted to create a new chat.
      if (!targetChatId) fetchChats();
      return;
    }

    setLoading(true);
    toast.info(`Uploading and processing ${files.length} PDF(s)...`);

    const formData = new FormData();
    files.forEach(file => {
      formData.append('file', file);
    });

    // Important: FormData sends chat_id as a separate part.
    if (finalChatId) {
      formData.append('chat_id', finalChatId);
    }

    try {
      const response = await fetch(`${API_BASE_URL}/upload-pdf/`, {
        method: 'POST',
        body: formData, // No 'Content-Type' header needed for FormData
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(`HTTP error! status: ${response.status} - ${errorData.detail || response.statusText}`);
      }

      const result = await response.json();
      
      // After upload, fetch the latest state of the chat.
      await fetchChatMessages(finalChatId);
      await fetchChats(); // Update the list of chats on the sidebar

      toast.success(`${files.length} PDF${files.length > 1 ? 's' : ''} processed successfully! ✅`);
    } catch (e) {
      toast.error(`PDF upload failed: ${e.message} ❌`);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchChats();
  }, []);

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  useEffect(() => {
    // Check for saved theme preference
    const savedTheme = localStorage.getItem('theme');
    if (savedTheme === 'dark') {
      setIsDarkMode(true);
      document.documentElement.classList.add('dark');
    }
  }, []);

  useEffect(() => {
    return () => {
      if (typingTimeoutRef.current) clearTimeout(typingTimeoutRef.current);
    };
  }, []);

  const fetchChats = async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/chats/`);
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      const data = await response.json();
      if (data && Array.isArray(data.chats)) {
        setChats(data.chats);
      } else {
        setChats([]);
        console.warn('Received invalid chat data format:', data);
      }
    } catch (e) {
      setChats([]); // Clear chats on error
    }
  };

  const fetchChatMessages = async (chatId) => {
    setLoading(true);
    try {
      const response = await fetch(`${API_BASE_URL}/chats/${chatId}/`);
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      const data = await response.json();
      if (data && Array.isArray(data.messages)) {
        setMessages(data.messages);
        // Load associated PDFs
        if (data.pdf_paths) {
          setAttachedPDFs(data.pdf_paths.map((path, index) => ({
            path: path,
            name: data.pdf_names?.[index] || path.split('/').pop(),
            language: data.language || 'en'
          })));
        }
      } else {
        setMessages([]);
        setAttachedPDFs([]);
        console.warn('Received invalid chat message data format:', data);
      }
    } catch (e) {
      setMessages([]); // Clear messages on error
      setAttachedPDFs([]); // Clear PDFs on error
      toast.error(`Failed to load chat: ${e.message}`);
    } finally {
      setLoading(false);
    }
  };

  const createNewChat = async () => {
    try {
      // Assuming you have an endpoint to create a new, empty chat
      const response = await fetch(`${API_BASE_URL}/chats`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({}), // No initial data needed
      });
      if (!response.ok) throw new Error('Failed to create chat');
      const newChat = await response.json();
      setChats(prev => [newChat, ...prev]);
      setCurrentChatId(newChat.id);
      setMessages([]);
      setAttachedPDFs([]);
      setIsNewChat(true); // Set to true for new chat
      return newChat;
    } catch (error) {
      toast.error("Failed to create a new chat.");
      return null;
    }
  };

  return (
    <div className={`flex h-screen bg-background-light dark:bg-background-dark text-text-primary-light dark:text-text-primary-dark transition-colors duration-200 ${!isSidebarOpen ? 'sidebar-closed' : ''}`}>
      <Tooltip id="delete-all-tooltip" place="bottom" delayShow={300} delayHide={100} />
      <Tooltip id="new-chat-tooltip" place="bottom" delayShow={300} delayHide={100} />
      <Tooltip id="rename-tooltip" place="bottom" delayShow={300} delayHide={100} />
      <Tooltip id="favorite-tooltip" place="bottom" delayShow={300} delayHide={100} />
      <Tooltip id="delete-tooltip" place="bottom" delayShow={300} delayHide={100} />
      <Tooltip id="sidebar-tooltip" place="bottom" delayShow={300} delayHide={100} />
      <Tooltip id="upload-tooltip" place="bottom" delayShow={300} delayHide={100} />
      <Tooltip id="remove-file-tooltip" place="top" delayShow={300} delayHide={100} />
      <Tooltip id="copy-tooltip" place="top" delayShow={300} delayHide={100} />
      <Tooltip id="edit-tooltip" place="top" delayShow={300} delayHide={100} />
      <ToastContainer
        position="bottom-right"
        autoClose={4000}
        hideProgressBar={false}
        newestOnTop
        closeOnClick
        rtl={false}
        pauseOnFocusLoss
        draggable
        pauseOnHover
        theme={isDarkMode ? "dark" : "light"}
        limit={4}
        transition={Slide}
        icon={({ type }) => getToastIcon(type)}
        toastClassName="!font-sans"
        bodyClassName="!font-sans"
        progressClassName="!bg-gradient-to-r !from-purple-500 !to-blue-500"
      />

      <div className="flex flex-col flex-1 h-full bg-gray-100 dark:bg-gray-900 transition-all duration-300">
        {/* Top Bar */}
        <div className="flex items-center justify-between p-2 border-b border-gray-200 dark:border-gray-700 bg-white dark:bg-gray-800">
          <button 
            onClick={toggleSidebar} 
            className="p-2 rounded-md hover:bg-gray-100 dark:hover:bg-gray-700"
            data-tooltip-id="sidebar-tooltip"
            data-tooltip-content={isSidebarOpen ? "Collapse Sidebar" : "Expand Sidebar"}
          >
            <Menu size={20} />
          </button>
          <h3 className="text-2xl font-extrabold tracking-wide gradient-text">Inquiero</h3>
          <button onClick={toggleDarkMode} className="p-2 rounded-md hover:bg-gray-100 dark:hover:bg-gray-700">
            {isDarkMode ? <Sun size={20} /> : <Moon size={20} />}
          </button>
        </div>

        {/* Main Content */}
        <div className="flex-1 flex overflow-hidden">
          {/* Sidebar */}
          <div className={`
            ${isSidebarOpen ? 'w-80' : 'w-0'}
            bg-white dark:bg-gray-800/50 border-r border-gray-200 dark:border-gray-700
            transition-all duration-300 overflow-hidden flex flex-col
          `}>
            {isSidebarOpen && (
              <div className="p-4 flex-1 flex flex-col overflow-hidden">
                <div className="flex justify-between items-center mb-4">
                  <h2 className="text-xl font-bold text-gray-800 dark:text-gray-100">Chats</h2>
                  <div className="flex gap-1">
                    <button
                      onClick={handleDeleteAllChats}
                      className="btn-icon btn-icon-danger"
                      data-tooltip-id="delete-all-tooltip"
                      data-tooltip-content="Delete All Chats"
                    >
                      <Trash2 size={16} />
                    </button>
                    <button
                      onClick={() => uploadFilesToChat([])}
                      className="btn-icon btn-icon-primary"
                      data-tooltip-id="new-chat-tooltip"
                      data-tooltip-content="New Chat"
                    >
                      <Plus size={16} />
                    </button>
                  </div>
                </div>
                <div className="flex-1 overflow-y-auto pr-1 -mr-1 space-y-2">
                  {favoritedChats.size > 0 && (
                    <div className="mb-4">
                      <h3 className="text-xs font-semibold text-gray-400 uppercase tracking-wider mb-2">Favorites</h3>
                      {chats
                        .filter(chat => favoritedChats.has(chat.id))
                        .map(chat => (
                          <ChatItem
                            key={chat.id}
                            chat={chat}
                            isSelected={currentChatId === chat.id}
                            isFavorited={true}
                            onSelect={handleSelectChat}
                            onRename={handleRenameChat}
                            onFavorite={toggleFavorite}
                            onDelete={handleDeleteChat}
                            renamingChatId={renamingChatId}
                            newChatName={newChatName}
                            setNewChatName={setNewChatName}
                            setRenamingChatId={setRenamingChatId}
                          />
                        ))}
                    </div>
                  )}

                  <h3 className="text-xs font-semibold text-gray-400 uppercase tracking-wider mb-2">Recent</h3>
                  {chats
                    .filter(chat => !favoritedChats.has(chat.id))
                    .map(chat => (
                      <ChatItem
                        key={chat.id}
                        chat={chat}
                        isSelected={currentChatId === chat.id}
                        isFavorited={false}
                        onSelect={handleSelectChat}
                        onRename={handleRenameChat}
                        onFavorite={toggleFavorite}
                        onDelete={handleDeleteChat}
                        renamingChatId={renamingChatId}
                        newChatName={newChatName}
                        setNewChatName={setNewChatName}
                        setRenamingChatId={setRenamingChatId}
                      />
                    ))}
                </div>
              </div>
            )}
          </div>

          {/* Main Chat Area */}
          <div className="flex flex-col flex-1 h-full overflow-hidden">
            {currentChatId ? (
              <>
                {/* Chat Header */}
                <div className="flex items-center justify-between p-4 border-b border-gray-200 dark:border-gray-700 bg-white dark:bg-gray-800/50 flex-shrink-0">
                  <h2 className="text-xl font-semibold text-gray-800 dark:text-gray-100 truncate">
                    {chats.find(c => c.id === currentChatId)?.name || 'Chat'}
                  </h2>
                  <div className="flex items-center gap-2">
                    <button
                      onClick={() => mainFileInputRef.current.click()}
                      className="flex items-center gap-2 px-3 py-2 text-sm font-medium text-white bg-purple-600 rounded-lg hover:bg-purple-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-purple-500 dark:bg-purple-500 dark:hover:bg-purple-600 transition-colors"
                      data-tooltip-id="upload-tooltip"
                      data-tooltip-content="Upload more files to this chat"
                    >
                      <FileUp size={16} />
                      <span>Upload More</span>
                    </button>
                  </div>
                </div>

                {/* Attached Files Bar */}
                {attachedPDFs.length > 0 && (
                  <div className="p-3 border-b border-gray-200 dark:border-gray-700 bg-gray-50 dark:bg-gray-800 flex-shrink-0">
                    <div className="flex items-center gap-2 flex-wrap">
                      <span className="text-sm font-medium text-gray-600 dark:text-gray-300 mr-2">Attached Files ({attachedPDFs.length})</span>
                      {attachedPDFs.map((pdf, index) => (
                        <div key={index} className="group relative flex items-center bg-blue-100/50 dark:bg-blue-900/20 text-blue-800 dark:text-blue-200 rounded-full text-xs">
                          <button
                              onClick={() => handlePdfPreview(pdf.path)}
                              className="flex items-center gap-1.5 pl-3 pr-2 py-1 hover:bg-blue-100 dark:hover:bg-blue-900/40 rounded-l-full"
                          >
                            <FileText size={12} />
                            <span className="truncate max-w-xs">{pdf.name}</span>
                          </button>
                          <button
                            onClick={() => handleRemoveFile(pdf.path)}
                            className="p-1 rounded-full hover:bg-red-200 dark:hover:bg-red-900/50 text-red-500 opacity-50 group-hover:opacity-100 transition-opacity"
                            data-tooltip-id="remove-file-tooltip"
                            data-tooltip-content={`Remove ${pdf.name}`}
                          >
                            <X size={12} />
                          </button>
                        </div>
                      ))}
                    </div>
                  </div>
                )}

                {/* Messages Area */}
                <div className="flex-1 overflow-y-auto p-4 md:p-6 space-y-6">
                  {/* Welcome message for new chats */}
                  {isNewChat && messages.length === 0 && (
                    <div className="group">
                      <div className="chat-bubble chat-bubble-ai">
                        <div className="chat-bubble-content">
                          <p className="text-lg font-semibold mb-2">👋 Welcome to Inquiero!</p>
                          <p className="mb-3">I'm here to help you analyze and understand your PDF documents. You can ask me questions about:</p>
                          <ul className="list-disc list-inside space-y-1 text-sm text-gray-600 dark:text-gray-300 mb-3">
                            <li>Specific content within your documents</li>
                            <li>Summaries and key points</li>
                            <li>Data analysis and insights</li>
                            <li>Comparisons between different sections</li>
                            <li>Definitions and explanations</li>
                          </ul>
                          <p className="text-sm text-gray-600 dark:text-gray-300">Just type your question below and I'll help you find the information you need!</p>
                        </div>
                      </div>
                    </div>
                  )}
                  
                  {messages.map((msg, index) => (
                    <div key={index} className="group">
                      <div className={`chat-bubble ${msg.role === 'user' ? 'chat-bubble-user' : 'chat-bubble-ai'}`}>
                        <div className="chat-bubble-content">
                          {editingMessageIndex === index ? (
                            <div className="space-y-3">
                              <textarea
                                value={editingMessageText}
                                onChange={(e) => setEditingMessageText(e.target.value)}
                                onKeyDown={(e) => {
                                  if (e.key === 'Enter' && !e.shiftKey) {
                                    e.preventDefault();
                                    handleSaveEdit(index);
                                  } else if (e.key === 'Escape') {
                                    handleCancelEdit();
                                  }
                                }}
                                className="w-full p-3 rounded-lg border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100 focus:ring-2 focus:ring-purple-500 focus:border-transparent resize-none"
                                rows="3"
                                autoFocus
                              />
                              <div className="flex items-center gap-2">
                                <button
                                  onClick={() => handleSaveEdit(index)}
                                  className="px-3 py-1.5 bg-green-600 text-white rounded-lg text-sm font-medium hover:bg-green-700 transition-colors"
                                >
                                  Save
                                </button>
                                <button
                                  onClick={handleCancelEdit}
                                  className="px-3 py-1.5 bg-gray-500 text-white rounded-lg text-sm font-medium hover:bg-gray-600 transition-colors"
                                >
                                  Cancel
                                </button>
                              </div>
                            </div>
                          ) : (
                            typeof msg.content === 'string' ? (
                              msg.content.split('\\n').map((line, i) => (
                                <p key={i}>{line}</p>
                              ))
                            ) : (
                              <p>{JSON.stringify(msg.content)}</p>
                            )
                          )}
                        </div>
                      </div>
                      {/* Copy and Edit buttons positioned below the chat bubble */}
                      <div className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'} mt-2 opacity-0 group-hover:opacity-100 transition-opacity duration-200 gap-2`}>
                        {msg.role === 'user' && editingMessageIndex !== index && (
                          <button
                            onClick={(e) => {
                              e.stopPropagation();
                              handleEditMessage(index, msg.content);
                            }}
                            className="px-3 py-1.5 rounded-lg transition-all duration-200 text-sm font-medium text-gray-500 dark:text-gray-400 hover:bg-gray-100 dark:hover:bg-gray-700 hover:text-gray-700 dark:hover:text-gray-200"
                            data-tooltip-id="edit-tooltip"
                            data-tooltip-content="Edit message"
                          >
                            <div className="flex items-center gap-1.5">
                              <Edit size={14} />
                            </div>
                          </button>
                        )}
                        <button
                          onClick={(e) => {
                            e.stopPropagation();
                            handleCopy(msg.content, index);
                          }}
                          className={`px-3 py-1.5 rounded-lg transition-all duration-200 text-sm font-medium ${
                            copiedMessageIndex === index
                              ? 'text-green-600 bg-green-100 dark:bg-green-900/30 dark:text-green-400'
                              : 'text-gray-500 dark:text-gray-400 hover:bg-gray-100 dark:hover:bg-gray-700 hover:text-gray-700 dark:hover:text-gray-200'
                          }`}
                          data-tooltip-id="copy-tooltip"
                          data-tooltip-content={copiedMessageIndex === index ? "Copied!" : "Copy message"}
                        >
                          <div className="flex items-center gap-1.5">
                            {copiedMessageIndex === index ? (
                              <>
                                <Check size={14} />
                                <span>Copied!</span>
                              </>
                            ) : (
                              <Copy size={14} />
                            )}
                          </div>
                        </button>
                      </div>
                    </div>
                  ))}
                  {showTyping && <TypingBubble />}
                  <div ref={messagesEndRef} />
                </div>

                {/* Message Input */}
                <div className="p-4 border-t border-gray-200 dark:border-gray-700 bg-white dark:bg-gray-800/50">
                  <form onSubmit={handleSendMessage} className="relative">
                    <textarea
                      value={question}
                      onChange={(e) => setQuestion(e.target.value)}
                      onKeyDown={handleKeyDown}
                      placeholder="Ask a question about your document..."
                      className="w-full pl-4 pr-12 py-3 rounded-lg border border-gray-300 dark:border-gray-600 bg-gray-50 dark:bg-gray-700 text-gray-900 dark:text-gray-100 focus:ring-2 focus:ring-purple-500 focus:border-transparent resize-none"
                      rows="1"
                      style={{ minHeight: '52px', maxHeight: '200px' }}
                    />
                    <button
                      type="submit"
                      disabled={loading}
                      className="absolute right-3 top-1/2 -translate-y-1/2 p-2 rounded-full bg-purple-600 text-white hover:bg-purple-700 disabled:bg-purple-300 dark:disabled:bg-purple-800"
                    >
                      <Send size={20} />
                    </button>
                  </form>
                </div>
              </>
            ) : (
              // Welcome Screen / Initial State
              <div className="flex flex-col items-center justify-center h-full text-center text-gray-500 dark:text-gray-400 p-8">
                <div
                  className="flex flex-col items-center justify-center w-full max-w-lg min-h-[400px] border-2 border-dashed border-gray-300 dark:border-gray-600 rounded-2xl bg-white dark:bg-gray-800/20 p-8 transition-colors duration-200 hover:border-purple-400 dark:hover:border-purple-500 hover:bg-gray-50 dark:hover:bg-gray-800/40 cursor-pointer"
                  onDragOver={handleDragOver}
                  onDrop={handleDrop}
                  onClick={() => mainFileInputRef.current.click()}
                >
                  <FileUp size={50} className="mb-4 text-purple-400 dark:text-purple-500" />
                  <h2 className="text-2xl font-bold text-gray-800 dark:text-gray-100 mb-2">Drop your PDF here</h2>
                  <p className="text-gray-600 dark:text-gray-300">or click to browse</p>
                </div>
              </div>
            )}
          </div>
        </div>
      </div>

      {/* PDF Preview Modal */}
      {showPdfModal && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center p-4 z-50 backdrop-blur-sm">
          <div className="bg-white dark:bg-gray-800 rounded-2xl shadow-2xl max-w-4xl w-full max-h-[90vh] overflow-y-auto border border-gray-200 dark:border-gray-700">
            <div className="flex justify-between items-center p-6 border-b border-gray-200 dark:border-gray-700">
              <h3 className="text-xl font-semibold text-gray-900 dark:text-gray-100">PDF Preview</h3>
              <button
                onClick={() => setShowPdfModal(false)}
                className="p-2 rounded-lg text-gray-400 hover:text-gray-600 dark:hover:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-700 transition-colors"
              >
                <X size={24} />
              </button>
            </div>
            <div className="p-6">
              <Document
                file={pdfUrlToPreview}
                onLoadSuccess={onDocumentLoadSuccess}
                className="flex justify-center"
              >
                <Page
                  pageNumber={pageNumber}
                  renderTextLayer={false}
                  renderAnnotationLayer={false}
                  className="shadow-lg rounded-lg"
                />
              </Document>
            </div>
            <div className="flex justify-center items-center gap-4 p-6 border-t border-gray-200 dark:border-gray-700">
              <button
                onClick={() => setPageNumber(Math.max(1, pageNumber - 1))}
                disabled={pageNumber <= 1}
                className="p-2 rounded-lg text-gray-400 hover:text-gray-600 dark:hover:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-700 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
              >
                <ChevronLeft size={20} />
              </button>
              <span className="px-4 py-2 text-sm font-medium text-gray-700 dark:text-gray-300 bg-gray-100 dark:bg-gray-700 rounded-lg">
                Page {pageNumber} of {numPages}
              </span>
              <button
                onClick={() => setPageNumber(Math.min(numPages, pageNumber + 1))}
                disabled={pageNumber >= numPages}
                className="p-2 rounded-lg text-gray-400 hover:text-gray-600 dark:hover:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-700 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
              >
                <ChevronRight size={20} />
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Confirmation Dialogs */}
      <ConfirmationDialog
        isOpen={showDeleteChatDialog}
        onClose={() => {
          setShowDeleteChatDialog(false);
          setChatToDelete(null);
        }}
        onConfirm={confirmDeleteChat}
        title="Delete Chat"
        message={`Are you sure you want to delete "${chatToDelete?.name || 'this chat'}"? This action cannot be undone.`}
        confirmText="Delete Chat"
        cancelText="Cancel"
        type="danger"
      />

      <ConfirmationDialog
        isOpen={showDeleteAllChatsDialog}
        onClose={() => setShowDeleteAllChatsDialog(false)}
        onConfirm={confirmDeleteAllChats}
        title="Delete All Chats"
        message="Are you sure you want to delete ALL chats? This action cannot be undone and will permanently remove all your conversations and uploaded files."
        confirmText="Delete All"
        cancelText="Cancel"
        type="danger"
      />

      <input
        type="file"
        ref={mainFileInputRef}
        accept=".pdf"
        multiple
        onChange={handleFileUpload}
        className="hidden"
      />
    </div>
  );
}

export default App; 