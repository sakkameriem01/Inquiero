import React, { useState, useEffect, useCallback, useRef } from 'react';
import PdfPreview from './PdfPreview';
import { toast } from 'react-hot-toast';

// API configuration
const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

const ChatHistory = ({ onSelectSession, onNewChat, currentSession }) => {
  const [sessions, setSessions] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [searchTerm, setSearchTerm] = useState('');
  const [selectedTag, setSelectedTag] = useState(null);
  const [editingSessionId, setEditingSessionId] = useState(null);
  const [editingTitle, setEditingTitle] = useState('');
  const [showPdfPreview, setShowPdfPreview] = useState(false);
  const [selectedPdf, setSelectedPdf] = useState(null);
  const fileInputRef = useRef(null);

  // Fetch sessions on component mount
  useEffect(() => {
    fetchSessions();
  }, []);

  const fetchSessions = async () => {
    try {
      setLoading(true);
      const response = await fetch(`${API_BASE_URL}/sessions/`);
      if (!response.ok) {
        throw new Error('Failed to fetch sessions');
      }
      const data = await response.json();
      setSessions(data);
      setError(null);
    } catch (error) {
      console.error('Error fetching sessions:', error);
      setError('Failed to load chat history');
      toast.error('Failed to load chat history');
    } finally {
      setLoading(false);
    }
  };

  const handleViewSession = async (session) => {
    try {
      // Show loading toast
      toast.loading('Loading chat session...', {
        id: 'loading-session',
        duration: 2000,
      });

      // Fetch full session details including messages
      const response = await fetch(`${API_BASE_URL}/sessions/${session.id}`);
      
      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.detail || `Failed to fetch session details: ${response.statusText}`);
      }

      const sessionData = await response.json();
      
      if (!sessionData) {
        throw new Error('No session data received');
      }

      // Call the parent's onSelectSession with the full session data
      onSelectSession(sessionData);

      // Show success toast
      toast.success('Chat session loaded', {
        id: 'loading-session',
      });
    } catch (error) {
      console.error("Error loading chat session:", error);
      toast.error(error.message || "Failed to load chat session", {
        id: 'loading-session',
      });
    }
  };

  const handleSaveSession = async (session) => {
    try {
      const response = await fetch(`${API_BASE_URL}/chat-sessions/${session.id}`, {
        method: 'PUT',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          pinned: !session.pinned
        }),
      });

      if (!response.ok) {
        throw new Error('Failed to update session');
      }

      // Update local state
      setSessions(prevSessions =>
        prevSessions.map(s =>
          s.id === session.id
            ? { ...s, pinned: !s.pinned }
            : s
        )
      );

      toast.success(session.pinned ? 'Chat unpinned' : 'Chat pinned');
    } catch (error) {
      console.error('Error updating session:', error);
      toast.error('Failed to update chat');
    }
  };

  const handleDeleteSession = async (sessionId) => {
    if (!window.confirm('Are you sure you want to delete this chat?')) {
      return;
    }

    try {
      const response = await fetch(`${API_BASE_URL}/chat-sessions/${sessionId}`, {
        method: 'DELETE',
      });

      if (!response.ok) {
        throw new Error('Failed to delete session');
      }

      // Update local state
      setSessions(prevSessions =>
        prevSessions.filter(s => s.id !== sessionId)
      );

      toast.success('Chat deleted');
    } catch (error) {
      console.error('Error deleting session:', error);
      toast.error('Failed to delete chat');
    }
  };

  const handleFileUpload = async (e) => {
    const file = e.target.files[0];
    if (!file) return;

    try {
      const formData = new FormData();
      formData.append('file', file);

      const response = await fetch(`${API_BASE_URL}/upload/`, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        throw new Error('Failed to upload file');
      }

      const data = await response.json();
      
      // Create a new session with the uploaded file
      const newSession = {
        id: Date.now().toString(),
        title: file.name,
        created_at: new Date().toISOString(),
        updated_at: new Date().toISOString(),
        files: [{
          id: data.file_id,
          filename: data.filename,
          original_filename: data.original_filename
        }]
      };

      setSessions(prev => [newSession, ...prev]);
      toast.success('File uploaded successfully');
    } catch (error) {
      console.error('Error uploading file:', error);
      toast.error('Failed to upload file');
    }
  };

  const handlePdfPreview = (filename) => {
    setSelectedPdf(filename);
    setShowPdfPreview(true);
  };

  const handleClosePdfPreview = () => {
    setShowPdfPreview(false);
    setSelectedPdf(null);
  };

  // Filter sessions based on search term and selected tag
  const filteredSessions = sessions.filter(session => {
    const matchesSearch = session.title.toLowerCase().includes(searchTerm.toLowerCase());
    const matchesTag = !selectedTag || session.tags?.includes(selectedTag);
    return matchesSearch && matchesTag;
  });

  // Get all unique tags from sessions
  const allTags = [...new Set(sessions.flatMap(session => session.tags || []))];

  return (
    <div className="h-full flex flex-col bg-white border-r border-gray-200">
      {/* Header */}
      <div className="p-6 border-b border-gray-200 bg-gradient-to-r from-blue-50 to-white">
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-xl font-semibold text-gray-800">Chat History</h2>
          <div className="flex space-x-2">
            {/* New Chat Button */}
            <button
              onClick={onNewChat}
              className="p-2 text-blue-600 hover:text-blue-700 rounded-lg hover:bg-blue-50 transition-all duration-200"
              title="New Chat"
            >
              <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M12 4v16m8-8H4" />
              </svg>
            </button>
          </div>
        </div>

        {/* Search */}
        <div className="relative">
          <input
            type="text"
            placeholder="Search chats..."
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
            className="w-full px-4 py-2 pl-10 border border-gray-200 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent transition-all duration-200"
          />
          <svg
            className="absolute left-3 top-2.5 h-5 w-5 text-gray-400"
            fill="none"
            stroke="currentColor"
            viewBox="0 0 24 24"
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth="2"
              d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z"
            />
          </svg>
        </div>
      </div>

      {/* Error Message */}
      {error && (
        <div className="p-4 bg-red-50 border-b border-red-200">
          <div className="flex items-center space-x-2">
            <svg className="w-5 h-5 text-red-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
            </svg>
            <p className="text-sm text-red-600">{error}</p>
          </div>
        </div>
      )}

      {/* Tags Filter */}
      {allTags.length > 0 && (
        <div className="p-4 border-b border-gray-200 bg-gray-50">
          <div className="flex flex-wrap gap-2">
            {allTags.map((tag) => (
              <button
                key={tag}
                onClick={() => setSelectedTag(selectedTag === tag ? null : tag)}
                className={`px-3 py-1.5 rounded-full text-sm font-medium transition-all duration-200 ${
                  selectedTag === tag
                    ? 'bg-blue-100 text-blue-700 shadow-sm'
                    : 'bg-white text-gray-700 hover:bg-gray-100'
                }`}
              >
                {tag}
              </button>
            ))}
          </div>
        </div>
      )}

      {/* Sessions List */}
      <div className="flex-1 overflow-y-auto p-4 space-y-3">
        {loading ? (
          <div className="flex justify-center items-center h-full">
            <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-500"></div>
          </div>
        ) : error ? (
          <div className="text-red-500 text-center p-4">{error}</div>
        ) : filteredSessions.length === 0 ? (
          <div className="text-center text-gray-500 py-8">
            <svg className="w-12 h-12 mx-auto mb-4 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="1.5" d="M20 13V6a2 2 0 00-2-2H6a2 2 0 00-2 2v7m16 0v5a2 2 0 01-2 2H6a2 2 0 01-2-2v-5m16 0h-2.586a1 1 0 00-.707.293l-2.414 2.414a1 1 0 01-.707.293h-3.172a1 1 0 01-.707-.293l-2.414-2.414A1 1 0 006.586 13H4" />
            </svg>
            <p className="text-lg font-medium">No chats found</p>
            <p className="text-sm">Start a new chat or upload a PDF to begin</p>
          </div>
        ) : (
          filteredSessions.map((session) => (
            <div
              key={session.id}
              className={`p-4 rounded-lg border transition-all duration-200 ${
                currentSession?.id === session.id
                  ? 'bg-blue-50 border-blue-200 shadow-sm'
                  : session.pinned
                  ? 'bg-yellow-50 border-yellow-200 border-l-4 shadow-sm'
                  : 'hover:bg-gray-50 border-gray-200'
              }`}
            >
              <div className="flex items-start justify-between">
                <div className="flex-1 min-w-0">
                  <div className="flex items-center space-x-2">
                    {session.pinned && (
                      <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4 text-yellow-500" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="1.5" d="M11.48 3.499a.562.562 0 011.04 0l2.125 5.111a.563.563 0 00.475.345l5.518.442c.499.04.701.663.321.988l-4.204 3.602a.563.563 0 00-.182.557l1.285 5.385a.562.562 0 01-.84.61l-4.725-2.885a.563.563 0 00-.586 0L6.982 20.54a.562.562 0 01-.84-.61l1.285-5.386a.562.562 0 00-.182-.557l-4.204-3.602a.563.563 0 01.321-.988l5.518-.442a.563.563 0 00.475-.345L11.48 3.5z" />
                      </svg>
                    )}
                    <h3 
                      className="text-sm font-medium text-gray-900 truncate cursor-pointer hover:text-blue-600 transition-colors"
                      onClick={() => handleViewSession(session)}
                      title={session.title}
                    >
                      {session.title}
                    </h3>
                  </div>
                  <p className="text-xs text-gray-500 mt-1">
                    {new Date(session.updated_at).toLocaleDateString()}
                  </p>
                </div>
                <div className="flex items-center space-x-1">
                  <button
                    onClick={() => handleViewSession(session)}
                    className="p-1.5 text-gray-400 hover:text-blue-500 transition-colors duration-200 rounded-full hover:bg-blue-50"
                    title="Open chat"
                  >
                    <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="1.5" d="M2.036 12.322a1.012 1.012 0 010-.639C3.423 7.51 7.36 4.5 12 4.5c4.638 0 8.573 3.007 9.963 7.178.07.207.07.431 0 .639C20.577 16.49 16.64 19.5 12 19.5c-4.638 0-8.573-3.007-9.963-7.178z" />
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="1.5" d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
                    </svg>
                  </button>
                  <button
                    onClick={() => handleSaveSession(session)}
                    className={`p-1.5 transition-colors duration-200 rounded-full ${
                      session.pinned
                        ? 'text-yellow-500 hover:bg-yellow-100'
                        : 'text-gray-400 hover:text-yellow-500 hover:bg-yellow-50'
                    }`}
                    title={session.pinned ? "Unfavorite chat" : "Favorite chat"}
                  >
                    <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4" fill={session.pinned ? "currentColor" : "none"} viewBox="0 0 24 24" stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="1.5" d="M11.48 3.499a.562.562 0 011.04 0l2.125 5.111a.563.563 0 00.475.345l5.518.442c.499.04.701.663.321.988l-4.204 3.602a.563.563 0 00-.182.557l1.285 5.385a.562.562 0 01-.84.61l-4.725-2.885a.563.563 0 00-.586 0L6.982 20.54a.562.562 0 01-.84-.61l1.285-5.386a.562.562 0 00-.182-.557l-4.204-3.602a.563.563 0 01.321-.988l5.518-.442a.563.563 0 00.475-.345L11.48 3.5z" />
                    </svg>
                  </button>
                  <button
                    onClick={() => handleDeleteSession(session.id)}
                    className="p-1.5 text-gray-400 hover:text-red-500 transition-colors duration-200 rounded-full hover:bg-red-50"
                    title="Delete chat"
                  >
                    <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="1.5" d="M14.74 9l-.346 9m-4.788 0L9.26 9m9.968-3.21c.342.052.682.107 1.022.166m-1.022-.165L18.16 19.673a2.25 2.25 0 01-2.244 2.077H8.084a2.25 2.25 0 01-2.244-2.077L4.772 5.79m14.456 0a48.108 48.108 0 00-3.478-.397m-12 .562c.34-.059.68-.114 1.022-.165m0 0a48.11 48.11 0 013.478-.397m7.5 0v-.916c0-1.18-.91-2.164-2.09-2.201a51.964 51.964 0 00-3.32 0c-1.18.037-2.09 1.022-2.09 2.201v.916m7.5 0a48.667 48.667 0 00-7.5 0" />
                    </svg>
                  </button>
                </div>
              </div>
              
              {session.files && session.files.length > 0 && (
                <div className="mt-3 pt-3 border-t border-gray-100">
                  <div className="flex items-center space-x-2 text-xs text-gray-500 mb-2">
                    <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="1.5" d="M7 21h10a2 2 0 002-2V9.414a1 1 0 00-.293-.707l-5.414-5.414A1 1 0 0012.586 3H7a2 2 0 00-2 2v14a2 2 0 002 2z" />
                    </svg>
                    <span>Attached Files</span>
                  </div>
                  <div className="space-y-1.5">
                    {session.files.map((file) => (
                      <div
                        key={file.id}
                        className="flex items-center space-x-2 text-sm text-gray-600 hover:text-blue-600 cursor-pointer group bg-gray-50 hover:bg-blue-50 px-2 py-1.5 rounded transition-colors duration-200"
                        onClick={() => handlePdfPreview(file.filename)}
                      >
                        <svg className="w-4 h-4 text-gray-400 group-hover:text-blue-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="1.5" d="M7 21h10a2 2 0 002-2V9.414a1 1 0 00-.293-.707l-5.414-5.414A1 1 0 0012.586 3H7a2 2 0 00-2 2v14a2 2 0 002 2z" />
                        </svg>
                        <span 
                          className="truncate max-w-[200px]" 
                          title={file.original_filename}
                        >
                          {file.original_filename}
                        </span>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          ))
        )}
      </div>

      {/* PDF Preview Modal */}
      {showPdfPreview && selectedPdf && (
        <PdfPreview
          filename={selectedPdf}
          onClose={handleClosePdfPreview}
        />
      )}
    </div>
  );
};

export default ChatHistory; 