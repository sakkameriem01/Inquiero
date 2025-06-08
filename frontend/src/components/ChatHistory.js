import React, { useState, useEffect, useCallback } from 'react';

// API configuration
const API_BASE_URL = 'http://localhost:8000';

const ChatHistory = ({ onSelectSession, onNewChat }) => {
  const [sessions, setSessions] = useState([]);
  const [searchTerm, setSearchTerm] = useState('');
  const [selectedTag, setSelectedTag] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const [editingTitle, setEditingTitle] = useState(null);
  const [editingTags, setEditingTags] = useState(null);
  const [previewPdf, setPreviewPdf] = useState(null);

  const fetchSessions = useCallback(async () => {
    try {
      setLoading(true);
      setError('');
      const response = await fetch(`${API_BASE_URL}/sessions/`);
      
      if (!response.ok) {
        throw new Error('Failed to fetch sessions');
      }

      const data = await response.json();
      setSessions(data);
    } catch (err) {
      console.error('Error fetching sessions:', err);
      setError('Failed to load chat history');
      setSessions([]);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchSessions();
  }, [fetchSessions]);

  const handleNewChat = () => {
    onNewChat();
    fetchSessions(); // Refresh sessions after new chat
  };

  const handleSelectSession = (session) => {
    onSelectSession(session);
  };

  const handleDeleteSession = async (sessionId) => {
    try {
      setError('');
      const response = await fetch(`${API_BASE_URL}/chat-sessions/${sessionId}`, {
        method: 'DELETE',
      });
      
      if (!response.ok) {
        const errorText = await response.text();
        throw new Error(`Failed to delete session: ${response.status} ${errorText}`);
      }
      
      await fetchSessions();
    } catch (err) {
      console.error('Error deleting session:', err);
      setError(err.message || 'Failed to delete chat session');
    }
  };

  const handleUpdateSession = async (sessionId, updates) => {
    try {
      setError('');
      const response = await fetch(`${API_BASE_URL}/chat-sessions/${sessionId}`, {
        method: 'PUT',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(updates),
      });
      
      if (!response.ok) {
        const errorText = await response.text();
        throw new Error(`Failed to update session: ${response.status} ${errorText}`);
      }
      
      await fetchSessions();
    } catch (err) {
      console.error('Error updating session:', err);
      setError(err.message || 'Failed to update session');
    }
  };

  const handleUpdateTags = async (sessionId, tags) => {
    await handleUpdateSession(sessionId, { tags });
    setEditingTags(null);
  };

  const handleTogglePin = async (sessionId, currentPinned) => {
    await handleUpdateSession(sessionId, { pinned: !currentPinned });
  };

  const handleTitleEdit = async (sessionId, newTitle) => {
    if (!newTitle) return;
    await handleUpdateSession(sessionId, { title: newTitle });
    setEditingTitle(null);
  };

  const handlePreviewPDF = (filename) => {
    const encodedFilename = encodeURIComponent(filename);
    setPreviewPdf(`${API_BASE_URL}/uploads/${encodedFilename}`);
  };

  const closePreview = () => {
    setPreviewPdf(null);
  };

  const allTags = Array.from(new Set(sessions.flatMap(session => session.tags)));

  if (loading) {
    return (
      <div className="h-full bg-white p-4">
        <div className="animate-pulse space-y-4">
          {[1, 2, 3].map((i) => (
            <div key={i} className="h-20 bg-gray-200 rounded-lg"></div>
          ))}
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="h-full bg-white p-4">
        <div className="text-red-500">Error loading chat history: {error}</div>
      </div>
    );
  }

  return (
    <div className="h-full flex flex-col bg-white border-r border-gray-200">
      {/* Header */}
      <div className="p-4 border-b border-gray-200">
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-lg font-semibold text-gray-900">Chat History</h2>
          <button
            onClick={handleNewChat}
            className="p-2 text-blue-600 hover:text-blue-700 rounded-lg hover:bg-blue-50"
          >
            <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M12 4v16m8-8H4" />
            </svg>
          </button>
        </div>
        
        {/* Search */}
        <div className="relative">
          <input
            type="text"
            placeholder="Search chats..."
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
            className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
          />
          <svg
            className="absolute right-3 top-2.5 h-5 w-5 text-gray-400"
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
          <p className="text-sm text-red-600">{error}</p>
        </div>
      )}

      {/* Tags Filter */}
      <div className="p-4 border-b border-gray-200">
        <div className="flex flex-wrap gap-2">
          {allTags.map((tag) => (
            <button
              key={tag}
              onClick={() => setSelectedTag(selectedTag === tag ? null : tag)}
              className={`px-3 py-1 rounded-full text-sm font-medium transition-colors ${
                selectedTag === tag
                  ? 'bg-blue-100 text-blue-700'
                  : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
              }`}
            >
              {tag}
            </button>
          ))}
        </div>
      </div>

      {/* Chat Sessions List */}
      <div className="flex-1 overflow-y-auto p-4 space-y-4">
        {sessions.map((session) => (
          <div
            key={session.id}
            className={`bg-white rounded-lg border ${
              session.pinned ? 'border-yellow-200 bg-yellow-50' : 'border-gray-200'
            } p-4 hover:shadow-md transition-shadow`}
          >
            <div className="flex items-start justify-between">
              <div className="flex-1">
                {editingTitle === session.id ? (
                  <input
                    type="text"
                    defaultValue={session.title}
                    className="w-full px-2 py-1 border border-gray-300 rounded focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                    onBlur={(e) => handleTitleEdit(session.id, e.target.value)}
                    onKeyDown={(e) => {
                      if (e.key === 'Enter') {
                        handleTitleEdit(session.id, e.target.value);
                      } else if (e.key === 'Escape') {
                        setEditingTitle(null);
                      }
                    }}
                    autoFocus
                  />
                ) : (
                  <div className="flex items-center gap-2">
                    <h3 
                      className="font-medium text-gray-900 mb-2 cursor-pointer hover:text-blue-600"
                      onClick={() => setEditingTitle(session.id)}
                    >
                      {session.title}
                    </h3>
                    {session.pinned && (
                      <svg className="w-4 h-4 text-yellow-500" fill="currentColor" viewBox="0 0 20 20">
                        <path d="M9.049 2.927c.3-.921 1.603-.921 1.902 0l1.07 3.292a1 1 0 00.95.69h3.462c.969 0 1.371 1.24.588 1.81l-2.8 2.034a1 1 0 00-.364 1.118l1.07 3.292c.3.921-.755 1.688-1.54 1.118l-2.8-2.034a1 1 0 00-1.175 0l-2.8 2.034c-.784.57-1.838-.197-1.539-1.118l1.07-3.292a1 1 0 00-.364-1.118L2.98 8.72c-.783-.57-.38-1.81.588-1.81h3.461a1 1 0 00.951-.69l1.07-3.292z" />
                      </svg>
                    )}
                  </div>
                )}
                <div className="flex flex-wrap gap-2 mb-2">
                  {editingTags === session.id ? (
                    <div className="flex items-center gap-2">
                      <input
                        type="text"
                        placeholder="Add tags (comma-separated)"
                        className="px-2 py-1 border border-gray-300 rounded focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                        onBlur={(e) => {
                          const tags = e.target.value.split(',').map(tag => tag.trim()).filter(Boolean);
                          handleUpdateTags(session.id, tags);
                        }}
                        onKeyDown={(e) => {
                          if (e.key === 'Enter') {
                            const tags = e.target.value.split(',').map(tag => tag.trim()).filter(Boolean);
                            handleUpdateTags(session.id, tags);
                          } else if (e.key === 'Escape') {
                            setEditingTags(null);
                          }
                        }}
                        defaultValue={session.tags.join(', ')}
                        autoFocus
                      />
                    </div>
                  ) : (
                    <>
                      {session.tags.map((tag) => (
                        <span
                          key={tag}
                          className="px-2 py-1 bg-gray-100 text-gray-700 rounded-full text-xs"
                        >
                          #{tag}
                        </span>
                      ))}
                      <button
                        onClick={() => setEditingTags(session.id)}
                        className="px-2 py-1 bg-gray-100 text-gray-700 rounded-full text-xs hover:bg-gray-200"
                      >
                        Edit Tags
                      </button>
                    </>
                  )}
                </div>
                <p className="text-sm text-gray-500">
                  {new Date(session.updated_at).toLocaleDateString()}
                </p>
              </div>
              <div className="flex space-x-2">
                <button
                  onClick={(e) => {
                    e.stopPropagation();
                    handleTogglePin(session.id, session.pinned);
                  }}
                  className={`p-1 rounded-lg hover:bg-yellow-50 ${
                    session.pinned ? 'text-yellow-500' : 'text-gray-400 hover:text-yellow-500'
                  }`}
                >
                  <svg className="w-5 h-5" fill="currentColor" viewBox="0 0 20 20">
                    <path d="M9.049 2.927c.3-.921 1.603-.921 1.902 0l1.07 3.292a1 1 0 00.95.69h3.462c.969 0 1.371 1.24.588 1.81l-2.8 2.034a1 1 0 00-.364 1.118l1.07 3.292c.3.921-.755 1.688-1.54 1.118l-2.8-2.034a1 1 0 00-1.175 0l-2.8 2.034c-.784.57-1.838-.197-1.539-1.118l1.07-3.292a1 1 0 00-.364-1.118L2.98 8.72c-.783-.57-.38-1.81.588-1.81h3.461a1 1 0 00.951-.69l1.07-3.292z" />
                  </svg>
                </button>
                <button
                  onClick={(e) => {
                    e.stopPropagation();
                    handleSelectSession(session);
                  }}
                  className="p-1 text-blue-600 hover:text-blue-700 rounded-lg hover:bg-blue-50"
                >
                  <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      strokeWidth="2"
                      d="M15 12a3 3 0 11-6 0 3 3 0 016 0z"
                    />
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      strokeWidth="2"
                      d="M2.458 12C3.732 7.943 7.523 5 12 5c4.478 0 8.268 2.943 9.542 7-1.274 4.057-5.064 7-9.542 7-4.477 0-8.268-2.943-9.542-7z"
                    />
                  </svg>
                </button>
                <button
                  onClick={(e) => {
                    e.stopPropagation();
                    handleDeleteSession(session.id);
                  }}
                  className="p-1 text-red-600 hover:text-red-700 rounded-lg hover:bg-red-50"
                >
                  <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      strokeWidth="2"
                      d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16"
                    />
                  </svg>
                </button>
              </div>
            </div>

            {/* Files */}
            {session.files && session.files.length > 0 && (
              <div className="mt-2 space-y-1">
                {session.files.map((file, index) => (
                  <div
                    key={index}
                    onClick={(e) => {
                      e.stopPropagation();
                      handlePreviewPDF(file);
                    }}
                    className="flex items-center space-x-2 text-sm text-gray-600 hover:text-blue-600 cursor-pointer group"
                  >
                    <span className="text-lg">📄</span>
                    <span className="truncate flex-1">{file}</span>
                  </div>
                ))}
              </div>
            )}
          </div>
        ))}
      </div>

      {/* PDF Preview Modal */}
      {previewPdf && (
        <div className="fixed inset-0 bg-black bg-opacity-50 z-50 flex items-center justify-center p-4">
          <div className="bg-white rounded-lg shadow-xl w-full max-w-4xl h-[80vh] flex flex-col">
            <div className="flex items-center justify-between p-4 border-b">
              <h3 className="text-lg font-semibold text-gray-900">PDF Preview</h3>
              <button
                onClick={closePreview}
                className="text-gray-400 hover:text-gray-600 focus:outline-none"
              >
                <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M6 18L18 6M6 6l12 12" />
                </svg>
              </button>
            </div>
            <div className="flex-1 overflow-hidden">
              <iframe
                src={previewPdf}
                className="w-full h-full"
                title="PDF Preview"
              />
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default ChatHistory; 