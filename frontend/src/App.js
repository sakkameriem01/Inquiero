import React, { useState, useRef, useEffect } from 'react';
import ChatHistory from './components/ChatHistory';

// API configuration
const API_BASE_URL = 'http://localhost:8000';

function App() {
  const [files, setFiles] = useState([]);
  const [uploadedFiles, setUploadedFiles] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [messages, setMessages] = useState([]);
  const [message, setMessage] = useState('');
  const [isProcessing, setIsProcessing] = useState(false);
  const [currentSession, setCurrentSession] = useState(null);
  const [showHistory, setShowHistory] = useState(true);
  const [isTyping, setIsTyping] = useState(false);
  const [showToast, setShowToast] = useState(true);
  const [showSuccessToast, setShowSuccessToast] = useState(false);
  const [successMessage, setSuccessMessage] = useState('');
  const [hasUploadedFiles, setHasUploadedFiles] = useState(false);
  const messagesEndRef = useRef(null);
  const fileInputRef = useRef(null);

  // Show welcome toast on component mount
  useEffect(() => {
    const timer = setTimeout(() => {
      setShowToast(false);
    }, 5000);

    return () => clearTimeout(timer);
  }, []);

  // Handle success toast auto-hide
  useEffect(() => {
    if (showSuccessToast) {
      const timer = setTimeout(() => {
        setShowSuccessToast(false);
      }, 5000);

      return () => clearTimeout(timer);
    }
  }, [showSuccessToast]);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const handleNewChat = () => {
    setCurrentSession(null);
    setMessages([]);
    setHasUploadedFiles(false);
    setUploadedFiles([]);
    // Clear the backend's processed files
    fetch(`${API_BASE_URL}/files/`, {
      method: 'DELETE',
    }).catch(console.error);
  };

  const handleSelectSession = (session) => {
    try {
      setCurrentSession(session);
      // Ensure messages is always an array
      setMessages(session.messages || []);
      // Check if the session has files
      setHasUploadedFiles(session.files && session.files.length > 0);
      // Scroll to bottom of messages
      setTimeout(scrollToBottom, 100);
    } catch (error) {
      console.error("Error selecting session:", error);
      setError("Failed to open chat session");
    }
  };

  const handleFileChange = (e) => {
    const selectedFiles = Array.from(e.target.files);
    const validFiles = selectedFiles.filter(file => file.type === 'application/pdf');
    
    if (validFiles.length > 0) {
      setFiles(prevFiles => [...prevFiles, ...validFiles]);
      setError('');
    } else {
      setError('Please select valid PDF files');
    }
  };

  const handleFileUpload = async () => {
    if (files.length === 0) {
      setError('Please select a file first');
      return;
    }

    try {
      setLoading(true);
      setError('');

      for (const file of files) {
        const formData = new FormData();
        formData.append('file', file);
        
        // If there's an active session, include its ID
        if (currentSession) {
          formData.append('session_id', currentSession.id);
        }

        const response = await fetch(`${API_BASE_URL}/upload/`, {
          method: 'POST',
          body: formData,
        });

        if (!response.ok) {
          const error = await response.json();
          throw new Error(error.detail || 'Failed to upload file');
        }

        const result = await response.json();
        
        // Add the uploaded file to the uploadedFiles state
        setUploadedFiles(prev => [...prev, {
          id: result.file_id,
          filename: result.original_filename,
          chunks: result.chunks,
          pages: result.pages,
          uploadedAt: new Date().toISOString()
        }]);
        
        // If no active session, create a new one
        if (!currentSession) {
          const newSession = {
            title: file.name,
            created_at: new Date().toISOString(),
            updated_at: new Date().toISOString(),
            pinned: false,
            files: [{
              id: result.file_id,
              filename: result.filename,
              original_filename: result.original_filename,
              file_type: 'pdf',
              chunk_count: result.chunks,
              page_count: result.pages
            }]
          };
          setCurrentSession(newSession);
        }
      }

      // Clear the files array and input
      setFiles([]);
      if (fileInputRef.current) {
        fileInputRef.current.value = '';
      }
      
      // Set hasUploadedFiles to true after successful upload
      setHasUploadedFiles(true);
      
      // Show success message
      setSuccessMessage('Files uploaded successfully');
      setShowSuccessToast(true);
    } catch (error) {
      console.error('Error uploading file:', error);
      setError(error.message || 'Failed to upload file');
    } finally {
      setLoading(false);
    }
  };

  const handleSendMessage = async (e) => {
    e.preventDefault();
    if (!message.trim()) return;

    // Check if files are uploaded
    if (!hasUploadedFiles) {
      setMessages(prev => [...prev, { 
        type: 'error', 
        content: 'Please upload a PDF file first to start chatting.' 
      }]);
      return;
    }

    const userMessage = message;
    setMessage('');
    setMessages(prev => [...prev, { type: 'user', content: userMessage }]);
    setIsProcessing(true);
    setIsTyping(true);

    try {
      const response = await fetch(`${API_BASE_URL}/chat/`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Accept': 'application/json'
        },
        body: JSON.stringify({
          question: userMessage,
          session_id: currentSession?.id,
          tags: currentSession?.tags || []
        }),
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.detail || data.message || 'Failed to get answer');
      }
      
      // Simulate typing delay
      setTimeout(() => {
        setIsTyping(false);
        // Add assistant's response
        setMessages(prev => [...prev, { 
          type: 'assistant', 
          content: data.answer || 'Sorry, I could not generate a response.'
        }]);

        // Only update session if we don't have one
        if (!currentSession && data.session) {
          setCurrentSession(data.session);
        }
      }, 1000);

    } catch (error) {
      console.error('Error:', error);
      setIsTyping(false);
      setMessages(prev => [...prev, { 
        type: 'error', 
        content: error.message || 'Sorry, I encountered an error. Please try again.' 
      }]);
    } finally {
      setIsProcessing(false);
    }
  };

  const removeFile = (index) => {
    setFiles(prevFiles => prevFiles.filter((_, i) => i !== index));
  };

  return (
    <div className="flex h-screen bg-gray-100">
      {/* Welcome Toast */}
      {showToast && (
        <div className="fixed top-4 right-4 z-50 animate-slide-in">
          <div className="bg-white rounded-lg shadow-xl p-4 border border-blue-100 flex items-center space-x-3 max-w-md transform transition-all duration-300 hover:scale-105">
            <div className="flex-shrink-0 bg-blue-100 p-2 rounded-full">
              <svg className="w-6 h-6 text-blue-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
              </svg>
            </div>
            <div className="flex-1">
              <p className="text-gray-800 font-medium">Welcome to Inquiero!</p>
              <p className="text-gray-600 text-sm mt-1">Upload your PDF file(s) to start chatting with your documents.</p>
            </div>
            <button
              onClick={() => setShowToast(false)}
              className="flex-shrink-0 text-gray-400 hover:text-gray-600 transition-colors duration-200"
            >
              <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M6 18L18 6M6 6l12 12" />
              </svg>
            </button>
          </div>
        </div>
      )}

      {/* Success Toast */}
      {showSuccessToast && (
        <div className="fixed top-4 right-4 z-50 animate-slide-in">
          <div className="bg-white rounded-lg shadow-xl p-4 border border-green-100 flex items-center space-x-3 max-w-md transform transition-all duration-300 hover:scale-105">
            <div className="flex-shrink-0 bg-green-100 p-2 rounded-full">
              <svg className="w-6 h-6 text-green-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
              </svg>
            </div>
            <div className="flex-1">
              <p className="text-gray-800 font-medium">Success!</p>
              <p className="text-gray-600 text-sm mt-1">{successMessage}</p>
            </div>
            <button
              onClick={() => setShowSuccessToast(false)}
              className="flex-shrink-0 text-gray-400 hover:text-gray-600 transition-colors duration-200"
            >
              <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M6 18L18 6M6 6l12 12" />
              </svg>
            </button>
          </div>
        </div>
      )}

      {/* Chat History Sidebar */}
      {showHistory && (
        <div className="w-80">
          <ChatHistory
            onSelectSession={handleSelectSession}
            onNewChat={handleNewChat}
            currentSession={currentSession}
          />
        </div>
      )}

      {/* Main Chat Area */}
      <div className="flex-1 flex flex-col">
        {/* Header */}
        <div className="bg-white shadow-sm p-4 flex items-center justify-between">
          <div className="flex items-center space-x-4">
            <button
              onClick={() => setShowHistory(!showHistory)}
              className="p-2 hover:bg-gray-100 rounded-lg"
            >
              <svg className="w-6 h-6 text-gray-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M4 6h16M4 12h16M4 18h16" />
              </svg>
            </button>
            <h1 className="text-xl font-semibold text-gray-900">
              {currentSession?.title || "New Chat"}
            </h1>
          </div>
        </div>

        {/* Main Content */}
        <div className="flex-1 flex">
          {/* File Upload Panel */}
          <div className="w-80 border-r bg-white p-4">
            <h2 className="text-lg font-semibold text-gray-900 mb-4">Upload PDFs</h2>
            <div className="space-y-4">
              <div className="flex items-center justify-center w-full">
                <label className="flex flex-col items-center justify-center w-full h-40 border-2 border-gray-300 border-dashed rounded-xl cursor-pointer bg-gray-50 hover:bg-gray-100 transition-all duration-200 hover:border-blue-400 group">
                  <div className="flex flex-col items-center justify-center pt-5 pb-6">
                    <svg className="w-12 h-12 mb-4 text-gray-400 group-hover:text-blue-500 transition-colors duration-200" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" />
                    </svg>
                    <p className="mb-2 text-sm text-gray-500 group-hover:text-gray-700">
                      <span className="font-semibold">Click to upload</span> or drag and drop
                    </p>
                    <p className="text-xs text-gray-500">PDF files only</p>
                  </div>
                  <input
                    ref={fileInputRef}
                    type="file"
                    accept=".pdf"
                    onChange={handleFileChange}
                    className="hidden"
                  />
                </label>
              </div>
              {/* Separate Upload Button */}
              <div className="mt-4 flex justify-center">
                <button
                  type="button"
                  onClick={handleFileUpload}
                  disabled={files.length === 0 || loading}
                  className="px-6 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 transition-colors duration-200 flex items-center space-x-2 disabled:bg-gray-400 disabled:cursor-not-allowed"
                >
                  {loading ? (
                    <>
                      <svg className="animate-spin h-5 w-5 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                        <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                        <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                      </svg>
                      <span>Uploading...</span>
                    </>
                  ) : (
                    <>
                      <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" />
                      </svg>
                      <span>Upload</span>
                    </>
                  )}
                </button>
              </div>
              {/* Selected Files */}
              {files && files.length > 0 && (
                <div className="space-y-2">
                  {files.map((file, index) => (
                    <div key={index} className="flex items-center justify-between p-3 bg-gray-50 rounded-lg border border-gray-200">
                      <div className="flex items-center space-x-3">
                        <svg className="w-5 h-5 text-blue-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M7 21h10a2 2 0 002-2V9.414a1 1 0 00-.293-.707l-5.414-5.414A1 1 0 0012.586 3H7a2 2 0 00-2 2v14a2 2 0 002 2z" />
                        </svg>
                        <span className="text-sm text-gray-600 truncate">{file.name}</span>
                      </div>
                      <button
                        onClick={() => removeFile(index)}
                        className="text-gray-400 hover:text-gray-600 transition-colors duration-200"
                      >
                        <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M6 18L18 6M6 6l12 12" />
                        </svg>
                      </button>
                    </div>
                  ))}
                </div>
              )}
              {/* Uploaded Files */}
              {uploadedFiles && uploadedFiles.length > 0 && (
                <div className="mt-6">
                  <h3 className="text-sm font-medium text-gray-700 mb-3">Uploaded Files</h3>
                  <div className="space-y-2">
                    {uploadedFiles.map((file) => (
                      <div key={file.id} className="flex items-center justify-between p-3 bg-gray-50 rounded-lg border border-gray-200">
                        <div className="flex items-center space-x-3">
                          <svg className="w-5 h-5 text-blue-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M7 21h10a2 2 0 002-2V9.414a1 1 0 00-.293-.707l-5.414-5.414A1 1 0 0012.586 3H7a2 2 0 00-2 2v14a2 2 0 002 2z" />
                          </svg>
                          <div className="flex flex-col">
                            <span className="text-sm text-gray-600 truncate">{file.filename}</span>
                            <span className="text-xs text-gray-400">
                              {file.pages} pages • {file.chunks} chunks
                            </span>
                          </div>
                        </div>
                        <button
                          onClick={() => {
                            setUploadedFiles(prev => prev.filter(f => f.id !== file.id));
                            // You might want to add an API call here to delete the file from the backend
                          }}
                          className="text-gray-400 hover:text-gray-600 transition-colors duration-200"
                        >
                          <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M6 18L18 6M6 6l12 12" />
                          </svg>
                        </button>
                      </div>
                    ))}
                  </div>
                </div>
              )}
              {error && (
                <div className="text-red-500 text-sm mt-2 bg-red-50 p-3 rounded-lg border border-red-200">
                  {error}
                </div>
              )}
            </div>
          </div>

          {/* Chat Panel */}
          <div className="flex-1 flex flex-col">
            {/* Messages */}
            <div className="flex-1 overflow-y-auto p-4 space-y-4">
              {messages && messages.length > 0 && messages.map((msg, index) => (
                <div
                  key={index}
                  className={`flex ${
                    msg.type === "user" ? "justify-end" : "justify-start"
                  }`}
                >
                  <div
                    className={`max-w-3xl rounded-lg p-4 ${
                      msg.type === "user"
                        ? "bg-blue-600 text-white"
                        : msg.type === "error"
                        ? "bg-red-100 text-red-700"
                        : msg.type === "system"
                        ? "bg-green-50 text-green-700 border border-green-200"
                        : "bg-white shadow-sm"
                    }`}
                  >
                    <div className="whitespace-pre-wrap">{msg.content}</div>
                  </div>
                </div>
              ))}
              {messages && messages.length === 0 && (
                <div className="flex justify-center items-center h-full text-gray-500">
                  No messages yet. Start the conversation!
                </div>
              )}
              {isTyping && (
                <div className="flex justify-start">
                  <div className="bg-white rounded-lg p-4 shadow-sm">
                    <div className="flex space-x-2">
                      <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: '0ms' }}></div>
                      <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: '150ms' }}></div>
                      <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: '300ms' }}></div>
                    </div>
                  </div>
                </div>
              )}
              <div ref={messagesEndRef} />
            </div>

            {/* Input */}
            <form onSubmit={handleSendMessage} className="bg-white border-t p-4">
              <div className="flex space-x-4">
                <input
                  type="text"
                  value={message}
                  onChange={(e) => setMessage(e.target.value)}
                  placeholder="Type your message..."
                  className="flex-1 border rounded-lg px-4 py-2 focus:outline-none focus:ring-2 focus:ring-blue-500"
                  disabled={isProcessing}
                />
                <button
                  type="submit"
                  disabled={isProcessing}
                  className="bg-blue-600 text-white px-6 py-2 rounded-lg hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-blue-500 disabled:bg-gray-400 disabled:cursor-not-allowed"
                >
                  Send
                </button>
              </div>
            </form>
          </div>
        </div>
      </div>
    </div>
  );
}

export default App; 