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
        
        // If no active session, create a new one, else update the current session
        if (!currentSession) {
          setCurrentSession({
            id: result.session_id, // Use the session_id from the backend
            title: `Chat with ${file.name}`, // Set initial title
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
            }],
            messages: [] // Initialize messages for the new session
          });
        } else {
          // If there's an existing session, update its files array
          setCurrentSession(prevSession => ({
            ...prevSession,
            files: [...(prevSession?.files || []), {
              id: result.file_id,
              filename: result.filename,
              original_filename: result.original_filename,
              file_type: 'pdf',
              chunk_count: result.chunks,
              page_count: result.pages
            }]
          }));
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

  const handleSendMessage = async () => {
    if (!message.trim() || isProcessing) return;

    console.log('currentSession before check:', currentSession);
    console.log('hasUploadedFiles before check:', hasUploadedFiles);

    // Check if we have an active session and uploaded files
    if (!currentSession && !hasUploadedFiles) {
      setError('Please upload a PDF file first');
      return;
    }

    try {
      setIsProcessing(true);
      setError('');
      setIsTyping(true);

      // Add user message to the current messages
      const userMessage = { type: 'user', content: message.trim() };
      setMessages(prev => [...prev, userMessage]);
      setMessage('');

      // Prepare the request body
      const requestBody = {
        question: userMessage.content,
        session_id: currentSession?.id ? String(currentSession.id) : null,
      };

      // Only include files if it's a new session being created with uploaded files
      if (!currentSession && uploadedFiles.length > 0) {
        requestBody.files = uploadedFiles.map(file => ({
          id: String(file.id),
          filename: String(file.filename),
        }));
      }

      // Send the message to the backend
      const response = await fetch(`${API_BASE_URL}/chat/`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(requestBody),
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.detail || 'Failed to get response from server');
      }

      const data = await response.json();
      
      // Update the current session with the new messages
      if (data.session) {
        setCurrentSession(data.session);
        setMessages(data.session.messages);
      }

      // Add assistant's response to messages
      setMessages(prev => [...prev, { type: 'assistant', content: data.answer }]);
      
      // Scroll to bottom after message is added
      setTimeout(scrollToBottom, 100);

    } catch (error) {
      console.error('Error sending message:', error);
      setError('Failed to send message. Please try again.');
      // Remove the user message if the request failed
      setMessages(prev => prev.slice(0, -1));
    } finally {
      setIsProcessing(false);
      setIsTyping(false);
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
        <div className="bg-white border-b border-gray-200 p-4 flex items-center justify-between shadow-sm">
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
          <div className="w-1/4 p-4 border-r border-gray-200 bg-gray-50 flex flex-col">
            <h3 className="text-lg font-semibold text-gray-800 mb-4">Upload Documents</h3>
            <div className="flex-1 overflow-y-auto pr-2">
              {files.length > 0 && (
                <div className="mb-4 space-y-2">
                  {files.map((file, index) => (
                    <div
                      key={index}
                      className="flex items-center justify-between bg-white p-3 rounded-lg shadow-sm border border-gray-200"
                    >
                      <span className="text-sm font-medium text-gray-700 truncate">
                        {file.name}
                      </span>
                      <button
                        onClick={() => removeFile(index)}
                        className="text-gray-400 hover:text-red-500 ml-2"
                      >
                        <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M6 18L18 6M6 6l12 12" />
                        </svg>
                      </button>
                    </div>
                  ))}
                </div>
              )}
              {uploadedFiles.length > 0 && (
                <div className="mt-4 pt-4 border-t border-gray-200">
                  <h4 className="text-md font-semibold text-gray-700 mb-3">Uploaded Files</h4>
                  <ul className="space-y-2">
                    {uploadedFiles.map((file) => (
                      <li key={file.id} className="flex items-center text-sm text-gray-600">
                        <svg className="w-4 h-4 mr-2 text-green-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
                        </svg>
                        {file.filename} ({file.pages} pages)
                      </li>
                    ))}
                  </ul>
                </div>
              )}
            </div>
            <div className="mt-4 border-t border-gray-200 pt-4">
              <label
                htmlFor="file-upload"
                className="w-full flex items-center justify-center px-4 py-2 border border-gray-300 rounded-lg shadow-sm text-sm font-medium text-gray-700 bg-white hover:bg-gray-50 cursor-pointer transition-colors duration-200"
              >
                <svg className="w-5 h-5 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-8l-4-4m0 0L8 8m4-4v12" />
                </svg>
                Add PDF
              </label>
              <input
                id="file-upload"
                ref={fileInputRef}
                type="file"
                multiple
                accept=".pdf"
                onChange={handleFileChange}
                className="hidden"
              />
              <button
                onClick={handleFileUpload}
                disabled={files.length === 0 || loading}
                className="w-full mt-3 px-4 py-2 border border-transparent text-sm font-medium rounded-lg shadow-sm text-white bg-blue-600 hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500 transition-colors duration-200 disabled:opacity-50 disabled:cursor-not-allowed"
              >
                {loading ? 'Uploading...' : 'Upload Selected'}
              </button>
              {error && <p className="text-red-500 text-xs mt-2">{error}</p>}
            </div>
          </div>

          {/* Chat Area */}
          <div className="flex-1 flex flex-col bg-white">
            <div className="flex-1 overflow-y-auto p-6 space-y-4" ref={messagesEndRef}>
              {messages.map((msg, index) => (
                <div
                  key={index}
                  className={`flex ${msg.type === 'user' ? 'justify-end' : 'justify-start'}`}
                >
                  <div
                    className={`max-w-[70%] px-4 py-3 rounded-lg shadow-md ${
                      msg.type === 'user'
                        ? 'bg-blue-500 text-white'
                        : 'bg-gray-100 text-gray-800'
                    }`}
                  >
                    <p className="text-sm">{msg.content}</p>
                  </div>
                </div>
              ))}
              {isTyping && (
                <div className="flex justify-start">
                  <div className="bg-gray-100 px-4 py-3 rounded-lg shadow-md">
                    <div className="flex space-x-1">
                      <span className="w-2 h-2 bg-gray-500 rounded-full animate-bounce" style={{ animationDelay: '0s' }}></span>
                      <span className="w-2 h-2 bg-gray-500 rounded-full animate-bounce" style={{ animationDelay: '.1s' }}></span>
                      <span className="w-2 h-2 bg-gray-500 rounded-full animate-bounce" style={{ animationDelay: '.2s' }}></span>
                    </div>
                  </div>
                </div>
              )}
            </div>

            {/* Message Input */}
            <div className="border-t border-gray-200 p-4 bg-white">
              <form onSubmit={(e) => {
                e.preventDefault();
                handleSendMessage();
              }} className="relative flex items-center">
                <textarea
                  value={message}
                  onChange={(e) => setMessage(e.target.value)}
                  onKeyDown={(e) => {
                    if (e.key === 'Enter' && !e.shiftKey) {
                      e.preventDefault(); // Prevent default to avoid new line AND form submission
                      handleSendMessage();
                    }
                  }}
                  placeholder="Type your message..."
                  className="flex-1 resize-none overflow-hidden pr-10 px-4 py-3 rounded-lg border border-gray-300 focus:outline-none focus:ring-2 focus:ring-blue-500 transition-shadow duration-200"
                  rows="1"
                  style={{ minHeight: '50px', maxHeight: '150px' }}
                ></textarea>
                <button
                  type="submit" // Set type to submit for form submission
                  disabled={isProcessing || !message.trim()}
                  className="absolute right-3 p-2 bg-blue-600 text-white rounded-full hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500 transition-colors duration-200 disabled:opacity-50 disabled:cursor-not-allowed"
                >
                  <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth="2">
                    <path strokeLinecap="round" strokeLinejoin="round" d="M14 5l7 7m0 0l-7 7m7-7H3" />
                  </svg>
                </button>
              </form>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

export default App; 