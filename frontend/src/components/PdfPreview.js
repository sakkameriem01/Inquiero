import React from 'react';

const PdfPreview = ({ filename, onClose }) => {
  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 z-50 flex items-center justify-center p-4">
      <div className="bg-white rounded-lg shadow-xl w-full max-w-4xl h-[80vh] flex flex-col">
        <div className="flex items-center justify-between p-4 border-b">
          <div className="flex-1 min-w-0">
            <h3 
              className="text-lg font-semibold text-gray-900 truncate pr-4" 
              title={filename}
            >
              {filename}
            </h3>
          </div>
          <button
            onClick={onClose}
            className="p-2 text-gray-400 hover:text-gray-600 transition-colors duration-200 flex-shrink-0"
          >
            <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M6 18L18 6M6 6l12 12" />
            </svg>
          </button>
        </div>
        <div className="flex-1 overflow-hidden">
          <iframe
            src={`http://localhost:8000/uploads/${filename}`}
            className="w-full h-full"
            title="PDF Preview"
          />
        </div>
      </div>
    </div>
  );
};

export default PdfPreview; 