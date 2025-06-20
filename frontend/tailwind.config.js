/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    "./src/**/*.{js,jsx,ts,tsx}",
  ],
  darkMode: 'class',
  theme: {
    extend: {
      colors: {
        background: {
          light: '#f8f9fa',
          dark: '#1e1e2f',
        },
        text: {
          primary: {
            light: '#1c1c1c',
            dark: '#f5f5f5',
          },
          secondary: '#6c757d',
        },
        accent: {
          blue: '#4e6cff',
          purple: '#a569bd',
        },
        chat: {
          user: '#e6e9ef',
          ai: {
            light: '#dce0f7',
            dark: '#2d3250',
          },
        },
        sidebar: {
          light: '#f4f4f4',
          dark: '#2a2a3b',
        },
      },
      fontFamily: {
        sans: ['Inter', 'Roboto', 'system-ui', 'sans-serif'],
      },
      boxShadow: {
        'soft': '0 2px 4px rgba(0, 0, 0, 0.05)',
      },
      borderRadius: {
        'xl': '1rem',
      },
    },
  },
  plugins: [],
} 