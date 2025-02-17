import './globals.css'
import { Inter } from 'next/font/google'

// Initialize the Inter font
const inter = Inter({ subsets: ['latin'] })

export const metadata = {
  title: 'SET Game Solver',
  description: 'AI-powered SET card game solver with Claude',
  keywords: ['SET game', 'card game', 'AI', 'Claude', 'puzzle solver'],
  authors: [{ name: 'Your Name' }],
  viewport: 'width=device-width, initial-scale=1',
  themeColor: '#1F2937', // Matches your dark theme
}

export default function RootLayout({ children }) {
  return (
    <html lang="en" className={inter.className}>
      <body className="min-h-screen bg-gradient-to-br from-gray-900 to-gray-800 antialiased">
        {/* Main content wrapper */}
        <div className="relative">
          {/* Background pattern (optional) */}
          <div className="absolute inset-0 bg-[radial-gradient(#374151_1px,transparent_1px)] [background-size:16px_16px] opacity-50 pointer-events-none" />
          
          {/* Content */}
          <div className="relative">
            {children}
          </div>
        </div>
      </body>
    </html>
  )
}