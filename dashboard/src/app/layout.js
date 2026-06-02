import { Playfair_Display, Inter } from 'next/font/google'
import './globals.css'

const playfair = Playfair_Display({
  subsets: ['latin'],
  variable: '--font-playfair',
  display: 'swap',
})

const inter = Inter({
  subsets: ['latin'],
  variable: '--font-inter',
  display: 'swap',
})

export const metadata = {
  title: 'GreenVision — Plant Disease Classifier',
  description:
    'AI-powered plant disease identification. Upload a leaf photo and get an instant diagnosis powered by EfficientNet-B0.',
}

export default function RootLayout({ children }) {
  return (
    <html lang="en" className={`${playfair.variable} ${inter.variable}`}>
      <body className="font-body min-h-screen bg-cream text-earth antialiased">
        {children}
      </body>
    </html>
  )
}
