'use client'

import { useCallback, useState } from 'react'
import Header from './components/Header'
import UploadZone from './components/UploadZone'
import PredictionPanel from './components/PredictionPanel'

export default function Dashboard() {
  const [file, setFile]       = useState(null)
  const [preview, setPreview] = useState(null)
  const [result, setResult]   = useState(null)
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError]     = useState(null)

  const handleImageSelect = useCallback((selectedFile, previewUrl) => {
    setFile(selectedFile)
    setPreview(previewUrl)
    setResult(null)
    setError(null)
  }, [])

  const handleAnalyze = useCallback(async () => {
    if (!file) return

    setIsLoading(true)
    setError(null)
    setResult(null)

    try {
      const body = new FormData()
      body.append('file', file)
      body.append('top_k', '5')

      const res = await fetch('/api/predict', { method: 'POST', body })

      if (!res.ok) {
        const json = await res.json().catch(() => ({}))
        throw new Error(json.detail || `Server error ${res.status}`)
      }

      setResult(await res.json())
    } catch (err) {
      setError(err.message)
    } finally {
      setIsLoading(false)
    }
  }, [file])

  const handleReset = useCallback(() => {
    setFile(null)
    setPreview(null)
    setResult(null)
    setError(null)
  }, [])

  return (
    <div className="min-h-screen flex flex-col bg-cream">
      <Header />

      {/* Hero banner */}
      <section className="hero-bg py-14 px-6 text-center">
        <h2 className="font-display text-forest text-4xl sm:text-5xl font-bold leading-tight mb-3">
          Identify Plant Diseases
          <br className="hidden sm:block" />
          <span className="text-sage"> Instantly</span>
        </h2>
        <p className="font-body text-earth/65 text-lg max-w-lg mx-auto leading-relaxed">
          Upload a clear leaf photo and our AI will diagnose it in seconds —
          trained on 39 plant conditions with 98.5% accuracy.
        </p>
      </section>

      {/* Main workspace */}
      <main className="flex-1 max-w-5xl w-full mx-auto px-6 py-10">
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6 items-start">
          {/* Left: upload */}
          <UploadZone
            onImageSelect={handleImageSelect}
            preview={preview}
            isLoading={isLoading}
            onAnalyze={handleAnalyze}
          />

          {/* Right: results */}
          <div className="min-h-[260px]">
            {error && (
              <div className="bg-red-50 border border-red-200 rounded-2xl p-5 font-body text-sm text-red-700 animate-fade-slide-in">
                <strong className="font-semibold">Something went wrong:</strong> {error}
              </div>
            )}

            {result && !error && (
              <PredictionPanel result={result} onReset={handleReset} />
            )}

            {!result && !error && (
              <div className="h-full min-h-[260px] flex flex-col items-center justify-center
                              text-center rounded-2xl border-2 border-dashed border-seafoam bg-white/30">
                <span className="text-7xl mb-4 opacity-20 select-none" aria-hidden="true">🌱</span>
                <p className="font-body text-earth/35 text-sm leading-relaxed">
                  Upload a leaf photo<br />to see your diagnosis here
                </p>
              </div>
            )}
          </div>
        </div>
      </main>

      {/* Footer */}
      <footer className="py-6 border-t border-seafoam text-center">
        <p className="font-body text-xs text-earth/35">
          GreenVision &middot; EfficientNet-B0 &middot; 39 plant conditions &middot; 98.54% top-1 accuracy
        </p>
      </footer>
    </div>
  )
}
