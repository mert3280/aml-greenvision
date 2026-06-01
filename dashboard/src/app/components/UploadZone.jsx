'use client'

import { useCallback, useRef, useState } from 'react'
import Image from 'next/image'

export default function UploadZone({ onImageSelect, preview, isLoading, onAnalyze }) {
  const [isDragOver, setIsDragOver] = useState(false)
  const inputRef = useRef(null)

  const handleFile = useCallback((file) => {
    if (!file || !file.type.startsWith('image/')) return
    const url = URL.createObjectURL(file)
    onImageSelect(file, url)
  }, [onImageSelect])

  const handleDrop = useCallback((e) => {
    e.preventDefault()
    setIsDragOver(false)
    handleFile(e.dataTransfer.files[0])
  }, [handleFile])

  const handleDragOver = (e) => { e.preventDefault(); setIsDragOver(true) }
  const handleDragLeave = () => setIsDragOver(false)
  const handleInputChange = (e) => handleFile(e.target.files[0])

  return (
    <div className="flex flex-col gap-4">
      {/* Drop zone */}
      <div
        onDrop={handleDrop}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onClick={() => !preview && inputRef.current?.click()}
        className={[
          'relative rounded-2xl border-2 border-dashed transition-all duration-200 overflow-hidden',
          'min-h-[260px] flex items-center justify-center',
          isDragOver
            ? 'border-sage bg-seafoam/70 scale-[1.01]'
            : preview
              ? 'border-mint/40 bg-white/60'
              : 'border-mint/50 bg-white/50 cursor-pointer hover:border-sage hover:bg-seafoam/30',
        ].join(' ')}
      >
        <input
          ref={inputRef}
          type="file"
          accept="image/*"
          onChange={handleInputChange}
          className="hidden"
          aria-label="Upload leaf image"
        />

        {preview ? (
          /* eslint-disable-next-line @next/next/no-img-element */
          <img
            src={preview}
            alt="Uploaded leaf"
            className="w-full h-full object-cover max-h-72"
          />
        ) : (
          <div className="text-center p-10 select-none">
            <div className="text-6xl mb-4" aria-hidden="true">🌿</div>
            <p className="font-display text-forest text-xl font-semibold">
              Drop your leaf photo here
            </p>
            <p className="font-body text-earth/55 text-sm mt-1.5">or click to browse files</p>
            <p className="font-body text-earth/35 text-xs mt-4">
              JPEG · PNG · WebP · HEIC supported
            </p>
          </div>
        )}

        {/* Drag overlay */}
        {isDragOver && (
          <div className="absolute inset-0 flex items-center justify-center bg-seafoam/80 rounded-2xl">
            <p className="font-display text-forest text-2xl font-bold">Drop to upload</p>
          </div>
        )}
      </div>

      {/* Action buttons */}
      {preview && (
        <div className="flex gap-3">
          <button
            onClick={() => inputRef.current?.click()}
            disabled={isLoading}
            className="px-5 py-2.5 rounded-xl border border-sage/40 text-sage font-body text-sm font-medium
                       hover:bg-seafoam/40 transition-colors duration-150 disabled:opacity-40"
          >
            Change Photo
          </button>

          <button
            onClick={onAnalyze}
            disabled={isLoading}
            className={[
              'flex-1 flex items-center justify-center gap-2 py-2.5 px-6 rounded-xl',
              'font-body font-semibold text-sm transition-all duration-200 shadow-md',
              isLoading
                ? 'bg-sage/60 text-white cursor-not-allowed'
                : 'bg-sage text-white hover:bg-forest hover:shadow-lg',
            ].join(' ')}
          >
            {isLoading ? (
              <>
                <span
                  className="w-4 h-4 border-2 border-white/40 border-t-white rounded-full animate-spinner"
                  aria-hidden="true"
                />
                Analyzing…
              </>
            ) : (
              'Analyze Plant'
            )}
          </button>
        </div>
      )}
    </div>
  )
}
