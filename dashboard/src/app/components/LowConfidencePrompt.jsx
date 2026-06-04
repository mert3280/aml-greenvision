'use client'

import ConfidenceBar from './ConfidenceBar'

function formatClass(name) {
  if (name === 'Background_without_leaves') return 'Background (no leaf)'
  const parts = name.split('___')
  if (parts.length === 1) return name.replace(/_/g, ' ')
  const [crop, condition] = parts
  return `${crop.replace(/_/g, ' ')} — ${condition.replace(/_/g, ' ')}`
}

export default function LowConfidencePrompt({ topGuesses, onReset }) {
  return (
    <div className="bg-white/90 border border-amber-200 rounded-2xl p-6 shadow-sm animate-fade-slide-in">
      <div className="flex items-start gap-3 mb-4">
        <span className="text-3xl select-none" aria-hidden="true">⚠️</span>
        <div>
          <h3 className="font-display text-forest text-xl font-bold leading-tight">
            Low confidence result
          </h3>
          <p className="font-body text-earth/70 text-sm leading-relaxed mt-1">
            The model isn&apos;t confident enough to give a single diagnosis. It could
            be <strong className="text-forest">any of the following</strong> — try a
            clearer photo for a definitive answer.
          </p>
        </div>
      </div>

      <ol className="space-y-3 mb-6">
        {topGuesses.map((guess, i) => (
          <li key={guess.class_index} className="bg-amber-50/60 border border-amber-100 rounded-xl p-3">
            <div className="flex items-center justify-between mb-1.5">
              <span className="font-body text-sm font-semibold text-forest">
                {i + 1}. {formatClass(guess.class)}
              </span>
              <span className="font-body text-xs text-earth/60 ml-2 shrink-0">
                {(guess.confidence * 100).toFixed(1)}%
              </span>
            </div>
            <ConfidenceBar confidence={guess.confidence} color="gold" showLabel={false} />
          </li>
        ))}
      </ol>

      <div className="bg-amber-50 border border-amber-300 rounded-xl p-4 mb-5 flex gap-3 items-start">
        <span className="text-amber-500 text-lg select-none shrink-0 mt-0.5" aria-hidden="true">🔬</span>
        <p className="font-body text-sm text-amber-800 leading-relaxed">
          <strong>Further research is required before acting.</strong> This prediction is
          uncertain and should not be used as the sole basis for treatment or intervention
          decisions. Consult an agronomist or plant pathologist to confirm any diagnosis.
        </p>
      </div>

      <button
        onClick={onReset}
        className="bg-sage text-white py-2.5 px-7 rounded-xl font-body font-semibold text-sm
                   hover:bg-forest transition-colors duration-200 shadow-md hover:shadow-lg"
      >
        Try Another Photo
      </button>
    </div>
  )
}
