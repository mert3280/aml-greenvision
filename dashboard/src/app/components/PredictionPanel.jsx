'use client'

import ConfidenceBar from './ConfidenceBar'
import AnalyticsPanel from './AnalyticsPanel'
import LowConfidencePrompt from './LowConfidencePrompt'

const CONFIDENCE_THRESHOLD = 0.50

function formatClass(name) {
  if (name === 'Background_without_leaves') return 'Background (no leaf)'
  const parts = name.split('___')
  if (parts.length === 1) return name.replace(/_/g, ' ')
  const [crop, condition] = parts
  return `${crop.replace(/_/g, ' ')} — ${condition.replace(/_/g, ' ')}`
}

function getStatusMeta(className) {
  if (className === 'Background_without_leaves') {
    return {
      emoji:       '🍃',
      badge:       'No Plant Detected',
      badgeCls:    'bg-gray-100 text-gray-500',
      barColor:    'grey',
    }
  }
  if (className.includes('healthy')) {
    return {
      emoji:    '✅',
      badge:    'Healthy Plant',
      badgeCls: 'bg-amber-50 text-amber-700 border border-amber-200',
      barColor: 'gold',
    }
  }
  return {
    emoji:    '⚠️',
    badge:    'Disease Detected',
    badgeCls: 'bg-red-50 text-alert border border-red-200',
    barColor: 'alert',
  }
}

export default function PredictionPanel({ result, onReset }) {
  if (!result) return null

  if (result.confidence < CONFIDENCE_THRESHOLD) {
    return <LowConfidencePrompt topGuesses={(result.top_k || []).slice(0, 3)} onReset={onReset} />
  }

  const { emoji, badge, badgeCls, barColor } = getStatusMeta(result.class)
  const label = formatClass(result.class)

  return (
    <div className="bg-white/90 border border-seafoam rounded-2xl p-6 shadow-sm animate-fade-slide-in">
      {/* Badge row */}
      <div className="flex items-center justify-between mb-3">
        <span className="text-2xl select-none" aria-hidden="true">{emoji}</span>
        <span className={`font-body text-xs font-semibold px-3 py-1 rounded-full ${badgeCls}`}>
          {badge}
        </span>
      </div>

      {/* Class name */}
      <h3 className="font-display text-forest text-2xl font-bold leading-tight mb-5">
        {label}
      </h3>

      {/* Confidence bar */}
      <ConfidenceBar confidence={result.confidence} color={barColor} showLabel={true} />

      {/* Analytics accordion */}
      <AnalyticsPanel topK={result.top_k} />
    </div>
  )
}
