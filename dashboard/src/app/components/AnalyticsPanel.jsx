'use client'

import { useState } from 'react'
import ConfidenceBar from './ConfidenceBar'

const MODEL_STATS = [
  { label: 'Top-1 Accuracy', value: '98.54%' },
  { label: 'Top-5 Accuracy', value: '99.98%' },
  { label: 'Plant Classes',  value: '39'      },
  { label: 'Macro F1 Score', value: '98.05%' },
]

function formatClass(name) {
  if (name === 'Background_without_leaves') return 'Background (no leaf)'
  const parts = name.split('___')
  if (parts.length === 1) return name.replace(/_/g, ' ')
  const [crop, condition] = parts
  return `${crop.replace(/_/g, ' ')} — ${condition.replace(/_/g, ' ')}`
}

export default function AnalyticsPanel({ topK }) {
  const [open, setOpen] = useState(false)

  return (
    <div className="mt-5 pt-4 border-t border-seafoam">
      <button
        onClick={() => setOpen(o => !o)}
        className="flex items-center gap-1.5 font-body text-sm font-medium text-sage
                   hover:text-forest transition-colors duration-150 group"
        aria-expanded={open}
      >
        <span>View Analytics</span>
        <span
          className={`text-xs transition-transform duration-200 ${open ? 'rotate-180' : ''}`}
          aria-hidden="true"
        >
          ▾
        </span>
      </button>

      {open && (
        <div className="mt-4 space-y-5 animate-fade-slide-in">
          {/* Top-5 predictions */}
          <div>
            <h4 className="font-body text-xs font-semibold uppercase tracking-wider text-earth/40 mb-3">
              Top 5 Predictions
            </h4>
            <div className="space-y-3">
              {topK.map((item, i) => (
                <div key={i}>
                  <div className="flex justify-between items-baseline mb-1">
                    <span className={`font-body text-sm ${i === 0 ? 'font-semibold text-forest' : 'text-earth/70'}`}>
                      {formatClass(item.class)}
                    </span>
                    <span className="font-body text-xs text-earth/50 tabular-nums ml-2 shrink-0">
                      {(item.confidence * 100).toFixed(1)}%
                    </span>
                  </div>
                  <ConfidenceBar
                    confidence={item.confidence}
                    color={i === 0 ? 'sage' : 'mint'}
                    showLabel={false}
                  />
                </div>
              ))}
            </div>
          </div>

          {/* Model performance stats */}
          <div>
            <h4 className="font-body text-xs font-semibold uppercase tracking-wider text-earth/40 mb-3">
              Model Performance
            </h4>
            <div className="grid grid-cols-2 gap-2.5">
              {MODEL_STATS.map(({ label, value }) => (
                <div key={label} className="bg-seafoam/60 rounded-xl p-3 text-center">
                  <p className="font-display text-forest text-2xl font-bold leading-none">{value}</p>
                  <p className="font-body text-earth/55 text-xs mt-1.5">{label}</p>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
