'use client'

const COLOR_MAP = {
  sage:  'bg-sage',
  mint:  'bg-mint',
  gold:  'bg-gold',
  alert: 'bg-alert',
  grey:  'bg-gray-400',
}

export default function ConfidenceBar({ confidence, color = 'sage', showLabel = true }) {
  const pct = Math.round(confidence * 100)

  return (
    <div className="w-full">
      {showLabel && (
        <div className="flex justify-between items-center mb-1">
          <span className="font-body text-xs text-earth/50 uppercase tracking-wide">Confidence</span>
          <span className="font-body text-sm font-semibold text-earth tabular-nums">{pct}%</span>
        </div>
      )}
      <div className="w-full h-3 bg-seafoam rounded-full overflow-hidden">
        <div
          className={`h-full rounded-full bar-grow ${COLOR_MAP[color] ?? 'bg-sage'}`}
          style={{ width: `${pct}%` }}
        />
      </div>
    </div>
  )
}
