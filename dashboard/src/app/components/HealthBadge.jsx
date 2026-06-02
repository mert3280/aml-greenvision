'use client'

import { useEffect, useState } from 'react'

const STATES = {
  checking: { dot: 'bg-gray-300',  label: 'Connecting…',  pulse: false },
  online:   { dot: 'bg-green-400', label: 'API Online',   pulse: true  },
  offline:  { dot: 'bg-red-400',   label: 'API Offline',  pulse: false },
}

export default function HealthBadge() {
  const [state, setState] = useState('checking')

  useEffect(() => {
    fetch('/api/health')
      .then(r => r.json())
      .then(data => setState(data.model_loaded ? 'online' : 'offline'))
      .catch(() => setState('offline'))
  }, [])

  const { dot, label, pulse } = STATES[state]

  return (
    <div className="flex items-center gap-2 font-body text-sm text-cream/70 select-none">
      <span
        className={`w-2.5 h-2.5 rounded-full ${dot} ${pulse ? 'animate-pulse' : ''}`}
        aria-hidden="true"
      />
      <span>{label}</span>
    </div>
  )
}
