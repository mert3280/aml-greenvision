'use client'

const TIPS = [
  'Get closer — fill the frame with a single leaf so it dominates the shot',
  'Shoot in natural daylight and avoid harsh shadows or direct sun glare',
  'Hold the camera steady; prop your elbow against something to reduce blur',
  'Photograph the top side of the leaf, not the underside',
  'Wipe your camera lens clean with a soft cloth before shooting',
]

export default function LowConfidencePrompt({ onReset }) {
  return (
    <div className="bg-white/90 border border-amber-200 rounded-2xl p-6 shadow-sm animate-fade-slide-in">
      <div className="flex items-start gap-4">
        <span className="text-4xl select-none" aria-hidden="true">📷</span>

        <div className="flex-1 min-w-0">
          <h3 className="font-display text-forest text-xl font-bold mb-1">
            We need a clearer photo
          </h3>
          <p className="font-body text-earth/70 text-sm leading-relaxed mb-5">
            Our model isn&apos;t confident enough to make a diagnosis with this image.
            Try the tips below for a better result:
          </p>

          <ul className="space-y-2.5 mb-6">
            {TIPS.map((tip, i) => (
              <li key={i} className="flex items-start gap-2.5 font-body text-sm text-earth">
                <span className="text-sage font-bold mt-0.5 shrink-0">✓</span>
                <span>{tip}</span>
              </li>
            ))}
          </ul>

          <button
            onClick={onReset}
            className="bg-sage text-white py-2.5 px-7 rounded-xl font-body font-semibold text-sm
                       hover:bg-forest transition-colors duration-200 shadow-md hover:shadow-lg"
          >
            Try Another Photo
          </button>
        </div>
      </div>
    </div>
  )
}
