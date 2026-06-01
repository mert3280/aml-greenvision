import HealthBadge from './HealthBadge'
import Image from 'next/image'

export default function Header() {
  return (
    <header className="bg-forest shadow-lg">
      <div className="max-w-5xl mx-auto px-6 py-4 flex items-center justify-between">
        {/* Logo + wordmark */}
        <div className="flex items-center gap-3">
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img
            src="/logo.svg"
            alt=""
            aria-hidden="true"
            className="w-9 h-11 drop-shadow-md"
          />
          <div>
            <span className="font-display text-cream text-2xl font-bold tracking-tight">
              GreenVision
            </span>
            <p className="font-body text-mint/75 text-xs leading-none mt-0.5">
              Plant Disease Classifier
            </p>
          </div>
        </div>

        {/* API status */}
        <HealthBadge />
      </div>
    </header>
  )
}
