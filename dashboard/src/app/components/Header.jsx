import Link from 'next/link'
import HealthBadge from './HealthBadge'

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

        {/* Nav + API status */}
        <div className="flex items-center gap-6">
          <nav className="hidden sm:flex items-center gap-4">
            <Link
              href="/"
              className="font-body text-sm text-cream/70 hover:text-cream transition-colors"
            >
              Predict
            </Link>
            <Link
              href="/augmentation"
              className="font-body text-sm text-cream/70 hover:text-cream transition-colors"
            >
              Augmentation
            </Link>
          </nav>
          <HealthBadge />
        </div>
      </div>
    </header>
  )
}
