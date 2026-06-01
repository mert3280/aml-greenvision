/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    './src/pages/**/*.{js,ts,jsx,tsx,mdx}',
    './src/components/**/*.{js,ts,jsx,tsx,mdx}',
    './src/app/**/*.{js,ts,jsx,tsx,mdx}',
  ],
  theme: {
    extend: {
      colors: {
        forest: '#1B4332',
        sage:   '#40916C',
        mint:   '#74C69D',
        seafoam:'#D8F3DC',
        cream:  '#F7F6EE',
        earth:  '#5C4033',
        gold:   '#D4A843',
        alert:  '#C1440E',
      },
      fontFamily: {
        display: ['var(--font-playfair)', 'Georgia', 'serif'],
        body:    ['var(--font-inter)', 'system-ui', 'sans-serif'],
      },
      keyframes: {
        fadeSlideIn: {
          '0%':   { opacity: '0', transform: 'translateY(-6px)' },
          '100%': { opacity: '1', transform: 'translateY(0)' },
        },
        spinnerSpin: {
          to: { transform: 'rotate(360deg)' },
        },
      },
      animation: {
        'fade-slide-in': 'fadeSlideIn 0.2s ease-out',
        'spinner':       'spinnerSpin 0.8s linear infinite',
      },
    },
  },
  plugins: [],
}
