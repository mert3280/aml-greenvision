'use client'

import { useCallback, useRef, useState } from 'react'
import Header from '../components/Header'

// ─── Pipeline metadata ──────────────────────────────────────────────────────

const TAG_STYLES = {
  domain:   'bg-forest/10 text-forest',
  geometry: 'bg-sage/15 text-sage',
  color:    'bg-gold/20 text-earth',
  noise:    'bg-earth/10 text-earth',
  required: 'bg-seafoam text-forest',
}

const TRAIN_STEPS = [
  {
    name: 'RandomBackground',
    params: 'p=0.9, mask_dir=data/masks/',
    desc: 'Load a pre-computed BiRefNet segmentation mask for the image, then composite the isolated leaf onto one of 8,638 random outdoor landscape photos. Falls back to brightness-threshold heuristic if no cached mask exists.',
    why: 'BiRefNet produces pixel-accurate leaf cutouts regardless of background color (white, black, gray, or complex). Masks are precomputed once so training reads them at zero inference overhead. The previous heuristic failed on gray studio backgrounds, pasting them over the landscape unchanged.',
    tag: 'domain',
  },
  {
    name: 'RandomResizedCrop',
    params: 'size=224, scale 50–100%',
    desc: 'Crop a random 50–100% patch of the image and resize it to 224×224.',
    why: 'Simulates partial leaf views and varying camera distances.',
    tag: 'geometry',
  },
  {
    name: 'RandomHorizontalFlip',
    params: 'p=0.5',
    desc: 'Mirror the image left-to-right with 50% probability.',
    why: 'Leaves look the same flipped; doubles effective training data.',
    tag: 'geometry',
  },
  {
    name: 'RandomVerticalFlip',
    params: 'p=0.2',
    desc: 'Mirror the image top-to-bottom with 20% probability.',
    why: 'Handles upside-down or overhead-angle shots.',
    tag: 'geometry',
  },
  {
    name: 'RandomRotation',
    params: '±45°',
    desc: 'Rotate by a random angle up to ±45°.',
    why: 'Field photos are captured at many orientations; the model should be rotation-invariant.',
    tag: 'geometry',
  },
  {
    name: 'ColorJitter',
    params: 'brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1',
    desc: 'Randomly adjust brightness, contrast, saturation, and hue.',
    why: 'Simulates varying lighting conditions, different cameras, and time-of-day.',
    tag: 'color',
  },
  {
    name: 'RandomPerspective',
    params: 'distortion_scale=0.4, p=0.5',
    desc: 'Apply random perspective warp with 50% probability.',
    why: 'Smartphones are rarely held perfectly flat above a leaf; models tilt, creating perspective distortion.',
    tag: 'geometry',
  },
  {
    name: 'RandomGrayscale',
    params: 'p=0.1',
    desc: 'Convert to grayscale with 10% probability.',
    why: 'Handles edge cases: monochrome sensors, printed scans, or very low-saturation images.',
    tag: 'color',
  },
  {
    name: 'GaussianBlur',
    params: 'kernel=5, σ∈[0.1, 2.0]',
    desc: 'Apply Gaussian blur with a randomly sampled sigma.',
    why: 'Simulates camera focus issues, motion blur, or low-resolution sensors.',
    tag: 'noise',
  },
  {
    name: 'Normalize',
    params: 'μ=[0.485,0.456,0.406], σ=[0.229,0.224,0.225]',
    desc: 'Convert to float tensor and normalize using ImageNet channel statistics.',
    why: 'EfficientNet-B0 was pretrained on ImageNet — these statistics are locked and must never be recomputed from the dataset.',
    tag: 'required',
  },
  {
    name: 'RandomErasing',
    params: 'p=0.4, area 2–25%, ratio 0.3–3.3',
    desc: 'Erase a random rectangular patch (filled with noise) with 40% probability.',
    why: 'Simulates occlusion by branches, dirt, fingers, or other objects covering part of the leaf.',
    tag: 'noise',
  },
]

const EVAL_STEPS = [
  {
    name: 'Resize(256)',
    params: 'shortest edge → 256px',
    desc: 'Resize the shortest edge to 256px while preserving the aspect ratio.',
    why: 'Standardizes input size for the subsequent crop without squashing the image.',
    tag: 'required',
  },
  {
    name: 'CenterCrop(224)',
    params: '224×224px',
    desc: 'Crop the center 224×224 pixels.',
    why: 'Produces a deterministic, consistent crop at the model\'s expected input resolution.',
    tag: 'required',
  },
  {
    name: 'Normalize',
    params: 'μ=[0.485,0.456,0.406], σ=[0.229,0.224,0.225]',
    desc: 'Convert to float tensor and normalize using ImageNet channel statistics.',
    why: 'Must exactly match the training normalization — the model has never seen unnormalized inputs.',
    tag: 'required',
  },
]

// ─── Background-removal strategy data ───────────────────────────────────────

const BIREFNET_HIGHLIGHTS = [
  {
    label: 'Neural segmentation',
    detail: 'BiRefNet uses a bilateral reference network trained on salient-object datasets — it understands leaf shapes, not just pixel brightness.',
    icon: '🧠',
    cardClass: 'bg-forest/5 border-forest/25',
  },
  {
    label: 'Any background color',
    detail: 'White, black, gray, gradient — BiRefNet segments the foreground object regardless of background color. No thresholds to tune.',
    icon: '🎨',
    cardClass: 'bg-sage/10 border-sage/30',
  },
  {
    label: 'Zero training overhead',
    detail: 'Masks are precomputed once (~3 hrs for 54k images on GPU) and cached as PNGs. Training reads them instantly — no model inference per batch.',
    icon: '⚡',
    cardClass: 'bg-gold/10 border-gold/30',
  },
]

const BG_REMOVAL_PHASES = [
  {
    num: '1',
    title: 'Precompute masks (run once)',
    body: 'A preprocessing script runs BiRefNet (ZhengPeng7/BiRefNet via HuggingFace) on every PlantVillage image at 1024×1024, producing a binary foreground mask. Masks are saved as PNG files mirroring the dataset folder structure under data/masks/. Supports --resume to skip already-computed masks.',
    tag: 'offline',
  },
  {
    num: '2',
    title: 'Load cached mask at training time',
    body: 'RandomBackground.__call__ resolves the source image path, looks up the corresponding mask PNG, and loads it in a single file read. If no cached mask exists (e.g. the script hasn\'t been run yet), it falls back to the heuristic brightness-threshold segmentation so training is never blocked.',
    tag: 'runtime',
    callout: 'Fallback: the heuristic white/black threshold remains as a safety net while masks are being precomputed.',
  },
  {
    num: '3',
    title: 'Composite onto a random landscape',
    body: 'The mask is resized to match the source image, then Image.composite() pastes the leaf over one of 8,638 outdoor landscape photos (randomly scaled and cropped). 15% of the time a procedural background — solid colour or Gaussian noise — is used instead.',
    tag: 'runtime',
  },
]

// ─── Sub-components ─────────────────────────────────────────────────────────

function TagPill({ tag }) {
  return (
    <span className={`font-body text-[10px] font-semibold px-2 py-0.5 rounded-full ${TAG_STYLES[tag] ?? 'bg-gray-100 text-gray-500'}`}>
      {tag}
    </span>
  )
}

function StepRow({ index, step, accent }) {
  const [open, setOpen] = useState(false)
  return (
    <div className="border-b border-seafoam last:border-0 py-3">
      <button
        className="w-full text-left flex items-start gap-3 group"
        onClick={() => setOpen(o => !o)}
        aria-expanded={open}
      >
        <span
          className={`mt-0.5 flex-shrink-0 w-6 h-6 rounded-full flex items-center justify-center
                      font-body text-xs font-bold text-white ${accent}`}
        >
          {index}
        </span>
        <div className="flex-1 min-w-0">
          <div className="flex flex-wrap items-center gap-2">
            <span className="font-body text-sm font-semibold text-earth">{step.name}</span>
            <TagPill tag={step.tag} />
          </div>
          <p className="font-body text-xs text-earth/55 mt-0.5 truncate">{step.params}</p>
        </div>
        <span className="font-body text-earth/30 text-xs mt-1 flex-shrink-0 group-hover:text-earth/60 transition-colors">
          {open ? '▲' : '▼'}
        </span>
      </button>
      {open && (
        <div className="mt-2 ml-9 space-y-1 animate-fade-slide-in">
          <p className="font-body text-xs text-earth/75 leading-relaxed">{step.desc}</p>
          <p className="font-body text-xs text-sage italic leading-relaxed">
            <span className="font-semibold not-italic text-earth/60">Why: </span>
            {step.why}
          </p>
        </div>
      )}
    </div>
  )
}

function PipelineCard({ title, subtitle, steps, accent, borderColor, headerBg }) {
  return (
    <div className={`rounded-2xl border ${borderColor} bg-white/80 shadow-sm flex flex-col`}>
      <div className={`${headerBg} rounded-t-2xl px-5 py-4`}>
        <h3 className="font-display text-forest text-lg font-bold">{title}</h3>
        <p className="font-body text-xs text-earth/60 mt-0.5">{subtitle}</p>
      </div>
      <div className="px-5 py-2 flex-1">
        {steps.map((step, i) => (
          <StepRow key={step.name} index={i + 1} step={step} accent={accent} />
        ))}
      </div>
      <div className="px-5 py-3 border-t border-seafoam">
        <p className="font-body text-xs text-earth/40">{steps.length} transforms</p>
      </div>
    </div>
  )
}

const TAG_BG = {
  offline: 'bg-gold/20 text-earth',
  runtime: 'bg-sage/15 text-sage',
}

function BgRemovalSection() {
  return (
    <section>
      <div className="mb-5">
        <div className="flex flex-wrap items-center gap-2 mb-1">
          <h3 className="font-display text-forest text-2xl font-bold">Background Removal Strategy</h3>
          <span className="font-body text-[11px] font-semibold px-2 py-0.5 rounded-full bg-forest text-cream">BiRefNet</span>
        </div>
        <p className="font-body text-earth/55 text-sm">
          <code className="font-mono text-xs bg-seafoam px-1.5 py-0.5 rounded">RandomBackground</code> uses{' '}
          <strong className="text-earth/80">BiRefNet</strong> — a state-of-the-art bilateral reference network —
          to produce pixel-accurate leaf cutouts. Unlike the previous brightness-threshold heuristic,
          BiRefNet segments by learned object shape, handling white, black, and gray studio backgrounds
          equally well. Masks are precomputed once and cached; training reads them at zero inference overhead.
        </p>
      </div>

      {/* BiRefNet highlights */}
      <div className="grid grid-cols-1 sm:grid-cols-3 gap-3 mb-7">
        {BIREFNET_HIGHLIGHTS.map(h => (
          <div key={h.label} className={`rounded-xl border p-4 ${h.cardClass}`}>
            <div className="flex items-center gap-2 mb-2">
              <span className="text-xl" aria-hidden="true">{h.icon}</span>
              <p className="font-body text-sm font-bold text-earth">{h.label}</p>
            </div>
            <p className="font-body text-xs text-earth/60 leading-relaxed">{h.detail}</p>
          </div>
        ))}
      </div>

      {/* 3-phase pipeline */}
      <p className="font-body text-[11px] font-semibold text-earth/40 uppercase tracking-wider mb-3">
        Pipeline
      </p>
      <div className="space-y-3">
        {BG_REMOVAL_PHASES.map(phase => (
          <div key={phase.num} className="bg-white/80 border border-seafoam rounded-2xl p-5 flex gap-4">
            <div className="flex-shrink-0 w-8 h-8 rounded-full bg-sage flex items-center justify-center font-body text-sm font-bold text-cream">
              {phase.num}
            </div>
            <div className="min-w-0">
              <div className="flex flex-wrap items-center gap-2 mb-1">
                <p className="font-body text-sm font-semibold text-earth">{phase.title}</p>
                {phase.tag && (
                  <span className={`font-body text-[10px] font-semibold px-2 py-0.5 rounded-full ${TAG_BG[phase.tag] ?? ''}`}>
                    {phase.tag}
                  </span>
                )}
              </div>
              <p className="font-body text-xs text-earth/65 leading-relaxed">{phase.body}</p>
              {phase.callout && (
                <p className="font-body text-xs text-sage italic mt-2 leading-relaxed">
                  <span className="font-semibold not-italic text-earth/60">Fallback: </span>
                  {phase.callout}
                </p>
              )}
            </div>
          </div>
        ))}
      </div>

      {/* Model reference */}
      <div className="mt-4 bg-seafoam/40 rounded-xl border border-seafoam px-4 py-3 flex items-start gap-3">
        <span className="font-mono text-xs text-earth/50 mt-0.5 flex-shrink-0">HF</span>
        <div>
          <p className="font-mono text-xs text-forest font-semibold">ZhengPeng7/BiRefNet</p>
          <p className="font-body text-[11px] text-earth/50 mt-0.5">
            1024×1024 input · binary foreground mask output · CPU or CUDA · ~200 ms/image on GPU
          </p>
        </div>
      </div>
    </section>
  )
}

function DropZone({ onFile, preview, fileName }) {
  const inputRef = useRef(null)

  const handleDrop = useCallback((e) => {
    e.preventDefault()
    const f = e.dataTransfer.files?.[0]
    if (f && f.type.startsWith('image/')) onFile(f)
  }, [onFile])

  const handleChange = useCallback((e) => {
    const f = e.target.files?.[0]
    if (f) onFile(f)
  }, [onFile])

  return (
    <div
      className="relative rounded-2xl border-2 border-dashed border-seafoam bg-white/40
                  hover:border-mint hover:bg-white/60 transition-colors cursor-pointer
                  flex flex-col items-center justify-center min-h-[160px] p-4"
      onDrop={handleDrop}
      onDragOver={e => e.preventDefault()}
      onClick={() => inputRef.current?.click()}
    >
      <input ref={inputRef} type="file" accept="image/*" className="hidden" onChange={handleChange} />
      {preview ? (
        <div className="flex items-center gap-4 w-full">
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img src={preview} alt="Selected" className="w-20 h-20 rounded-xl object-cover shadow-sm flex-shrink-0" />
          <div className="min-w-0">
            <p className="font-body text-sm font-semibold text-earth truncate">{fileName}</p>
            <p className="font-body text-xs text-earth/50 mt-0.5">Click or drop to replace</p>
          </div>
        </div>
      ) : (
        <>
          <span className="text-4xl mb-2 opacity-25 select-none" aria-hidden="true">🌿</span>
          <p className="font-body text-sm text-earth/50 text-center leading-relaxed">
            Drop a leaf image here<br />or click to browse
          </p>
        </>
      )}
    </div>
  )
}

function ImageGrid({ steps }) {
  const [tooltip, setTooltip] = useState(null)

  return (
    <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-4 gap-3">
      {steps.map((step, i) => (
        <div
          key={i}
          className="bg-white/90 rounded-xl border border-seafoam shadow-sm overflow-hidden
                     hover:shadow-md hover:border-mint transition-all cursor-pointer"
          onClick={() => setTooltip(tooltip === i ? null : i)}
        >
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img
            src={`data:image/png;base64,${step.image}`}
            alt={step.name}
            className="w-full aspect-square object-cover"
          />
          <div className="px-2.5 py-2">
            <p className="font-body text-xs font-semibold text-earth leading-tight">{step.name}</p>
            {tooltip === i && (
              <p className="font-body text-[11px] text-earth/60 mt-1 leading-relaxed animate-fade-slide-in">
                {step.description}
              </p>
            )}
          </div>
        </div>
      ))}
    </div>
  )
}

// ─── Page ────────────────────────────────────────────────────────────────────

export default function AugmentationPage() {
  const [file, setFile]         = useState(null)
  const [preview, setPreview]   = useState(null)
  const [result, setResult]     = useState(null)
  const [activeTab, setActiveTab] = useState('train')
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError]       = useState(null)

  const handleFile = useCallback((f) => {
    setFile(f)
    setPreview(URL.createObjectURL(f))
    setResult(null)
    setError(null)
  }, [])

  const handleGenerate = useCallback(async () => {
    if (!file) return
    setIsLoading(true)
    setError(null)
    setResult(null)

    try {
      const body = new FormData()
      body.append('file', file)
      const res = await fetch('/api/augment-preview', { method: 'POST', body })
      if (!res.ok) {
        const json = await res.json().catch(() => ({}))
        throw new Error(json.detail || `Server error ${res.status}`)
      }
      setResult(await res.json())
      setActiveTab('train')
    } catch (err) {
      setError(err.message)
    } finally {
      setIsLoading(false)
    }
  }, [file])

  const handleRegenerate = useCallback(() => {
    if (file) handleGenerate()
  }, [file, handleGenerate])

  const displayedSteps = result
    ? (activeTab === 'train' ? result.train_steps : result.eval_steps)
    : null

  return (
    <div className="min-h-screen flex flex-col bg-cream">
      <Header />

      {/* Hero */}
      <section className="hero-bg py-12 px-6 text-center">
        <h2 className="font-display text-forest text-4xl sm:text-5xl font-bold leading-tight mb-3">
          Augmentation
          <span className="text-sage"> Explorer</span>
        </h2>
        <p className="font-body text-earth/60 text-lg max-w-xl mx-auto leading-relaxed">
          How GreenVision transforms leaf images to close the domain gap between
          studio dataset photos and real-world field conditions.
        </p>
      </section>

      <main className="flex-1 max-w-5xl w-full mx-auto px-6 py-10 space-y-12">

        {/* ── Pipeline overview ───────────────────────────────────────── */}
        <section>
          <div className="mb-5">
            <h3 className="font-display text-forest text-2xl font-bold">Pipeline Overview</h3>
            <p className="font-body text-earth/55 text-sm mt-1">
              Click any step to expand its purpose and rationale. Transforms are applied
              cumulatively from top to bottom.
            </p>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <PipelineCard
              title="Training Pipeline"
              subtitle={`${TRAIN_STEPS.length} stochastic transforms — applied only to the train split`}
              steps={TRAIN_STEPS}
              accent="bg-sage"
              borderColor="border-mint/40"
              headerBg="bg-seafoam/60"
            />
            <PipelineCard
              title="Inference Pipeline"
              subtitle={`${EVAL_STEPS.length} deterministic transforms — val, test, and live predictions`}
              steps={EVAL_STEPS}
              accent="bg-gold"
              borderColor="border-gold/30"
              headerBg="bg-gold/10"
            />
          </div>
        </section>

        {/* ── Background removal strategy ─────────────────────────────── */}
        <BgRemovalSection />

        {/* ── Live preview ────────────────────────────────────────────── */}
        <section>
          <div className="mb-5">
            <h3 className="font-display text-forest text-2xl font-bold">Try It Live</h3>
            <p className="font-body text-earth/55 text-sm mt-1">
              Upload any leaf photo to see it processed through each transform step.
              Each generation is random — regenerate to see a different augmentation sample.
            </p>
          </div>

          <div className="bg-white/80 border border-seafoam rounded-2xl p-6 shadow-sm space-y-4">
            <DropZone onFile={handleFile} preview={preview} fileName={file?.name} />

            <button
              onClick={handleGenerate}
              disabled={!file || isLoading}
              className="w-full py-3 rounded-xl font-body text-sm font-semibold
                         bg-forest text-cream hover:bg-sage transition-colors
                         disabled:opacity-40 disabled:cursor-not-allowed
                         flex items-center justify-center gap-2"
            >
              {isLoading ? (
                <>
                  <span className="w-4 h-4 rounded-full border-2 border-cream/40 border-t-cream animate-spinner inline-block" />
                  Applying transforms…
                </>
              ) : (
                'Generate Augmentation Preview'
              )}
            </button>

            {error && (
              <div className="bg-red-50 border border-red-200 rounded-xl p-4 font-body text-sm text-red-700">
                <strong className="font-semibold">Error: </strong>{error}
              </div>
            )}
          </div>

          {/* Results */}
          {result && (
            <div className="mt-6 space-y-4 animate-fade-slide-in">
              {/* Tabs */}
              <div className="flex items-center justify-between flex-wrap gap-3">
                <div className="flex gap-2">
                  {[
                    { id: 'train', label: `Training (${result.train_steps.length} steps)` },
                    { id: 'eval',  label: `Inference (${result.eval_steps.length} steps)` },
                  ].map(tab => (
                    <button
                      key={tab.id}
                      onClick={() => setActiveTab(tab.id)}
                      className={`font-body text-sm font-semibold px-4 py-2 rounded-xl transition-colors
                        ${activeTab === tab.id
                          ? 'bg-forest text-cream shadow-sm'
                          : 'bg-white/70 text-earth/60 hover:text-earth border border-seafoam'
                        }`}
                    >
                      {tab.label}
                    </button>
                  ))}
                </div>
                <button
                  onClick={handleRegenerate}
                  disabled={isLoading}
                  className="font-body text-xs text-sage hover:text-forest transition-colors
                             disabled:opacity-40 flex items-center gap-1"
                >
                  ↺ Regenerate
                </button>
              </div>

              <p className="font-body text-xs text-earth/40">
                Click any image to toggle its description.
              </p>

              <ImageGrid steps={displayedSteps} />
            </div>
          )}
        </section>

      </main>

      <footer className="py-6 border-t border-seafoam text-center">
        <p className="font-body text-xs text-earth/35">
          GreenVision &middot; EfficientNet-B0 &middot; 39 plant conditions
        </p>
      </footer>
    </div>
  )
}
