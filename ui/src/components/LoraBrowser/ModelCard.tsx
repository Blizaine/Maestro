import { Download, ThumbsUp } from 'lucide-react'
import type { CivitAIModel } from '../../types'

interface Props {
  model: CivitAIModel
  onClick: () => void
}

function formatCount(n: number): string {
  if (n >= 1000000) return `${(n / 1000000).toFixed(1)}M`
  if (n >= 1000) return `${(n / 1000).toFixed(1)}K`
  return String(n)
}

export function ModelCard({ model, onClick }: Props) {
  // Get first image/video from first version
  const allMedia = model.modelVersions?.[0]?.images || []
  // Prefer a still image over video for the card thumbnail
  const firstImage = allMedia.find(m => m.type === 'image') || allMedia[0]
  const isVideo = firstImage?.type === 'video' ||
    firstImage?.url?.endsWith('.mp4') || firstImage?.url?.endsWith('.webm')

  return (
    <button
      onClick={onClick}
      className="group relative rounded-lg border border-border overflow-hidden bg-bg-tertiary hover:border-accent-blue transition-all text-left"
    >
      {/* Thumbnail */}
      <div className="aspect-[3/4] bg-bg-active overflow-hidden">
        {firstImage ? (
          isVideo ? (
            <video
              src={firstImage.url}
              className="w-full h-full object-cover group-hover:scale-105 transition-transform duration-300"
              muted
              loop
              autoPlay
              playsInline
            />
          ) : (
            <img
              src={firstImage.url}
              alt={model.name}
              className="w-full h-full object-cover group-hover:scale-105 transition-transform duration-300"
              loading="lazy"
              referrerPolicy="no-referrer"
            />
          )
        ) : (
          <div className="w-full h-full flex items-center justify-center text-text-muted text-xs">
            No preview
          </div>
        )}
      </div>

      {/* Info overlay at bottom */}
      <div className="absolute bottom-0 left-0 right-0 bg-gradient-to-t from-black/80 via-black/50 to-transparent p-2 pt-6">
        <div className="text-xs font-medium text-white truncate">{model.name}</div>
        <div className="flex items-center gap-2 mt-0.5">
          <span className="flex items-center gap-0.5 text-[10px] text-white/70">
            <ThumbsUp size={9} />
            {formatCount(model.stats.thumbsUpCount)}
          </span>
          <span className="flex items-center gap-0.5 text-[10px] text-white/70">
            <Download size={9} />
            {formatCount(model.stats.downloadCount)}
          </span>
          {model.creator?.username && (
            <span className="text-[10px] text-white/50 truncate ml-auto">
              {model.creator.username}
            </span>
          )}
        </div>
      </div>

      {/* Type badge */}
      <div className="absolute top-1.5 left-1.5">
        <span className="text-[9px] px-1.5 py-0.5 rounded bg-black/60 text-white/80 font-medium">
          {model.type}
        </span>
      </div>
    </button>
  )
}
