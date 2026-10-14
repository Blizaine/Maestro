import { formatMediaDuration } from '../../lib/mediaDuration'

interface Props {
  seconds: number | null
  className: string
}

export function MediaDurationBadge({ seconds, className }: Props) {
  const label = formatMediaDuration(seconds)
  if (!label) return null

  return (
    <span
      data-media-duration-badge="true"
      aria-label={`Video duration ${label}`}
      title={`Duration ${label}`}
      className={`pointer-events-none absolute z-10 inline-flex items-center rounded bg-black/75 font-semibold leading-none text-white shadow ${className}`}
    >
      {label}
    </span>
  )
}
