import { useEffect, useMemo, useRef, useState } from 'react'
import { Upload } from 'lucide-react'

export interface GalleryImageChoice {
  id: string
  name: string
  url: string
}

type ImageOption = GalleryImageChoice & {
  key: string
  group: 'source' | 'gallery' | 'device'
  fileFingerprint?: string
}

interface ImageComparisonProps {
  currentImage: GalleryImageChoice
  sourceImages: GalleryImageChoice[]
  comparisonImages: GalleryImageChoice[]
}

function choiceKey(group: ImageOption['group'], image: GalleryImageChoice): string {
  return `${group}:${image.id}:${image.url}`
}

function isSameImage(a: ImageOption | undefined, b: ImageOption | undefined): boolean {
  return Boolean(a && b && (
    a.key === b.key
    || a.url === b.url
    || (a.fileFingerprint && a.fileFingerprint === b.fileFingerprint)
  ))
}

function ImagePicker({
  side,
  value,
  options,
  disabledUrl,
  onChange,
  onChooseDevice,
}: {
  side: 'Before' | 'After'
  value: string
  options: ImageOption[]
  disabledUrl: string | undefined
  onChange: (key: string) => void
  onChooseDevice: () => void
}) {
  return (
    <label className="flex min-w-0 flex-1 items-center gap-1.5">
      <span className="shrink-0 text-[10px] font-semibold text-white/75">{side}</span>
      <select
        aria-label={`${side} comparison image`}
        value={value}
        onChange={event => onChange(event.currentTarget.value)}
        className="h-8 min-w-0 flex-1 rounded-md border border-white/15 bg-black/65 px-2 text-[11px] text-white outline-none focus:border-accent-blue"
      >
        {!value && <option value="">Choose an image…</option>}
        {options.some(option => option.group === 'source') && (
          <optgroup label="Generation sources">
            {options.filter(option => option.group === 'source').map(option => (
              <option key={option.key} value={option.key} disabled={option.url === disabledUrl}>
                {option.name}
              </option>
            ))}
          </optgroup>
        )}
        {options.some(option => option.group === 'gallery') && (
          <optgroup label="Gallery images">
            {options.filter(option => option.group === 'gallery').map(option => (
              <option key={option.key} value={option.key} disabled={option.url === disabledUrl}>
                {option.name}
              </option>
            ))}
          </optgroup>
        )}
        {options.some(option => option.group === 'device') && (
          <optgroup label="This device · private to this viewer">
            {options.filter(option => option.group === 'device').map(option => (
              <option key={option.key} value={option.key} disabled={option.url === disabledUrl}>
                {option.name}
              </option>
            ))}
          </optgroup>
        )}
      </select>
      <button
        type="button"
        onClick={onChooseDevice}
        className="flex h-8 shrink-0 items-center gap-1 rounded-md border border-white/15 bg-black/55 px-2 text-[10px] text-white/80 hover:bg-white/10 hover:text-white focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent-blue"
        aria-label={`Choose ${side.toLowerCase()} image from this device`}
        title="Choose an image from this device. It stays in this viewer."
      >
        <Upload size={13} />
        <span className="hidden sm:inline">Device</span>
      </button>
    </label>
  )
}

export function ImageComparison({ currentImage, sourceImages, comparisonImages }: ImageComparisonProps) {
  const stageRef = useRef<HTMLDivElement>(null)
  const beforeInputRef = useRef<HTMLInputElement>(null)
  const afterInputRef = useRef<HTMLInputElement>(null)
  const nextDeviceId = useRef(0)
  const deviceOptionsRef = useRef<ImageOption[]>([])
  const [deviceOptions, setDeviceOptions] = useState<ImageOption[]>([])
  const [beforeKey, setBeforeKey] = useState<string | null>(null)
  const [afterKey, setAfterKey] = useState<string | null>(null)
  const [position, setPosition] = useState(50)
  const [deviceError, setDeviceError] = useState('')
  const [failedUrls, setFailedUrls] = useState<Set<string>>(() => new Set())

  const options = useMemo<ImageOption[]>(() => [
    ...sourceImages.map(image => ({ ...image, group: 'source' as const, key: choiceKey('source', image) })),
    ...comparisonImages.map(image => ({ ...image, group: 'gallery' as const, key: choiceKey('gallery', image) })),
    ...(comparisonImages.some(image => image.url === currentImage.url)
      ? []
      : [{ ...currentImage, group: 'gallery' as const, key: choiceKey('gallery', currentImage) }]),
    ...deviceOptions,
  ], [comparisonImages, currentImage, deviceOptions, sourceImages])

  const currentOption = useMemo<ImageOption>(() => ({
    ...currentImage,
    group: 'gallery',
    key: choiceKey('gallery', currentImage),
  }), [currentImage])

  const beforeOption = options.find(option => option.key === beforeKey)
    || (beforeKey === null ? options.find(option => option.group === 'source') : undefined)
  const afterOption = options.find(option => option.key === afterKey)
    || (afterKey === null ? options.find(option => option.group === 'gallery' && option.url === currentImage.url) || currentOption : undefined)
  const canCompare = Boolean(beforeOption && afterOption && !isSameImage(beforeOption, afterOption))

  const markImageLoad = (url: string, failed: boolean) => {
    setFailedUrls(current => {
      if (current.has(url) === failed) return current
      const next = new Set(current)
      if (failed) next.add(url)
      else next.delete(url)
      return next
    })
  }

  useEffect(() => () => {
    for (const option of deviceOptionsRef.current) URL.revokeObjectURL(option.url)
  }, [])

  const chooseDeviceFile = (side: 'before' | 'after', file: File | undefined) => {
    if (!file) return
    if (!file.type.startsWith('image/')) {
      setDeviceError('Choose an image file to compare.')
      return
    }

    const url = URL.createObjectURL(file)
    const id = `device-${++nextDeviceId.current}`
    const option: ImageOption = {
      id,
      name: file.name || 'Device image',
      url,
      key: choiceKey('device', { id, name: file.name || 'Device image', url }),
      group: 'device',
      fileFingerprint: `${file.name.toLocaleLowerCase()}\u0000${file.size}\u0000${file.lastModified}`,
    }
    setDeviceError('')
    const nextDeviceOptions = [...deviceOptionsRef.current, option]
    deviceOptionsRef.current = nextDeviceOptions
    setDeviceOptions(nextDeviceOptions)
    if (side === 'before') setBeforeKey(option.key)
    else setAfterKey(option.key)
  }

  const updatePositionFromPointer = (clientX: number) => {
    const bounds = stageRef.current?.getBoundingClientRect()
    if (!bounds || bounds.width <= 0) return
    setPosition(Math.min(100, Math.max(0, ((clientX - bounds.left) / bounds.width) * 100)))
  }

  const activeBeforeLabel = beforeOption?.name || 'No image selected'
  const activeAfterLabel = afterOption?.name || 'No image selected'

  return (
    <div className="flex min-h-0 flex-1 flex-col">
      <div
        ref={stageRef}
        role="group"
        aria-label="Before and after image comparison"
        className="relative flex min-h-0 flex-1 items-center justify-center overflow-hidden bg-black"
        style={{ touchAction: 'none' }}
      >
        {afterOption && (
          <img
            src={afterOption.url}
            alt={`After: ${activeAfterLabel}`}
            draggable={false}
            className={`pointer-events-none absolute inset-0 h-full w-full select-none object-contain ${failedUrls.has(afterOption.url) ? 'hidden' : ''}`}
            onLoad={() => markImageLoad(afterOption.url, false)}
            onError={() => markImageLoad(afterOption.url, true)}
          />
        )}
        {beforeOption && (
          <div
            className="pointer-events-none absolute inset-0 overflow-hidden bg-black"
            style={{ clipPath: `inset(0 ${100 - position}% 0 0)` }}
          >
            <img
              src={beforeOption.url}
              alt={`Before: ${activeBeforeLabel}`}
              draggable={false}
              className={`h-full w-full select-none object-contain ${failedUrls.has(beforeOption.url) ? 'hidden' : ''}`}
              onLoad={() => markImageLoad(beforeOption.url, false)}
              onError={() => markImageLoad(beforeOption.url, true)}
            />
          </div>
        )}

        {canCompare && (
          <div
            className="absolute inset-y-0 z-10 w-8 -translate-x-1/2 cursor-ew-resize touch-none"
            style={{ left: `${position}%` }}
            data-gallery-gesture-ignore
            onPointerDown={event => {
              event.preventDefault()
              event.stopPropagation()
              event.currentTarget.setPointerCapture(event.pointerId)
              updatePositionFromPointer(event.clientX)
            }}
            onPointerMove={event => {
              if (event.currentTarget.hasPointerCapture(event.pointerId)) {
                event.preventDefault()
                event.stopPropagation()
                updatePositionFromPointer(event.clientX)
              }
            }}
            onPointerUp={event => {
              if (event.currentTarget.hasPointerCapture(event.pointerId)) event.currentTarget.releasePointerCapture(event.pointerId)
            }}
            onPointerCancel={event => {
              if (event.currentTarget.hasPointerCapture(event.pointerId)) event.currentTarget.releasePointerCapture(event.pointerId)
            }}
            aria-hidden="true"
          >
            <span className="absolute inset-y-0 left-1/2 w-px -translate-x-1/2 bg-white shadow-[0_0_5px_rgba(0,0,0,0.8)]" />
            <span className="absolute left-1/2 top-1/2 grid h-9 w-7 -translate-x-1/2 -translate-y-1/2 place-items-center rounded-full border border-white/50 bg-black/75 text-white shadow-lg">
              <span className="h-3.5 w-0.5 rounded-full bg-white/80" />
            </span>
          </div>
        )}

        {beforeOption && <span className="pointer-events-none absolute left-2 top-2 z-20 max-w-[42%] truncate rounded bg-black/60 px-2 py-1 text-[10px] text-white/90">Before · {activeBeforeLabel}</span>}
        {afterOption && <span className="pointer-events-none absolute right-2 top-2 z-20 max-w-[42%] truncate rounded bg-black/60 px-2 py-1 text-[10px] text-white/90">After · {activeAfterLabel}</span>}

        {!beforeOption && (
          <div className="pointer-events-none absolute inset-x-4 top-1/2 z-20 -translate-y-1/2 text-center text-xs text-white/75">
            No generation source is available. Choose a Before image below.
          </div>
        )}
        {beforeOption && afterOption && isSameImage(beforeOption, afterOption) && (
          <div className="pointer-events-none absolute inset-x-4 bottom-3 z-20 rounded bg-black/70 px-3 py-2 text-center text-[11px] text-white/85">
            Choose a different image for Before and After to compare them.
          </div>
        )}
        {(beforeOption && failedUrls.has(beforeOption.url) || afterOption && failedUrls.has(afterOption.url)) && (
          <div role="alert" className="pointer-events-none absolute inset-x-3 bottom-3 z-20 rounded bg-black/85 px-3 py-2 text-center text-[11px] text-rose-200">
            {beforeOption && failedUrls.has(beforeOption.url) && afterOption && failedUrls.has(afterOption.url)
              ? 'Before and After images could not be loaded. Choose different images.'
              : beforeOption && failedUrls.has(beforeOption.url)
                ? 'The Before image could not be loaded. Choose another image.'
                : 'The After image could not be loaded. Choose another image.'}
          </div>
        )}
      </div>

      <div className="shrink-0 border-t border-white/10 bg-black/85 px-2 py-2 sm:px-4">
        <div className="flex min-w-0 gap-2">
          <ImagePicker
            side="Before"
            value={beforeOption?.key || ''}
            options={options}
            disabledUrl={afterOption?.url}
            onChange={setBeforeKey}
            onChooseDevice={() => beforeInputRef.current?.click()}
          />
          <ImagePicker
            side="After"
            value={afterOption?.key || ''}
            options={options}
            disabledUrl={beforeOption?.url}
            onChange={setAfterKey}
            onChooseDevice={() => afterInputRef.current?.click()}
          />
        </div>
        <div className="mt-1.5 flex items-center gap-2">
          <span className="w-12 shrink-0 text-[9px] text-white/55">After</span>
          <input
            type="range"
            min={0}
            max={100}
            step={1}
            value={position}
            disabled={!canCompare}
            onChange={event => setPosition(Number(event.currentTarget.value))}
            aria-label="Comparison divider; zero shows all After and one hundred shows all Before"
            aria-valuetext={`${position}% Before, ${100 - position}% After`}
            className="h-5 min-w-0 flex-1 cursor-ew-resize accent-accent-blue disabled:cursor-not-allowed disabled:opacity-40"
            style={{ touchAction: 'none' }}
          />
          <span className="w-12 shrink-0 text-right text-[9px] text-white/55">Before</span>
        </div>
        <p aria-live="polite" className="min-h-3 text-center text-[9px] text-white/55">
          {deviceError || (!sourceImages.length ? 'No source images are available. Device images stay private to this viewer.' : 'Drag the divider left to reveal After, or right to reveal Before.')}
        </p>
        <input
          ref={beforeInputRef}
          type="file"
          accept="image/*"
          tabIndex={-1}
          aria-label="Choose Before image from this device"
          className="sr-only"
          onChange={event => {
            chooseDeviceFile('before', event.currentTarget.files?.[0])
            event.currentTarget.value = ''
          }}
        />
        <input
          ref={afterInputRef}
          type="file"
          accept="image/*"
          tabIndex={-1}
          aria-label="Choose After image from this device"
          className="sr-only"
          onChange={event => {
            chooseDeviceFile('after', event.currentTarget.files?.[0])
            event.currentTarget.value = ''
          }}
        />
      </div>
    </div>
  )
}
