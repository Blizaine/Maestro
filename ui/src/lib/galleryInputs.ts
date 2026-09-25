import { create } from 'zustand'

export type GalleryInputImage = File | { url: string; name: string }

export type GalleryInputTarget = {
  id: string
  kind: 'image' | 'video'
  label: string
  disabledReason?: string
  receive: (file: File) => void | boolean | Promise<void | boolean>
  getImages?: () => GalleryInputImage[]
}

export const useGalleryInputs = create<{ targets: GalleryInputTarget[]; receiving: boolean }>(() => ({
  targets: [], receiving: false,
}))

/** Snapshot only mounted image inputs, including local files not uploaded yet. */
export function captureGallerySourceImages() {
  const objectUrls: string[] = []
  const seen = new Set<File | string>()
  const images: { id: string; url: string; name: string }[] = []
  for (const target of useGalleryInputs.getState().targets) {
    if (target.kind !== 'image') continue
    for (const source of target.getImages?.() || []) {
      const key = source instanceof File ? source : source.url
      if (!key || seen.has(key)) continue
      seen.add(key)
      const url = source instanceof File ? URL.createObjectURL(source) : source.url
      if (source instanceof File) objectUrls.push(url)
      images.push({ id: `source:${target.id}:${images.length}`, url, name: `${target.label}: ${source.name}` })
    }
  }
  return { images, release: () => objectUrls.forEach(url => URL.revokeObjectURL(url)) }
}

export async function sendToGalleryInput(id: string, file: File) {
  if (useGalleryInputs.getState().receiving) throw new Error('Wait for the current media upload to finish.')
  const target = useGalleryInputs.getState().targets.find(item => item.id === id)
  if (!target) throw new Error('The input changed. Choose a destination in the current mode.')
  if (target.disabledReason) throw new Error(target.disabledReason)
  if (!file.type.startsWith(`${target.kind}/`)) throw new Error(`This input needs a ${target.kind}.`)
  useGalleryInputs.setState({ receiving: true })
  try {
    if (await target.receive(file) === false) throw new Error(`Could not add media to ${target.label}.`)
  } finally {
    useGalleryInputs.setState({ receiving: false })
  }
}
