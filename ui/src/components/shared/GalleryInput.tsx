import { useId, useLayoutEffect, useRef } from 'react'
import { useGalleryInputs, type GalleryInputTarget } from '../../lib/galleryInputs'

type Props = Omit<GalleryInputTarget, 'id' | 'receive'> & {
  onFile: GalleryInputTarget['receive']
}

// Inputs register only while their actual sidecar controls are mounted. This
// keeps gallery routing in sync with the same handlers used by uploads/drops.
export function GalleryInput({ kind, label, disabledReason, onFile, getImages }: Props) {
  const id = useId()
  const receiver = useRef(onFile)
  const imageSource = useRef(getImages)
  useLayoutEffect(() => { receiver.current = onFile; imageSource.current = getImages })
  useLayoutEffect(() => {
    const target = { id, kind, label, disabledReason, receive: (file: File) => receiver.current(file),
      getImages: () => imageSource.current?.() || [] }
    useGalleryInputs.setState(state => ({ targets: [...state.targets, target] }))
    return () => {
      useGalleryInputs.setState(state => ({ targets: state.targets.filter(item => item.id !== id) }))
    }
  }, [id, kind, label, disabledReason])
  return null
}
