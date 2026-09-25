/** Decode an uploaded source before accepting it into a timeline or canvas.
 * The caller owns the returned preview URL; failed probes release it here.
 */
export async function loadMediaInput(file: File): Promise<{ url: string; duration: number; resolution: string }> {
  const url = URL.createObjectURL(file)
  const video = file.type.startsWith('video/') ? document.createElement('video') : null
  const image = video ? null : new Image()
  try {
    return await new Promise((resolve, reject) => {
      const timer = setTimeout(() => reject(new Error('Media loading timed out.')), 15000)
      const fail = () => { clearTimeout(timer); reject(new Error('Could not read this media file.')) }
      if (video) {
        video.preload = 'metadata'
        video.onloadedmetadata = () => {
          clearTimeout(timer)
          resolve({ url, duration: Number.isFinite(video.duration) ? video.duration : 0,
            resolution: `${video.videoWidth}x${video.videoHeight}` })
        }
        video.onerror = fail
        video.src = url
      } else if (image) {
        image.onload = () => {
          clearTimeout(timer)
          resolve({ url, duration: 0, resolution: `${image.naturalWidth}x${image.naturalHeight}` })
        }
        image.onerror = fail
        image.src = url
      }
    })
  } catch (error) {
    URL.revokeObjectURL(url)
    throw error
  } finally {
    if (video) {
      video.onloadedmetadata = null
      video.onerror = null
      video.removeAttribute('src')
      video.load()
    }
    if (image) { image.onload = null; image.onerror = null }
  }
}
