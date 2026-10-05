export type AvatarModelIdentity = {
  model_type?: string
  architecture?: string
}

export type AvatarSpeakerRegion = {
  left: number
  top: number
  right: number
  bottom: number
}

export const DEFAULT_AVATAR_SPEAKER_LOCATIONS = '0:0:50:100 50:0:100:100'

export function isLongCatAvatarModel(model: AvatarModelIdentity | undefined): boolean {
  return model?.model_type === 'longcat_avatar' || model?.model_type === 'longcat_avatar_multi'
}

export function isMultiSpeakerAvatarModel(model: AvatarModelIdentity | undefined): boolean {
  return model?.model_type === 'longcat_avatar_multi'
}

function parseCoordinate(value: string): number | null {
  const trimmed = value.trim()
  if (!/^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:e[+-]?\d+)?$/i.test(trimmed)) return null
  const number = Number(trimmed)
  return Number.isFinite(number) && number >= 0 && number <= 100 ? number : null
}

function isUsableRegion(region: AvatarSpeakerRegion): boolean {
  if (!(region.left < region.right && region.top < region.bottom)) return false

  // LongCat clips speaker masks to the 5..95% image interior. Reject regions
  // that would disappear entirely after that clipping step.
  const clippedLeft = Math.max(region.left, 5)
  const clippedTop = Math.max(region.top, 5)
  const clippedRight = Math.min(region.right, 95)
  const clippedBottom = Math.min(region.bottom, 95)
  return clippedLeft < clippedRight && clippedTop < clippedBottom
}

export function parseAvatarSpeakerRegions(
  value: unknown,
): [AvatarSpeakerRegion, AvatarSpeakerRegion] | null {
  if (typeof value !== 'string') return null
  const boxes = value.trim().split(/\s+/)
  if (boxes.length !== 2 || boxes.some(box => !box)) return null

  const parsed = boxes.map(box => {
    const coordinates = box.split(':')
    if (coordinates.length === 2) {
      const left = parseCoordinate(coordinates[0])
      const right = parseCoordinate(coordinates[1])
      if (left === null || right === null) return null
      const region = { left, top: 0, right, bottom: 100 }
      return isUsableRegion(region) ? region : null
    }
    if (coordinates.length !== 4) return null
    const numbers = coordinates.map(parseCoordinate)
    if (numbers.some(number => number === null)) return null
    const [left, top, right, bottom] = numbers as [number, number, number, number]
    const region = { left, top, right, bottom }
    return isUsableRegion(region) ? region : null
  })

  if (!parsed[0] || !parsed[1]) return null
  return [parsed[0], parsed[1]]
}

export function formatAvatarSpeakerRegions(
  regions: [AvatarSpeakerRegion, AvatarSpeakerRegion],
): string {
  return regions
    .map(region => `${region.left}:${region.top}:${region.right}:${region.bottom}`)
    .join(' ')
}
