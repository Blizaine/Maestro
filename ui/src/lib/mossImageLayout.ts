export type MossImageBox = [number, number, number, number]

export interface MossImageSize {
  width: number
  height: number
}

export interface MossImageItem {
  label?: string
  caption?: string
  text?: string
  bbox?: unknown
  [key: string]: unknown
}

export interface MossImageSummary {
  meta: unknown[]
  canvas: string
  caption: string
  [key: string]: unknown
}

export interface MossImageDocument {
  summary: MossImageSummary
  items: MossImageItem[]
  bbox_coord_space: 'abs_centralized'
  bbox_format: 'yxyx'
  [key: string]: unknown
}

export interface ParsedMossImageLayout {
  document: MossImageDocument
  comments: string[]
  plain: boolean
}

export interface CanvasRect {
  left: number
  top: number
  width: number
  height: number
}

export function emptyMossImageDocument(caption = ''): MossImageDocument {
  return {
    summary: { meta: [], canvas: 'single', caption },
    items: [],
    bbox_coord_space: 'abs_centralized',
    bbox_format: 'yxyx',
  }
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return Boolean(value) && typeof value === 'object' && !Array.isArray(value)
}

function removeCommentLines(source: string): { text: string; comments: string[] } {
  const comments: string[] = []
  const lines = source.split(/\r\n|\n|\r/)
  const text = lines.map(line => {
    if (/^[\t ]*#/.test(line)) {
      comments.push(line)
      return ''
    }
    return line
  }).join('\n')
  return { text, comments }
}

function jsonError(error: unknown, source: string): Error {
  const detail = error instanceof Error ? error.message : String(error)
  const location = /position\s+(\d+)/i.exec(detail)
  if (!location) return new Error(`The layout JSON is invalid. ${detail} Original prompt text is preserved.`)
  const position = Math.min(Number(location[1]), source.length)
  const before = source.slice(0, position)
  const line = before.split('\n').length
  const column = position - before.lastIndexOf('\n')
  return new Error(`The layout JSON is invalid at line ${line}, column ${column}. ${detail} Original prompt text is preserved.`)
}

function normalizeDocument(value: unknown): MossImageDocument {
  if (!isRecord(value) || !isRecord(value.summary) || !Array.isArray(value.items)) {
    throw new Error('Expected a JSON object with a summary object and an items array. Original prompt text is preserved.')
  }

  const summary = value.summary
  if (summary.caption !== undefined && typeof summary.caption !== 'string') {
    throw new Error('summary.caption must be text. Original prompt text is preserved.')
  }
  if (summary.meta !== undefined && !Array.isArray(summary.meta)) {
    throw new Error('summary.meta must be a list. Original prompt text is preserved.')
  }
  if (summary.canvas !== undefined && typeof summary.canvas !== 'string') {
    throw new Error('summary.canvas must be text. Original prompt text is preserved.')
  }

  value.items.forEach((item, index) => {
    if (!isRecord(item)) throw new Error(`Item ${index + 1} must be an object. Original prompt text is preserved.`)
    for (const key of ['label', 'caption', 'text']) {
      if (item[key] !== undefined && typeof item[key] !== 'string') {
        throw new Error(`Item ${index + 1}: ${key} must be text. Original prompt text is preserved.`)
      }
    }
  })

  const document = value as unknown as MossImageDocument
  document.summary = {
    ...summary,
    meta: summary.meta === undefined ? [] : summary.meta,
    canvas: summary.canvas === undefined ? 'single' : summary.canvas,
    caption: summary.caption === undefined ? '' : summary.caption,
  }
  if (document.bbox_coord_space === undefined) document.bbox_coord_space = 'abs_centralized'
  if (document.bbox_format === undefined) document.bbox_format = 'yxyx'
  if (document.bbox_coord_space !== 'abs_centralized' || document.bbox_format !== 'yxyx') {
    throw new Error('This editor requires bbox_coord_space "abs_centralized" and bbox_format "yxyx"; coordinates were not changed. Original prompt text is preserved.')
  }
  return document
}

export function parseMossImageLayout(source: string): ParsedMossImageLayout {
  const { text: withoutComments, comments } = removeCommentLines(source)
  const candidate = withoutComments.trim()

  if (!candidate || !/^[{[]/.test(candidate)) {
    if (/^[\s]*['"]?(?:summary|items)['"]?\s*:/i.test(candidate)) {
      throw new Error('The layout is missing its opening object brace. Original prompt text is preserved.')
    }
    const caption = comments.length ? candidate : source
    return { document: emptyMossImageDocument(caption), comments, plain: true }
  }

  let parsed: unknown
  try {
    parsed = JSON.parse(candidate)
  } catch (error) {
    throw jsonError(error, candidate)
  }
  return { document: normalizeDocument(parsed), comments, plain: false }
}

export function serializeMossImageLayout(document: MossImageDocument, comments: readonly string[] = []): string {
  const body = JSON.stringify(document, null, 2)
  return comments.length ? `${comments.join('\n')}\n${body}` : body
}

export function parseMossImageSize(resolution: string | null | undefined): MossImageSize {
  const match = /^(\d+)\s*x\s*(\d+)$/i.exec(String(resolution || '').trim())
  if (match) {
    const width = Number(match[1])
    const height = Number(match[2])
    if (Number.isSafeInteger(width) && width > 0 && Number.isSafeInteger(height) && height > 0) return { width, height }
  }
  return { width: 1024, height: 1024 }
}

export function validMossImageBox(value: unknown, size: MossImageSize): value is MossImageBox {
  if (!Array.isArray(value) || value.length !== 4 || !value.every(Number.isInteger)) return false
  const [top, left, bottom, right] = value as number[]
  return top < bottom && left < right
    && top >= -size.height / 2 && bottom <= size.height / 2
    && left >= -size.width / 2 && right <= size.width / 2
}

export function mossImageBoxProblem(value: unknown, size: MossImageSize): string {
  if (!Array.isArray(value) || value.length !== 4 || !value.every(Number.isFinite)) {
    return 'Enter four finite numbers in [ymin, xmin, ymax, xmax] order.'
  }
  if (!value.every(Number.isInteger)) return 'Use integer pixel corners in [ymin, xmin, ymax, xmax] order.'
  const [top, left, bottom, right] = value as number[]
  if (top >= bottom || left >= right) return 'The top must be above the bottom and the left must be before the right.'
  const problems: string[] = []
  if (top < -size.height / 2) problems.push(`ymin ${top} is above ${-size.height / 2}`)
  if (bottom > size.height / 2) problems.push(`ymax ${bottom} is below ${size.height / 2}`)
  if (left < -size.width / 2) problems.push(`xmin ${left} is left of ${-size.width / 2}`)
  if (right > size.width / 2) problems.push(`xmax ${right} is right of ${size.width / 2}`)
  return problems.length ? `${problems.join('; ')} on the ${size.width} × ${size.height} canvas.` : ''
}

export function fitMossImageBoxToCanvas(value: unknown, size: MossImageSize): MossImageBox | null {
  if (!Array.isArray(value) || value.length !== 4 || !value.every(Number.isFinite)) return null
  const axis = (startValue: number, endValue: number, length: number): [number, number] => {
    const lowEdge = Math.ceil(-length / 2)
    const highEdge = Math.floor(length / 2)
    const start = Math.round(Math.min(startValue, endValue))
    const end = Math.round(Math.max(startValue, endValue))
    const extent = Math.min(highEdge - lowEdge, Math.max(1, end - start))
    const low = Math.max(lowEdge, Math.min(highEdge - extent, start))
    return [low, low + extent]
  }
  const [top, bottom] = axis(value[0], value[2], size.height)
  const [left, right] = axis(value[1], value[3], size.width)
  return [top, left, bottom, right]
}

export function mossCanvasPoint(clientX: number, clientY: number, rect: CanvasRect, size: MossImageSize): [number, number] {
  const x = rect.width > 0 ? ((clientX - rect.left) / rect.width) * size.width - size.width / 2 : 0
  const y = rect.height > 0 ? ((clientY - rect.top) / rect.height) * size.height - size.height / 2 : 0
  return [
    Math.max(Math.ceil(-size.height / 2), Math.min(Math.floor(size.height / 2), Math.round(y))),
    Math.max(Math.ceil(-size.width / 2), Math.min(Math.floor(size.width / 2), Math.round(x))),
  ]
}

export function mossImageBoxStyle(box: MossImageBox, size: MossImageSize) {
  return {
    top: `${((box[0] + size.height / 2) / size.height) * 100}%`,
    left: `${((box[1] + size.width / 2) / size.width) * 100}%`,
    height: `${((box[2] - box[0]) / size.height) * 100}%`,
    width: `${((box[3] - box[1]) / size.width) * 100}%`,
  }
}
