export type LayoutBox = [number, number, number, number]
export interface Flux3Element {
  id: string
  desc: string
  bbox?: LayoutBox
  from?: string
  src_bbox?: LayoutBox | null
  tgt_bbox?: LayoutBox | null
  [key: string]: unknown
}

export function parseFlux3Layout(prompt: string): { caption: string; elements: Flux3Element[] } {
  // Search candidate arrays backwards; prose can contain brackets and comments.
  for (let index = prompt.lastIndexOf('['); index >= 0; index = prompt.lastIndexOf('[', index - 1)) {
    try {
      const value: unknown = JSON.parse(prompt.slice(index).trim())
      if (Array.isArray(value) && value.every(item => item && typeof item.id === 'string' && typeof item.desc === 'string')) {
        return { caption: prompt.slice(0, index).trimEnd(), elements: value as Flux3Element[] }
      }
    } catch { /* This bracket belongs to prose or a nested array. */ }
    if (index === 0) break
  }
  if (/\[\s*\{/.test(prompt)) throw new Error('The layout JSON is incomplete. Correct it in the prompt before opening the helper.')
  return { caption: prompt, elements: [] }
}

export function validBox(value: unknown): value is LayoutBox {
  return Array.isArray(value) && value.length === 4
    && value.every(number => Number.isInteger(number) && number >= 0 && number <= 1000)
    && value[2] > value[0] && value[3] > value[1]
}

export function serializeFlux3Layout(caption: string, elements: Flux3Element[]): string {
  if (elements.some(element => element.from !== undefined)) elements = elements.map(element => {
    if (element.from !== undefined) return element
    const result = { ...element, from: 'target', src_bbox: null, tgt_bbox: element.bbox }
    delete result.bbox
    return result
  })
  const ids = new Set<string>()
  for (const element of elements) {
    if (!/^[A-Za-z][\w-]*$/.test(element.id) || ids.has(element.id)) throw new Error('Use a unique element ID beginning with a letter.')
    ids.add(element.id)
    if (!caption.includes(`<${element.id}>`)) throw new Error(`Add <${element.id}> to the scene caption.`)
    if (!element.desc.trim()) throw new Error(`Describe ${element.id}.`)
    if (element.from !== undefined) {
      if (!/^(target|ref_image_[0-9])$/.test(element.from)) throw new Error('Choose an uploaded reference or new content.')
      if (element.from !== 'target' && !validBox(element.src_bbox)) throw new Error(`Draw the source region for ${element.id}.`)
      if (element.tgt_bbox !== null && !validBox(element.tgt_bbox)) throw new Error(`Draw the output box for ${element.id}.`)
    } else if (!validBox(element.bbox)) throw new Error(`Draw the output box for ${element.id}.`)
  }
  return elements.length ? `${caption.trimEnd()}\n${JSON.stringify(elements, null, 2)}` : caption
}
