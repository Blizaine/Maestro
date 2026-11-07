import type {
  EditorAsset,
  EditorGraphic,
  EditorGraphicElement,
  EditorGraphicImageElement,
  EditorGraphicLayout,
} from '../types'

export const EDITOR_GRAPHIC_WIDTH = 1344 as const
export const EDITOR_GRAPHIC_HEIGHT = 768 as const

export function createEditorGraphic(
  layout: EditorGraphicLayout,
  createId: () => string,
): EditorGraphic {
  const text = (options: Omit<Extract<EditorGraphicElement, { type: 'text' }>, 'id' | 'type'>) => ({
    id: createId(),
    type: 'text' as const,
    ...options,
  })
  const rect = (options: Omit<Extract<EditorGraphicElement, { type: 'rect' }>, 'id' | 'type'>) => ({
    id: createId(),
    type: 'rect' as const,
    ...options,
  })

  const elements: EditorGraphicElement[] = layout === 'lower-third'
    ? [
        rect({ x: 84, y: 566, width: 690, height: 116, fill: '#101827', opacity: 0.9, radius: 14, border_color: '#64d6bf', border_width: 3 }),
        text({ x: 118, y: 584, width: 620, height: 72, text: 'YOUR TITLE', font_family: 'Arial Black', font_size: 42, color: '#ffffff', bold: true, align: 'left' }),
      ]
    : layout === 'feature-panel'
      ? [
          rect({ x: 88, y: 98, width: 520, height: 572, fill: '#101827', opacity: 0.88, radius: 22 }),
          text({ x: 132, y: 172, width: 430, height: 178, text: 'A FEATURED\nMOMENT', font_family: 'Arial Black', font_size: 64, color: '#ffffff', bold: true, align: 'left' }),
          text({ x: 134, y: 386, width: 424, height: 166, text: 'Add a short supporting detail here.', font_family: 'Arial', font_size: 34, color: '#d4d9e4', align: 'left' }),
        ]
      : [
          text({ x: 142, y: 258, width: 1060, height: 152, text: 'YOUR TITLE', font_family: 'Arial Black', font_size: 82, color: '#ffffff', bold: true, align: 'center' }),
          text({ x: 242, y: 422, width: 860, height: 74, text: 'A subtitle or date', font_family: 'Arial', font_size: 34, color: '#d4d9e4', align: 'center' }),
        ]

  return {
    version: 1,
    width: EDITOR_GRAPHIC_WIDTH,
    height: EDITOR_GRAPHIC_HEIGHT,
    elements,
  }
}

/**
 * Resolve only image records explicitly referenced by this graphic. Preview
 * requests never receive the rest of the project's media library.
 */
export function editorGraphicPreviewAssets(
  graphic: EditorGraphic,
  assets: Record<string, EditorAsset>,
): Record<string, EditorAsset> {
  const referencedIds = new Set(graphic.elements.flatMap(element => (
    element.type === 'image' ? [element.asset_id] : []
  )))
  return Object.fromEntries(
    [...referencedIds].sort().flatMap(id => {
      const asset = assets[id]
      return asset?.type === 'image' ? [[id, asset] as const] : []
    }),
  )
}

/** A stable key for cache reuse across playhead-only animation updates. */
export function editorGraphicPreviewKey(
  graphic: EditorGraphic,
  width: number,
  height: number,
  assets: Record<string, EditorAsset>,
  workspace: string,
): string {
  const previewAssets = editorGraphicPreviewAssets(graphic, assets)
  return JSON.stringify([workspace, width, height, graphic, previewAssets])
}

export function editorGraphicImageAssetIds(graphic: EditorGraphic): string[] {
  return [...new Set(graphic.elements.flatMap((element): string[] => (
    element.type === 'image' ? [element.asset_id] : []
  )))]
}

export function createEditorGraphicImageElement(
  assetId: string,
  createId: () => string,
): EditorGraphicImageElement {
  return {
    id: createId(),
    type: 'image',
    x: 762,
    y: 194,
    width: 470,
    height: 380,
    asset_id: assetId,
    opacity: 1,
    fit: 'contain',
  }
}
