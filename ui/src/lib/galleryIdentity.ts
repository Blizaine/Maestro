import type { OutputFile } from '../types'
import type { ApiOutput } from '../api/client'

export function outputIdentity(output: Pick<OutputFile, 'id' | 'workspace' | 'name'>): string {
  return output.id || `${output.workspace || 'default'}/${output.name}`
}

type GalleryOrderFields = Pick<OutputFile, 'created_at' | 'workspace' | 'name'>

function compareCodePointStrings(left: string, right: string): number {
  const leftPoints = left[Symbol.iterator]()
  const rightPoints = right[Symbol.iterator]()
  while (true) {
    const leftPoint = leftPoints.next()
    const rightPoint = rightPoints.next()
    if (leftPoint.done || rightPoint.done) {
      return leftPoint.done === rightPoint.done ? 0 : leftPoint.done ? -1 : 1
    }
    const leftCodePoint = leftPoint.value.codePointAt(0)!
    const rightCodePoint = rightPoint.value.codePointAt(0)!
    if (leftCodePoint !== rightCodePoint) return leftCodePoint - rightCodePoint
  }
}

/** The API orders gallery files by (mtime, workspace, name), newest first. */
export function galleryOutputIsOlder(output: GalleryOrderFields, boundary: GalleryOrderFields): boolean {
  if (output.created_at !== boundary.created_at) return output.created_at < boundary.created_at
  const workspaceOrder = compareCodePointStrings(output.workspace || 'default', boundary.workspace || 'default')
  if (workspaceOrder !== 0) return workspaceOrder < 0
  return compareCodePointStrings(output.name, boundary.name) < 0
}

export function galleryOutput(output: ApiOutput): OutputFile {
  return {
    ...output,
    mode: (output.mode as OutputFile['mode']) || null,
    edit_sub_mode: (output.edit_sub_mode as OutputFile['edit_sub_mode']) || null,
    favorite: Boolean(output.favorite),
  }
}
