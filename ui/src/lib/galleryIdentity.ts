import type { OutputFile } from '../types'
import type { ApiOutput } from '../api/client'

export function outputIdentity(output: Pick<OutputFile, 'id' | 'workspace' | 'name'>): string {
  return output.id || `${output.workspace || 'default'}/${output.name}`
}

export function galleryOutput(output: ApiOutput): OutputFile {
  return {
    ...output,
    mode: (output.mode as OutputFile['mode']) || null,
    edit_sub_mode: (output.edit_sub_mode as OutputFile['edit_sub_mode']) || null,
    favorite: Boolean(output.favorite),
  }
}
