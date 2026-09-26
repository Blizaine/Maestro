import type { MiniMaxH3Reference, ReferencePackView, SavedOmniCharacter } from '../types'

export const PACK_VIEWS: [ReferencePackView, string][] = [
  ['face_closeup', 'Face close-up'],
  ['full_body_front', 'Full body — front'],
  ['full_body_three_quarter', 'Full body — three-quarter'],
  ['upper_body_three_quarter', 'Upper body — three-quarter'],
  ['full_body_back', 'Full body — back'],
]

export function referencePackImages(character: SavedOmniCharacter, revisionId?: string): MiniMaxH3Reference[] {
  const revision = character.reference_pack?.revisions.find(item => item.id === revisionId)
  if (!revision?.approved) throw new Error('Select a reviewed, approved Reference Pack revision.')
  return PACK_VIEWS.map(([view]) => {
    const image = revision.images[view]
    if (!image) throw new Error('This revision is missing a required view.')
    return {
      id: `${character.id}:${revision.id}:${view}`, type: 'image', path: image.path, url: image.url,
      filename: `${view}.png`, role: character.name, character_name: character.name,
      library_character_id: character.id, reference_pack_revision_id: revision.id,
      reference_pack_revision_number: revision.number, reference_pack_view: view,
      image_intent: 'identity', remove_background: false,
    }
  })
}
