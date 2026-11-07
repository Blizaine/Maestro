import assert from 'node:assert/strict'
import test from 'node:test'
import { createJiti } from 'jiti'
const jiti = createJiti(import.meta.url)
const { createEditorGraphic, editorGraphicPreviewAssets, editorGraphicPreviewKey } = await jiti.import('./editorGraphics.ts')
const { useEditorStore } = await jiti.import('./useEditorStore.ts')

function imageAsset(id, name = `${id}.png`) {
  return {
    id,
    name,
    type: 'image',
    origin: 'project',
    workspace: 'default',
    path: `project/${name}`,
    url: `/api/v1/file/${name}`,
    duration: 0,
    width: 1200,
    height: 800,
    fps: 0,
    has_audio: false,
  }
}

function textProject(graphic) {
  return {
    schema_version: 1,
    id: 'graphic-project',
    name: 'Graphic test',
    workspace: 'default',
    created_at: 1,
    updated_at: 1,
    canvas: { width: 1920, height: 1080, fps: 30, background: '#000000' },
    assets: {},
    tracks: [{
      id: 'text-track', name: 'Text', type: 'text', z_index: 10, muted: false, locked: false,
      items: [{
        id: 'graphic-item', name: 'Feature panel', start: 0, duration: 4, source_in: 0, speed: 1,
        volume: 1, opacity: 1, fit: 'contain', transform: { x: 0, y: 0, scale: 1, rotation: 0 }, graphic,
      }],
    }],
    markers: [],
    export: { quality: 'balanced', include_audio: true },
    exports: [],
  }
}

test('graphic layouts use the versioned design canvas and unique element ids', () => {
  const graphic = createEditorGraphic('feature-panel', (() => {
    let next = 0
    return () => `element-${++next}`
  })())

  assert.deepEqual([graphic.version, graphic.width, graphic.height], [1, 1344, 768])
  assert.equal(graphic.elements.length, 3)
  assert.equal(new Set(graphic.elements.map(element => element.id)).size, graphic.elements.length)
  assert.equal(graphic.elements[0].type, 'rect')
  assert.equal(graphic.elements.filter(element => element.type === 'text').length, 2)
})

test('graphic preview assets include only referenced project images and keys ignore unrelated media', () => {
  const graphic = {
    version: 1,
    width: 1344,
    height: 768,
    elements: [{ id: 'image', type: 'image', x: 0, y: 0, width: 100, height: 100, asset_id: 'brand' }],
  }
  const assets = {
    brand: imageAsset('brand'),
    unused: imageAsset('unused'),
    sound: { ...imageAsset('sound'), type: 'audio' },
  }
  const requestAssets = editorGraphicPreviewAssets(graphic, assets)
  assert.deepEqual(Object.keys(requestAssets), ['brand'])
  assert.equal(requestAssets.brand.path, 'project/brand.png')
  assert.equal(editorGraphicPreviewKey(graphic, 1920, 1080, assets, 'default'),
    editorGraphicPreviewKey(graphic, 1920, 1080, { brand: assets.brand }, 'default'))
  assert.notEqual(editorGraphicPreviewKey(graphic, 1920, 1080, assets, 'default'),
    editorGraphicPreviewKey(graphic, 1280, 720, assets, 'default'))
  const changedGraphic = structuredClone(graphic)
  changedGraphic.elements[0].width = 101
  assert.notEqual(editorGraphicPreviewKey(graphic, 1920, 1080, assets, 'default'),
    editorGraphicPreviewKey(changedGraphic, 1920, 1080, assets, 'default'))
})

test('graphic documents serialize with edits and participate in Editor undo and redo', () => {
  let nextId = 0
  const original = createEditorGraphic('lower-third', () => `lower-third-element-${++nextId}`)
  const project = textProject(original)
  useEditorStore.setState({ project, history: [], future: [], dirty: false, error: null })

  const edited = structuredClone(original)
  edited.elements[1].text = 'A serialized title'
  useEditorStore.getState().updateItem('graphic-item', { graphic: edited })

  const current = useEditorStore.getState().project
  assert.equal(current.tracks[0].items[0].graphic.elements[1].text, 'A serialized title')
  assert.equal(JSON.parse(JSON.stringify(current)).tracks[0].items[0].graphic.elements[0].type, 'rect')
  assert.equal(useEditorStore.getState().history.length, 1)

  useEditorStore.getState().undo()
  assert.equal(useEditorStore.getState().project.tracks[0].items[0].graphic.elements[1].text, 'YOUR TITLE')
  useEditorStore.getState().redo()
  assert.equal(useEditorStore.getState().project.tracks[0].items[0].graphic.elements[1].text, 'A serialized title')

  useEditorStore.setState({ project: textProject(original), history: [], future: [], dirty: false })
  useEditorStore.getState().addGraphicOverlay('full-title')
  const createdId = useEditorStore.getState().selectedItemId
  const created = useEditorStore.getState().project.tracks[0].items.find(item => item.id === createdId)
  assert.equal(created.graphic.version, 1)
  assert.equal(created.graphic.elements.length, 2)
  useEditorStore.getState().undo()
  assert.equal(useEditorStore.getState().project.tracks[0].items.some(item => item.id === createdId), false)
  useEditorStore.getState().redo()
  assert.equal(useEditorStore.getState().project.tracks[0].items.some(item => item.id === createdId), true)

  useEditorStore.setState({ project: null, history: [], future: [], dirty: false, error: null })
})
