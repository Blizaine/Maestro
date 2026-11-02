import assert from 'node:assert/strict'
import test from 'node:test'
import { getSidebarDialogPosition } from './sidebarDialogGeometry.ts'

const sidebar390 = { left: 0, right: 366, top: 0, bottom: 844, width: 366 }

test('mobile settings open upward near their footer control instead of at screen top', () => {
  assert.deepEqual(getSidebarDialogPosition({
    variant: 'settings', viewportTop: 0, viewportHeight: 844,
    windowWidth: 390, windowHeight: 844, sidebar: sidebar390,
    trigger: { left: 250, right: 350, top: 700, bottom: 740, width: 100 },
  }), { left: 8, bottom: 150, width: 350, maxHeight: 686 })
})

test('keyboard height and an offscreen anchor do not collapse mobile settings', () => {
  const position = getSidebarDialogPosition({
    variant: 'settings', viewportTop: 220, viewportHeight: 514,
    windowWidth: 390, windowHeight: 844, sidebar: sidebar390,
    // A long script has scrolled the settings trigger above the visible area.
    trigger: { left: 250, right: 350, top: -310, bottom: -270, width: 100 },
  })

  assert.deepEqual(position, { left: 8, bottom: 118, width: 350, maxHeight: 498 })
  assert.equal(844 - position.bottom, 726)
  assert.equal(220 + 514 - 8, 726)
})

test('viewport resize and visual viewport scrolling produce fresh mobile geometry', () => {
  const initial = getSidebarDialogPosition({
    variant: 'settings', viewportTop: 0, viewportHeight: 844,
    windowWidth: 390, windowHeight: 844, sidebar: sidebar390,
  })
  const afterKeyboardAndPan = getSidebarDialogPosition({
    variant: 'settings', viewportTop: 260, viewportHeight: 480,
    windowWidth: 390, windowHeight: 844, sidebar: sidebar390,
  })

  assert.deepEqual(initial, { left: 8, bottom: 8, width: 350, maxHeight: 828 })
  assert.deepEqual(afterKeyboardAndPan, { left: 8, bottom: 112, width: 350, maxHeight: 464 })
})

test('short 320x568 viewport still leaves a scrollable settings region above the keyboard', () => {
  const position = getSidebarDialogPosition({
    variant: 'settings', viewportTop: 300, viewportHeight: 238,
    windowWidth: 320, windowHeight: 568,
    sidebar: { left: 0, right: 300.8, top: 0, bottom: 568, width: 300.8 },
  })

  assert.deepEqual(position, { left: 8, bottom: 38, width: 284.8, maxHeight: 222 })
  assert.equal(568 - position.bottom, 530)
  assert.equal(300 + 238 - 8, 530)
})

test('touch landscape sidecar sizing ignores an offscreen anchor above 768px width', () => {
  const position = getSidebarDialogPosition({
    variant: 'settings', viewportTop: 130, viewportHeight: 180,
    windowWidth: 900, windowHeight: 320, mobile: true,
    sidebar: { left: 0, right: 380, top: 0, bottom: 320, width: 380 },
    trigger: { left: 250, right: 350, top: -90, bottom: -50, width: 100 },
  })

  assert.deepEqual(position, { left: 8, bottom: 18, width: 364, maxHeight: 164 })
  assert.equal(320 - position.bottom, 302)
  assert.equal(130 + 180 - 8, 302)
})

test('desktop settings retain their trigger-anchored placement', () => {
  assert.deepEqual(getSidebarDialogPosition({
    variant: 'settings', viewportTop: 0, viewportHeight: 900,
    windowWidth: 1280, windowHeight: 900,
    sidebar: { left: 0, right: 420, top: 0, bottom: 900, width: 420 },
    trigger: { left: 250, right: 350, top: 540, bottom: 580, width: 100 },
  }), { left: 12, bottom: 366, width: 360, maxHeight: 522 })
})

test('fixed-height mobile panels remain bounded by the visible viewport', () => {
  assert.deepEqual(getSidebarDialogPosition({
    variant: 'settings', viewportTop: 300, viewportHeight: 238,
    windowWidth: 320, windowHeight: 568,
    sidebar: { left: 0, right: 300.8, top: 0, bottom: 568, width: 300.8 },
    fixedHeight: 360,
  }), { left: 8, bottom: 38, width: 284.8, maxHeight: 222, height: 222 })
})

test('keyboard-sized mobile viewport prioritizes usable text-entry space over a crowded footer', () => {
  const position = getSidebarDialogPosition({
    variant: 'settings', viewportTop: 0, viewportHeight: 350,
    windowWidth: 390, windowHeight: 844, sidebar: sidebar390,
    trigger: { left: 250, right: 350, top: 218, bottom: 256, width: 100 },
  })
  assert.deepEqual(position, { left: 8, bottom: 502, width: 350, maxHeight: 334 })
  assert.equal(844 - position.bottom - position.maxHeight, 8)
})

test('mobile settings without an explicit trigger use the pinned settings row', () => {
  const position = getSidebarDialogPosition({
    variant: 'settings', viewportTop: 0, viewportHeight: 844,
    windowWidth: 390, windowHeight: 844, sidebar: sidebar390,
    settings: { top: 710 },
  })
  assert.deepEqual(position, { left: 8, bottom: 140, width: 350, maxHeight: 696 })
})

test('panned mobile viewport remains bounded when innerHeight also shrinks with the keyboard', () => {
  const position = getSidebarDialogPosition({
    variant: 'settings', viewportTop: 180, viewportHeight: 330,
    windowWidth: 390, windowHeight: 330, sidebar: sidebar390,
    trigger: { left: 250, right: 350, top: -10, bottom: 28, width: 100 },
  })
  assert.deepEqual(position, { left: 8, bottom: -172, width: 350, maxHeight: 314 })
  assert.equal(330 - position.bottom - position.maxHeight, 188)
  assert.equal(330 - position.bottom, 502)
})
