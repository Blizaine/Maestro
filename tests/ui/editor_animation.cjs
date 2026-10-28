// Run after the UI dependencies are installed: node tests/ui/editor_animation.cjs
const assert = require('node:assert/strict');
const path = require('node:path');
const esbuild = require(path.join(__dirname, '../../ui/node_modules/esbuild'));

function makeItem(overrides = {}) {
  return {
    id: 'animated-item', name: 'Animated item', start: 10, duration: 4,
    source_in: 0, speed: 1, volume: 0.75, opacity: 0.6, fit: 'contain',
    transform: {x: 24, y: -12, scale: 1.2, rotation: 5},
    ...overrides,
  };
}

function loadCommonJs(source) {
  const module = {exports: {}};
  new Function('module', 'exports', source)(module, module.exports);
  return module.exports;
}

(async () => {
  const uiRoot = path.resolve(__dirname, '../../ui');
  const nodeModules = path.join(uiRoot, 'node_modules');
  const result = await esbuild.build({
    absWorkingDir: uiRoot,
    entryPoints: [
      path.join(uiRoot, 'src/editor/editorAnimation.ts'),
      path.join(uiRoot, 'src/editor/editorColor.ts'),
    ],
    outbase: path.join(uiRoot, 'src/editor'),
    outdir: 'compiled-test',
    bundle: true,
    write: false,
    format: 'cjs',
    platform: 'node',
    jsx: 'automatic',
    nodePaths: [nodeModules],
    define: {'process.env.NODE_ENV': '"test"'},
    logLevel: 'silent',
  });
  const modules = Object.fromEntries(result.outputFiles.map(file => [path.basename(file.path), loadCommonJs(file.text)]));
  const animation = modules['editorAnimation.js'];
  const color = modules['editorColor.js'];

  const textItem = makeItem({
    text: 'A title',
    style: {x: 10, y: 20, font_size: 64, color: '#ffffff'},
    keyframes: {
      x: [{time: 0, value: 10, easing: 'ease-in'}, {time: 4, value: 50, easing: 'linear'}],
      y: [{time: 0, value: 20, easing: 'ease-out'}, {time: 4, value: 60, easing: 'linear'}],
      scale: [{time: 0, value: 1, easing: 'ease-in-out'}, {time: 4, value: 2, easing: 'linear'}],
      rotation: [{time: 0, value: 0, easing: 'linear'}, {time: 4, value: 80, easing: 'linear'}],
      opacity: [{time: 0, value: 0.5, easing: 'linear'}, {time: 4, value: 1, easing: 'linear'}],
      volume: [{time: 0, value: 0, easing: 'linear'}, {time: 4, value: 2, easing: 'linear'}],
    },
  });
  const authoredSnapshot = structuredClone(textItem);
  const halfway = animation.evaluateEditorItem(textItem, 12);
  assert.equal(halfway.transform.x, 20, 'Ease-in uses u² at the midpoint');
  assert.equal(halfway.transform.y, 50, 'Ease-out uses 1-(1-u)² at the midpoint');
  assert.equal(halfway.transform.scale, 1.5, 'Ease-in-out uses smoothstep at the midpoint');
  assert.equal(halfway.transform.rotation, 40, 'Linear rotation interpolates by clip-relative time');
  assert.equal(halfway.opacity, 0.75, 'Opacity interpolates');
  assert.equal(halfway.volume, 1, 'Volume interpolates');
  assert.equal(halfway.style.x, 20, 'Text style X follows the evaluated X property');
  assert.equal(halfway.style.y, 50, 'Text style Y follows the evaluated Y property');
  assert.deepEqual(textItem, authoredSnapshot, 'Evaluating an item does not mutate its authored data');
  assert.notEqual(halfway.transform, textItem.transform, 'Evaluated transform is a fresh object');
  assert.notEqual(halfway.style, textItem.style, 'Evaluated text style is a fresh object');

  const edgeCases = makeItem({keyframes: {
    x: [{time: 1, value: 7, easing: 'linear'}, {time: 3, value: 15, easing: 'linear'}],
    y: [{time: 0, value: 3, easing: 'linear'}, {time: 1, value: 8, easing: 'linear'}],
    opacity: [{time: 0, value: 0.2, easing: 'hold'}, {time: 2, value: 0.8, easing: 'linear'}, {time: 4, value: 1, easing: 'linear'}],
  }});
  assert.equal(animation.evaluateEditorProperty(edgeCases, 'x', 10), 7, 'Before the first key, the first key value is held');
  assert.equal(animation.evaluateEditorProperty(edgeCases, 'y', 14), 8, 'After the last key, the last key value is held');
  assert.equal(animation.evaluateEditorProperty(edgeCases, 'opacity', 11), 0.2, 'Hold easing keeps the left value before its next key');
  assert.equal(animation.evaluateEditorProperty(edgeCases, 'opacity', 12), 0.8, 'Hold easing switches exactly at the next key');
  assert.equal(animation.evaluateEditorProperty(edgeCases, 'opacity', 100), 1, 'Playback time clamps to the clip end');

  const easingCases = [
    ['linear', 5], ['hold', 0], ['ease-in', 2.5], ['ease-out', 7.5], ['ease-in-out', 5],
  ];
  for (const [easing, expected] of easingCases) {
    const fixture = makeItem({keyframes: {x: [
      {time: 0, value: 0, easing}, {time: 2, value: 10, easing: 'linear'},
    ]}});
    assert.equal(animation.evaluateEditorProperty(fixture, 'x', 11), expected,
      `${easing} easing is applied from the left keyframe`);
  }

  const staticItem = makeItem();
  const staticEvaluation = animation.evaluateEditorItem(staticItem, 12);
  assert.equal(staticEvaluation.transform.x, staticItem.transform.x, 'Missing keys fall back to authored transform values');
  assert.equal(staticEvaluation.opacity, staticItem.opacity, 'Missing opacity keys fall back to authored opacity');
  assert.equal(staticEvaluation.volume, staticItem.volume, 'Missing volume keys fall back to authored volume');

  const originalForEdit = makeItem();
  const originalSnapshot = structuredClone(originalForEdit);
  const seeded = animation.upsertEditorKeyframe(originalForEdit, 'x', 1, 300, 'ease-out');
  assert.deepEqual(seeded.keyframes.x.map(frame => frame.time), [0, 1], 'The first later key seeds the static value at time zero');
  assert.equal(seeded.keyframes.x[0].value, originalForEdit.transform.x, 'The seeded key preserves the original static value');
  assert.equal(seeded.keyframes.x[1].value, 300, 'The new key stores the requested value');
  assert.deepEqual(originalForEdit, originalSnapshot, 'Adding a keyframe is immutable for undo history');
  assert.notEqual(seeded.keyframes, originalForEdit.keyframes, 'Keyframe maps are replaced instead of mutated');

  const updated = animation.upsertEditorKeyframe(seeded, 'x', 1, -250);
  assert.equal(updated.keyframes.x.length, 2, 'Setting a key at the same time updates it instead of duplicating it');
  assert.equal(updated.keyframes.x[1].value, -250);
  assert.equal(updated.keyframes.x[1].easing, 'ease-out', 'Updating a value preserves its easing when omitted');
  const bounded = animation.upsertEditorKeyframe(updated, 'rotation', 99, 500, 'linear');
  assert.equal(bounded.keyframes.rotation.at(-1).time, 4, 'Key times clamp to the clip duration');
  assert.equal(bounded.keyframes.rotation.at(-1).value, 180, 'Rotation values stay within their supported range');
  const collision = makeItem({keyframes: {x: [
    {time: 0, value: 0, easing: 'linear'},
    {time: 1, value: 1, easing: 'linear'},
    {time: 2, value: 2, easing: 'linear'},
  ]}});
  const collisionSnapshot = structuredClone(collision);
  const merged = animation.updateEditorKeyframe(collision, 'x', 1, {time: 2, value: 12});
  assert.deepEqual(merged.keyframes.x.map(frame => frame.value), [0, 12], 'Moving a key onto another key merges without duplicates');
  assert.deepEqual(collision, collisionSnapshot, 'Updating a key preserves the prior undo snapshot');
  assert.notEqual(merged.keyframes, collision.keyframes, 'Updating a key replaces the keyframe map');
  assert.notEqual(merged.keyframes.x, collision.keyframes.x, 'Updating a key replaces its array');
  const deleted = animation.deleteEditorKeyframe(merged, 'x', 0);
  assert.equal(deleted.keyframes.x.length, 1);
  const cleared = animation.deleteEditorKeyframe(deleted, 'x', 0);
  assert.equal(cleared.keyframes, undefined, 'Deleting the last key removes the empty keyframe map');
  assert.equal(merged.keyframes.x.length, 2, 'Deleting keys leaves the previous item version intact');

  assert.deepEqual(animation.getApplicableEditorProperties('audio'), ['volume']);
  assert.deepEqual(animation.getApplicableEditorProperties('text'), ['x', 'y', 'scale', 'opacity']);
  assert.equal(animation.getApplicableEditorProperties('video', 'video', true).includes('volume'), true);
  assert.equal(animation.getApplicableEditorProperties('video', 'image', false).includes('volume'), false);

  assert.equal(color.editorColorFilter(), 'brightness(1.0000) contrast(1.0000) saturate(1.0000) url(#editor-temperature)');
  assert.equal(color.editorColorFilter({exposure: 1, contrast: 1.25, saturation: 0.8, temperature: 0.5}, 'clip-color-1'),
    'brightness(2.0000) contrast(1.2500) saturate(0.8000) url(#clip-color-1)');
  assert.equal(color.editorColorMatrix({exposure: 0, contrast: 1, saturation: 1, temperature: 1}),
    '1.2000 0 0 0 0  0 1 0 0 0  0 0 0.8000 0 0  0 0 0 1 0',
    'Warm temperature increases red gain and lowers blue gain');
  assert.equal(color.editorColorMatrix({exposure: 0, contrast: 1, saturation: 1, temperature: -1}),
    '0.8000 0 0 0 0  0 1 0 0 0  0 0 1.2000 0 0  0 0 0 1 0',
    'Cool temperature lowers red gain and increases blue gain');

  console.log('Editor keyframe evaluation, interpolation, immutable updates, bounds, property applicability, and color filter semantics passed');
})().catch(error => {
  console.error(error);
  process.exitCode = 1;
});
