// CPU-only contract tests for deterministic Editor auto-edit previews.
// No browser, API, media, or model calls are used.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const {createRequire} = require('node:module');
const {before, test} = require('node:test');
const root = path.resolve(__dirname, '../..');
const ui = path.join(root, 'ui');
const requireFromUi = createRequire(path.join(ui, 'package.json'));
const esbuild = requireFromUi('esbuild');

async function loadAutoEdit() {
  const sourceExtensions = ['.tsx', '.ts', '.jsx', '.js', '.mjs', '.cjs', '.json'];
  const resolveFile = candidate => {
    if (fs.existsSync(candidate) && fs.statSync(candidate).isFile()) return fs.realpathSync(candidate);
    for (const extension of sourceExtensions) {
      if (fs.existsSync(candidate + extension) && fs.statSync(candidate + extension).isFile()) {
        return fs.realpathSync(candidate + extension);
      }
    }
    for (const extension of sourceExtensions) {
      const indexFile = path.join(candidate, `index${extension}`);
      if (fs.existsSync(indexFile) && fs.statSync(indexFile).isFile()) return fs.realpathSync(indexFile);
    }
    return null;
  };
  const resolver = {
    name: 'workspace-filesystem-resolver',
    setup(build) {
      build.onResolve({filter: /.*/}, args => {
        if (args.path.startsWith('node:')) return {path: args.path, external: true};
        if (args.path.startsWith('.') || path.isAbsolute(args.path)) {
          const resolved = resolveFile(path.resolve(args.resolveDir, args.path));
          return resolved
            ? {path: resolved, namespace: 'workspace-fs'}
            : {errors: [{text: `Unable to resolve ${args.path} from ${args.resolveDir}`} ]};
        }
        const resolved = requireFromUi.resolve(args.path, {paths: [args.resolveDir, ui]});
        return {path: fs.realpathSync(resolved), namespace: 'workspace-fs'};
      });
      build.onLoad({filter: /.*/, namespace: 'workspace-fs'}, args => {
        const extension = path.extname(args.path);
        const loader = extension === '.tsx' ? 'tsx' : extension === '.ts' ? 'ts'
          : extension === '.jsx' ? 'jsx' : extension === '.json' ? 'json' : 'js';
        return {contents: fs.readFileSync(args.path, 'utf8'), loader, resolveDir: path.dirname(args.path)};
      });
    },
  };
  const result = await esbuild.build({
    stdin: {
      contents: "export { editorAutoEditSignature, previewEditorAutoEdit } from './src/editor/editorAutoEdit'",
      resolveDir: ui,
      loader: 'ts',
    },
    bundle: true,
    platform: 'node',
    format: 'cjs',
    write: false,
    target: 'node22',
    logLevel: 'silent',
    plugins: [resolver],
  });
  const module = {exports: {}};
  new Function('module', 'exports', 'require', result.outputFiles[0].text)(module, module.exports, require);
  return module.exports;
}

const asset = (overrides = {}) => ({
  id: 'video-asset', name: 'video.mp4', type: 'video', origin: 'upload',
  workspace: 'Workspace A', path: 'media/video.mp4', url: '/media/video.mp4',
  duration: 60, width: 1920, height: 1080, fps: 30, has_audio: true,
  ...overrides,
});

const clip = (id, start, duration, overrides = {}) => ({
  id, asset_id: 'video-asset', name: id, start, duration, source_in: 0,
  speed: 1, volume: 1, opacity: 1, fit: 'contain',
  transform: {x: 0, y: 0, scale: 1, rotation: 0},
  fade_in: 0, fade_out: 0,
  ...overrides,
});

function project(items = [clip('clip-a', 0, 2), clip('clip-b', 2, 2)], overrides = {}) {
  return {
    schema_version: 6,
    revision: 7,
    id: 'project-1',
    name: 'Fixture project',
    workspace: 'Workspace A',
    created_at: 10,
    updated_at: 20,
    canvas: {width: 1920, height: 1080, fps: 30, background: '#000000'},
    assets: {'video-asset': asset()},
    tracks: [
      {id: 'video-track', name: 'Video', type: 'video', z_index: 1, muted: false, locked: false, items},
      {id: 'audio-track', name: 'Audio', type: 'audio', z_index: 0, muted: false, locked: false, items: []},
    ],
    markers: [],
    export: {
      quality: 'balanced', codec: 'h264', encoder: 'auto', include_audio: true,
      resolution: 'canvas', frame_rate: 'project', filename: 'fixture.mp4',
      spatial_upsampling: '', film_grain_intensity: 0, film_grain_saturation: 0.5,
    },
    exports: [],
    ...overrides,
  };
}

function action(id, type, fields = {}) {
  return {id, type, ...fields};
}

test('strict server nullable fields are treated as absent during trim', () => {
  const result = preview(project(), [action('proposal-test-act-001', 'trim', {
    item_id:'clip-a',asset_id:null,track_id:null,start:null,duration:1,source_in:.5,volume:null,seconds:null,reason:null,
  })]);
  assert.equal(result.project.tracks[0].items[0].duration, 1);
  assert.equal(result.project.tracks[0].items[0].start, 0);
  assert.equal(result.project.tracks[0].items[0].source_in, .5);
});

function proposalFor(projectValue, actions, overrides = {}) {
  const autoEdit = globalThis.autoEdit;
  return {
    proposal_id: 'proposal-test',
    project_id: projectValue.id,
    workspace: projectValue.workspace,
    base_revision: projectValue.revision,
    client_signature: autoEdit.editorAutoEditSignature(projectValue),
    mode: 'assembly',
    status: 'proposed',
    summary: 'Fixture edit',
    actions,
    warnings: [],
    missing_media: [],
    ...overrides,
  };
}

function preview(projectValue, actions, selectedIDs = actions.map(entry => entry.id), overrides = {}) {
  const proposal = proposalFor(projectValue, actions, overrides);
  return globalThis.autoEdit.previewEditorAutoEdit(projectValue, proposal, selectedIDs);
}

before(async () => {
  globalThis.autoEdit = await loadAutoEdit();
});

test('signature ignores revision, timestamps, exports and transient missing state but includes media paths', () => {
  const first = project();
  first.assets['video-asset'].created_at = 15;
  first.tracks[0].items[0].ai_history = [{
    id: 'history-1', tool: 'video_to_video', asset_id: 'video-asset', created_at: 16,
  }];
  const second = structuredClone(first);
  second.revision = 99;
  second.created_at = -1;
  second.updated_at = 900;
  second.exports = [{id: 'export', filename: 'render.mp4'}];
  second.assets['video-asset'].missing = true;
  second.assets['video-asset'].created_at = 200;
  second.tracks[0].items[0].ai_history[0].created_at = 300;
  assert.equal(globalThis.autoEdit.editorAutoEditSignature(first), globalThis.autoEdit.editorAutoEditSignature(second));
  second.assets['video-asset'].path = 'media/replaced.mp4';
  assert.notEqual(globalThis.autoEdit.editorAutoEditSignature(first), globalThis.autoEdit.editorAutoEditSignature(second));
});

test('partial acceptance applies only selected action IDs and reports readable changes', () => {
  const original = project();
  const result = preview(original, [
    action('proposal-test-act-001', 'move', {item_id: 'clip-a', start: 4, reason: 'Place the opener after a title.'}),
    action('proposal-test-act-002', 'move', {item_id: 'clip-b', start: 0}),
  ], ['proposal-test-act-001']);
  assert.equal(result.project.tracks[0].items.find(item => item.id === 'clip-a').start, 4);
  assert.equal(result.project.tracks[0].items.find(item => item.id === 'clip-b').start, 2);
  assert.deepEqual(result.changes, ['Move “clip-a” to 4.00s on “Video”. Reason: Place the opener after a title.']);
  assert.equal(original.tracks[0].items[0].start, 0, 'preview leaves the source project untouched');
});

test('batch swaps are checked against final geometry rather than intermediate overlaps', () => {
  const result = preview(project(), [
    action('proposal-test-act-001', 'move', {item_id: 'clip-a', start: 2}),
    action('proposal-test-act-002', 'move', {item_id: 'clip-b', start: 0}),
  ]);
  assert.deepEqual(result.project.tracks[0].items.map(item => [item.id, item.start]), [['clip-b', 0], ['clip-a', 2]]);
});

test('proposal identity, revision, signature, unknown selected IDs and duplicate action IDs are guarded', () => {
  const base = project();
  const move = action('proposal-test-act-001', 'move', {item_id: 'clip-a', start: 4});
  const proposal = proposalFor(base, [move]);
  assert.throws(() => globalThis.autoEdit.previewEditorAutoEdit(base, {...proposal, project_id: 'other'}, [move.id]), /different project or workspace/);
  assert.throws(() => globalThis.autoEdit.previewEditorAutoEdit(base, {...proposal, base_revision: 6}, [move.id]), /revision changed/);
  assert.throws(() => globalThis.autoEdit.previewEditorAutoEdit(base, {...proposal, client_signature: 'old'}, [move.id]), /content changed/);
  assert.throws(() => globalThis.autoEdit.previewEditorAutoEdit(base, proposal, ['unknown-action']), /Selected proposal action/);
  assert.throws(() => globalThis.autoEdit.previewEditorAutoEdit(base, {...proposal, actions: [move, {...move}]}, [move.id]), /present and unique/);
  assert.throws(() => globalThis.autoEdit.previewEditorAutoEdit(base, proposal, []), /at least one proposed action/);
});

test('unknown target IDs and actions outside the proposal scope are rejected', () => {
  const base = project();
  assert.throws(() => preview(base, [action('proposal-test-act-001', 'move', {item_id: 'missing', start: 0})]), /unknown clip/);
  assert.throws(() => preview(base, [action('proposal-test-act-001', 'move', {item_id: 'clip-a', start: 0})], ['another-proposal-act']), /Selected proposal action/);
});

test('pinned and locked clips reject direct changes', () => {
  const pinned = project([clip('clip-a', 0, 2, {pinned: true}), clip('clip-b', 2, 2)]);
  assert.throws(() => preview(pinned, [action('proposal-test-act-001', 'remove', {item_id: 'clip-a'})]), /pinned clip/);
  const locked = project([clip('clip-a', 0, 2), clip('clip-b', 2, 2)]);
  locked.tracks[0].locked = true;
  assert.throws(() => preview(locked, [action('proposal-test-act-001', 'move', {item_id: 'clip-a', start: 4})]), /Unlock track/);
});

test('source bounds, finite time, 24-hour limit and trim keyframe/fade clamps are enforced', () => {
  const base = project([clip('clip-a', 0, 4, {
    fade_in: 3,
    fade_out: 3,
    keyframes: {x: [
      {time: 0, value: 0, easing: 'linear'},
      {time: 2, value: 20, easing: 'linear'},
      {time: 4, value: 40, easing: 'linear'},
    ]},
  })]);
  const trimmed = preview(base, [action('proposal-test-act-001', 'trim', {item_id: 'clip-a', duration: 2})]).project.tracks[0].items[0];
  assert.equal(trimmed.fade_in, 2);
  assert.equal(trimmed.fade_out, 2);
  assert.deepEqual(trimmed.keyframes.x.map(frame => frame.time), [0, 2]);
  assert.throws(() => preview(base, [action('proposal-test-act-001', 'trim', {item_id: 'clip-a', duration: 2, source_in: 59})]), /source media duration/);
  assert.throws(() => preview(base, [action('proposal-test-act-001', 'move', {item_id: 'clip-a', start: Infinity})]), /finite target start/);
  assert.throws(() => preview(base, [action('proposal-test-act-001', 'move', {item_id: 'clip-a', start: 86_399})]), /24-hour/);
});

test('trim preserves the interpolated animated value at the new clip end', () => {
  const base = project([clip('clip-a', 0, 4, {
    keyframes: {x: [
      {time: 0, value: 0, easing: 'linear'},
      {time: 4, value: 4, easing: 'linear'},
    ]},
  })]);
  const trimmed = preview(base, [action('proposal-test-act-001', 'trim', {item_id: 'clip-a', duration: 2})]).project.tracks[0].items[0];
  assert.deepEqual(trimmed.keyframes.x, [
    {time: 0, value: 0, easing: 'linear'},
    {time: 2, value: 2, easing: 'linear'},
  ]);
});

test('insert and trim require probed duration for non-image source media', () => {
  const unknownDuration = project([clip('clip-a', 0, 2)], {
    assets: {'video-asset': asset({duration: 0})},
  });
  assert.throws(() => preview(unknownDuration, [
    action('proposal-test-act-001', 'trim', {item_id: 'clip-a', duration: 1}),
  ]), /known source duration/);
  assert.throws(() => preview(unknownDuration, [
    action('proposal-test-act-001', 'insert', {asset_id: 'video-asset', track_id: 'video-track', start: 4, duration: 1}),
  ]), /known source duration/);
});

test('crossfade creates an accepted explicit dissolve and protects indirectly moved pinned clips', () => {
  const base = project([clip('clip-a', 0, 3), clip('clip-b', 3, 3), clip('clip-c', 6, 2)]);
  const result = preview(base, [action('proposal-test-act-001', 'crossfade', {item_id: 'clip-b', seconds: 0.5})]);
  const [previous, incoming, following] = result.project.tracks[0].items;
  assert.equal(incoming.transition_in, 'dissolve');
  assert.equal(incoming.transition_overlap, 0.5);
  assert.equal(incoming.start, 2.5);
  assert.equal(following.start, 5.5);
  assert.equal(previous.fade_out, 0);

  const pinnedFollowing = project([clip('clip-a', 0, 3), clip('clip-b', 3, 3), clip('clip-c', 6, 2, {pinned: true})]);
  assert.throws(() => preview(pinnedFollowing, [action('proposal-test-act-001', 'crossfade', {item_id: 'clip-b', seconds: 0.5})]), /pinned clip/);
});

test('gap closure protects pinned linked clips and linked geometry cannot be moved unsafely', () => {
  const base = project([clip('clip-a', 0, 1), clip('clip-b', 3, 1), clip('clip-c', 4, 1, {link_group_id: 'linked-1'})]);
  base.tracks[1].items.push(clip('linked-audio', 4, 1, {link_group_id: 'linked-1', pinned: true}));
  assert.throws(() => preview(base, [action('proposal-test-act-001', 'close-gap', {item_id: 'clip-b'})]), /pinned clip/);

  const linkedMove = project([clip('clip-a', 0, 1, {link_group_id: 'linked-1'}), clip('clip-b', 2, 1)]);
  assert.throws(() => preview(linkedMove, [action('proposal-test-act-001', 'move', {item_id: 'clip-a', start: 3})]), /linked clip geometry/);

  const pinnedTrack = project([clip('clip-a', 0, 1), clip('clip-b', 2, 1)]);
  pinnedTrack.tracks[0].pinned = true;
  assert.throws(() => preview(pinnedTrack, [action('proposal-test-act-001', 'move', {item_id: 'clip-a', start: 3})]), /pinned track/);
});

test('invalid final overlaps fail atomically while a valid explicit dissolve remains legal', () => {
  const original = project();
  const before = structuredClone(original);
  assert.throws(() => preview(original, [
    action('proposal-test-act-001', 'move', {item_id: 'clip-a', start: 1}),
    action('proposal-test-act-002', 'move', {item_id: 'clip-b', start: Infinity}),
  ]), /Action 2/);
  assert.deepEqual(original, before, 'a failed multi-action preview leaves the source untouched');

  const dissolved = project([clip('clip-a', 0, 3), clip('clip-b', 2.5, 3, {
    transition_in: 'dissolve', transition_overlap: 0.5, fade_in: 0.5,
  })]);
  assert.doesNotThrow(() => preview(dissolved, [action('proposal-test-act-001', 'set-volume', {item_id: 'clip-a', volume: 1.2})]));
  const invalid = project([clip('clip-a', 0, 3), clip('clip-b', 2.5, 3)]);
  assert.throws(() => preview(invalid, [action('proposal-test-act-001', 'set-volume', {item_id: 'clip-a', volume: 1.2})]), /overlap without a valid explicit dissolve/);
});

test('removing or moving away a dissolve predecessor rejects dangling transition metadata atomically', () => {
  const makeDissolve = () => project([clip('clip-a', 0, 3), clip('clip-b', 2.5, 3, {
    transition_in: 'dissolve', transition_overlap: 0.5, fade_in: 0.5,
  })]);

  const removedPredecessor = makeDissolve();
  const beforeRemove = structuredClone(removedPredecessor);
  assert.throws(() => preview(removedPredecessor, [
    action('proposal-test-act-001', 'remove', {item_id: 'clip-a'}),
  ]), /without a preceding video clip/);
  assert.deepEqual(removedPredecessor, beforeRemove, 'failed predecessor removal leaves the source project unchanged');

  const movedPredecessor = makeDissolve();
  const beforeMove = structuredClone(movedPredecessor);
  assert.throws(() => preview(movedPredecessor, [
    action('proposal-test-act-001', 'move', {item_id: 'clip-a', start: 10}),
  ]), /without a preceding video clip/);
  assert.deepEqual(movedPredecessor, beforeMove, 'failed predecessor move leaves the source project unchanged');

  const movedIncoming = makeDissolve();
  assert.throws(() => preview(movedIncoming, [
    action('proposal-test-act-001', 'move', {item_id: 'clip-b', start: 4}),
  ]), /no longer overlaps its predecessor/);

  const fadedPredecessor = project([clip('clip-a', 0, 3, {fade_out: 0.2}), clip('clip-b', 2.5, 3, {
    transition_in: 'dissolve', transition_overlap: 0.5, fade_in: 0.5,
  })]);
  assert.throws(() => preview(fadedPredecessor, [
    action('proposal-test-act-001', 'set-volume', {item_id: 'clip-a', volume: 1.2}),
  ]), /opaque predecessor with no fade-out/);

  const disabledPredecessor = project([clip('clip-a', 0, 3, {disabled: true}), clip('clip-b', 2.5, 3, {
    transition_in: 'dissolve', transition_overlap: 0.5, fade_in: 0.5,
  })]);
  assert.throws(() => preview(disabledPredecessor, [
    action('proposal-test-act-001', 'set-volume', {item_id: 'clip-a', volume: 1.2}),
  ]), /enabled predecessor/);
});

test('remove preserves original media and insert IDs are deterministic and unique', () => {
  const base = project();
  const removed = preview(base, [action('proposal-test-act-001', 'remove', {item_id: 'clip-a'})]).project;
  assert.ok(removed.assets['video-asset']);
  assert.equal(removed.tracks[0].items.some(item => item.id === 'clip-a'), false);

  const inserted = preview(base, [
    action('proposal-test-act-001', 'insert', {asset_id: 'video-asset', track_id: 'video-track', start: 4, duration: 1}),
    action('proposal-test-act-002', 'insert', {asset_id: 'video-asset', track_id: 'video-track', start: 5, duration: 1}),
  ]).project.tracks[0].items.filter(item => item.id.startsWith('auto-proposal-test-'));
  assert.equal(inserted.length, 2);
  assert.equal(new Set(inserted.map(item => item.id)).size, 2);
  assert.equal(inserted[0].asset_id, 'video-asset');
  assert.equal(inserted[1].asset_id, 'video-asset');
});
