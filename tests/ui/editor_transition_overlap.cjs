// Regression coverage for Inspector edits around explicit timeline crossfades.
const assert = require('node:assert/strict');
const path = require('node:path');
const {createRequire} = require('node:module');
const root = path.resolve(__dirname, '../..');
const ui = path.join(root, 'ui');
const requireFromUi = createRequire(path.join(ui, 'package.json'));
const esbuild = requireFromUi('esbuild');
const playwrightPath = process.env.MAESTRO_PLAYWRIGHT ||
  'C:/Users/bliza/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright-core';
const {chromium} = require(playwrightPath);

async function bundleProbe() {
  const fs = require('node:fs');
  const sourceExtensions = ['.tsx', '.ts', '.jsx', '.js', '.mjs', '.cjs', '.json'];
  const resolveFile = candidate => {
    if (fs.existsSync(candidate) && fs.statSync(candidate).isFile()) return fs.realpathSync(candidate);
    for (const extension of sourceExtensions) {
      if (fs.existsSync(candidate + extension)) return fs.realpathSync(candidate + extension);
    }
    return null;
  };
  const fsResolver = {
    name: 'workspace-filesystem-resolver',
    setup(build) {
      build.onResolve({filter: /.*/}, args => {
        if (args.path.startsWith('node:')) return {path: args.path, external: true};
        if (args.path.startsWith('.') || path.isAbsolute(args.path)) {
          const resolved = resolveFile(path.resolve(args.resolveDir, args.path));
          return resolved ? {path: resolved, namespace: 'workspace-fs'}
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
      contents: `
        import React from 'react';
        import {createRoot} from 'react-dom/client';
        import {useEditorStore} from './src/editor/useEditorStore';
        window.editorStore = useEditorStore;
        function Probe() { return <div data-testid="probe" />; }
        window.mountProbe = () => createRoot(document.getElementById('root')).render(<Probe />);
      `,
      resolveDir: ui,
      loader: 'tsx',
    },
    bundle: true, write: false, format: 'iife', target: 'es2022', jsx: 'automatic',
    define: {'process.env.NODE_ENV': '"development"'}, logLevel: 'silent', plugins: [fsResolver],
  });
  return result.outputFiles[0].text;
}

const project = {
  schema_version: 1, revision: 1, id: 'crossfade-project', name: 'Crossfade fixture',
  workspace: 'Crossfade fixtures', created_at: 1, updated_at: 1,
  canvas: {width: 1920, height: 1080, fps: 30, background: '#000000'},
  assets: {
    a: {id: 'a', name: 'a.mp4', type: 'video', origin: 'output', workspace: 'Crossfade fixtures', path: '/fixtures/a.mp4', duration: 20, width: 1280, height: 720, fps: 30, has_audio: true},
    b: {id: 'b', name: 'b.mp4', type: 'video', origin: 'output', workspace: 'Crossfade fixtures', path: '/fixtures/b.mp4', duration: 20, width: 1280, height: 720, fps: 30, has_audio: true},
  },
  tracks: [{
    id: 'video-track', name: 'Video 1', type: 'video', z_index: 1, muted: false, locked: false,
    items: [
      {id: 'clip-a', asset_id: 'a', name: 'Clip A', start: 0, duration: 4, source_in: 0, speed: 1, volume: 1, opacity: 1, fit: 'contain', transform: {x: 0, y: 0, scale: 1, rotation: 0}, fade_in: 0, fade_out: 0, transition_in: 'none', transition_out: 'none'},
      {id: 'clip-b', asset_id: 'b', name: 'Clip B', start: 3, duration: 3, source_in: 0, speed: 1, volume: 1, opacity: 1, fit: 'contain', transform: {x: 0, y: 0, scale: 1, rotation: 0}, fade_in: 1, fade_out: 0, transition_in: 'dissolve', transition_out: 'none', transition_overlap: 1},
      {id: 'clip-c', asset_id: 'a', name: 'Clip C', start: 8, duration: 2, source_in: 0, speed: 1, volume: 1, opacity: 1, fit: 'contain', transform: {x: 0, y: 0, scale: 1, rotation: 0}, fade_in: 0, fade_out: 0, transition_in: 'none', transition_out: 'none'},
    ],
  }],
  markers: [],
  export: {quality: 'balanced', codec: 'h264', encoder: 'auto', include_audio: true, resolution: 'canvas', frame_rate: 'project', filename: 'edit.mp4', spatial_upsampling: '', film_grain_intensity: 0, film_grain_saturation: 0.5},
  exports: [],
};

(async () => {
  const browser = await chromium.launch({
    headless: true,
    ...(process.platform === 'win32' ? {executablePath: process.env.MAESTRO_CHROME || 'C:/Program Files/Google/Chrome/Application/chrome.exe'} : {}),
  });
  try {
    const page = await browser.newPage();
    const errors = [];
    page.on('pageerror', error => errors.push(error.message));
    await page.route('**/*', async route => {
      const url = new URL(route.request().url());
      if (url.pathname === '/') return route.fulfill({contentType: 'text/html', body: '<div id="root"></div>'});
      if (url.pathname.startsWith('/api/')) return route.fulfill({json: {assets: [], projects: [], outputs: [], jobs: []}});
      return route.fulfill({status: 204, body: ''});
    });
    await page.goto('http://editor-crossfade.test');
    await page.addScriptTag({content: await bundleProbe()});
    await page.evaluate(fixture => {
      window.editorStore.setState({project: fixture, dirty: false, history: [], future: []});
      window.mountProbe();
    }, project);

    await page.evaluate(() => {
      const store = window.editorStore.getState();
      if (!store.executeEditOperations([{type: 'crossfade', itemId: 'clip-b', seconds: 1}])) throw new Error('could not prepare explicit crossfade');
      store.updateItem('clip-b', {fade_in: 10, transition_in: 'dissolve'});
    });
    let state = await page.evaluate(() => {
      const doc = window.editorStore.getState().project;
      return {items: doc.tracks[0].items, error: window.editorStore.getState().error};
    });
    let clipA = state.items.find(item => item.id === 'clip-a');
    let clipB = state.items.find(item => item.id === 'clip-b');
    let clipC = state.items.find(item => item.id === 'clip-c');
    assert.equal(state.error, null);
    assert.equal(clipB.transition_overlap, 3, 'fade duration is clamped to both clips');
    assert.equal(clipB.fade_in, 3);
    assert.equal(clipB.start, 1, 'changing overlap repositions the clip and following timeline');
    assert.equal(clipC.start, 6, 'following clips move with the overlap edit');
    assert.equal(clipA.fade_out, 0);

    await page.evaluate(() => window.editorStore.getState().updateItem('clip-b', {
      transition_in: 'fade_black', fade_in: 0.5,
    }));
    state = await page.evaluate(() => ({items: window.editorStore.getState().project.tracks[0].items, error: window.editorStore.getState().error}));
    clipB = state.items.find(item => item.id === 'clip-b');
    clipC = state.items.find(item => item.id === 'clip-c');
    assert.equal(state.error, null);
    assert.equal(clipB.transition_overlap, 0, 'switching away from dissolve clears the explicit overlap');
    assert.equal(clipB.start, 4, 'the former overlap returns to a butt cut before applying fade black');
    assert.equal(clipB.transition_in, 'fade_black');
    assert.equal(clipB.fade_in, 0.5);
    assert.equal(clipC.start, 9, 'the following timeline preserves its pre-overlap gap when the butt cut is restored');

    await page.evaluate(fixture => {
      window.editorStore.setState({project: structuredClone(fixture), dirty: false, history: [], future: []});
      const store = window.editorStore.getState();
      store.executeEditOperations([{type: 'crossfade', itemId: 'clip-b', seconds: 1}]);
      store.updateItem('clip-b', {duration: 2});
    }, project);
    state = await page.evaluate(() => ({items: window.editorStore.getState().project.tracks[0].items, error: window.editorStore.getState().error}));
    clipB = state.items.find(item => item.id === 'clip-b');
    clipC = state.items.find(item => item.id === 'clip-c');
    assert.equal(state.error, null);
    assert.equal(clipB.duration, 2);
    assert.equal(clipB.transition_overlap, 0, 'trimming clears stale overlap metadata');
    assert.equal(clipB.transition_in, 'none');
    assert.equal(clipB.start, 4);
    assert.equal(clipC.start, 9, 'trim leaves subsequent clips on a valid butt cut');

    await page.evaluate(fixture => {
      window.editorStore.setState({project: structuredClone(fixture), dirty: false, history: [], future: []});
      const store = window.editorStore.getState();
      store.executeEditOperations([{type: 'crossfade', itemId: 'clip-b', seconds: 1}]);
      store.updateItem('clip-b', {start: 5});
    }, project);
    state = await page.evaluate(() => ({items: window.editorStore.getState().project.tracks[0].items, error: window.editorStore.getState().error}));
    clipB = state.items.find(item => item.id === 'clip-b');
    assert.equal(state.error, null);
    assert.equal(clipB.start, 5, 'manual move is applied after overlap geometry is cleared');
    assert.equal(clipB.transition_overlap, 0, 'moving a clip clears stale overlap metadata');
    assert.equal(clipB.transition_in, 'none');

    await page.evaluate(fixture => {
      window.editorStore.setState({project: structuredClone(fixture), dirty: false, history: [], future: []});
      window.editorStore.getState().executeEditOperations([{type: 'crossfade', itemId: 'clip-b', seconds: 1}]);
      window.editorStore.getState().updateItem('clip-a', {duration: 3});
    }, project);
    state = await page.evaluate(() => ({items: window.editorStore.getState().project.tracks[0].items, error: window.editorStore.getState().error}));
    clipA = state.items.find(item => item.id === 'clip-a');
    clipB = state.items.find(item => item.id === 'clip-b');
    assert.equal(state.error, null);
    assert.equal(clipA.duration, 3);
    assert.equal(clipB.transition_overlap, 0, 'trimming the preceding clip clears its outgoing overlap');
    assert.equal(clipB.start, 4);

    // A malformed linked overlap is surfaced as store error and does not throw
    // through the UI event handler or partially mutate project state.
    const linked = structuredClone(project);
    linked.tracks[0].items[0].link_group_id = 'linked-a';
    linked.tracks[0].items[1].link_group_id = 'linked-b';
    await page.evaluate(fixture => {
      window.editorStore.setState({project: fixture, dirty: false, history: [], future: []});
      window.editorStore.getState().updateItem('clip-b', {fade_in: 0.5});
    }, linked);
    state = await page.evaluate(() => ({items: window.editorStore.getState().project.tracks[0].items, error: window.editorStore.getState().error}));
    assert.match(state.error || '', /unlink these clips/i);
    assert.equal(state.items.find(item => item.id === 'clip-b').start, 3);
    assert.equal(state.items.find(item => item.id === 'clip-b').transition_overlap, 1);
    assert.deepEqual(errors, [], `Editor updateItem escaped errors: ${errors.join('; ')}`);
    console.log('Editor crossfade regression passed (duration coupling, transition change, move/trim invalidation, and caught edit failures).');
  } finally {
    await browser.close();
  }
})().catch(error => {
  console.error(error);
  process.exitCode = 1;
});
