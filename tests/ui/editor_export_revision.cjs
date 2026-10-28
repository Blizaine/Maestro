// Mocked browser regression for Editor export/save revision ordering.
// Uses the real Editor store and workspace; no media or generation requests run.
// Run `npm run build` from ui/ first, then `node tests/ui/editor_export_revision.cjs`.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const {createRequire} = require('node:module');
const root = path.resolve(__dirname, '../..');
const ui = path.join(root, 'ui');
const requireFromUi = createRequire(path.join(ui, 'package.json'));
const esbuild = requireFromUi('esbuild');
const playwrightPath = process.env.MAESTRO_PLAYWRIGHT ||
  'C:/Users/bliza/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright-core';
const {chromium} = require(playwrightPath);

const WORKSPACE = 'Export revision fixtures';
const PROJECT_ID = 'editor-export-revision-project';
const project = {
  schema_version: 6,
  revision: 1,
  id: PROJECT_ID,
  name: 'Export revision fixture',
  workspace: WORKSPACE,
  created_at: 1,
  updated_at: 1,
  canvas: {width: 1920, height: 1080, fps: 30, background: '#000000'},
  assets: {},
  tracks: [{
    id: 'titles-1', name: 'Titles', type: 'text', z_index: 1,
    muted: false, locked: false,
    items: [{
      id: 'clip-1', name: 'Initial title', start: 0, duration: 8, source_in: 0,
      speed: 1, volume: 1, opacity: 1, fit: 'contain',
      transform: {x: 0, y: 0, scale: 1, rotation: 0},
      text: 'Revision fixture',
      style: {
        x: 0, y: 0, font_size: 48, font_family: 'Arial', color: '#ffffff',
        background_color: '#000000', background_opacity: 0, text_align: 'center',
      },
    }],
  }],
  markers: [],
  export: {
    quality: 'balanced', codec: 'h264', encoder: 'auto', include_audio: true,
    resolution: 'canvas', frame_rate: 'project', filename: 'revision-test.mp4',
    spatial_upsampling: '', film_grain_intensity: 0, film_grain_saturation: 0.5,
  },
  exports: [],
};

const clone = value => JSON.parse(JSON.stringify(value));
const wait = ms => new Promise(resolve => setTimeout(resolve, ms));
function deferred() {
  let resolve;
  const promise = new Promise(done => { resolve = done; });
  return {promise, resolve};
}
async function until(predicate, label, timeoutMs = 12_000) {
  const deadline = Date.now() + timeoutMs;
  while (Date.now() < deadline) {
    if (predicate()) return;
    await wait(25);
  }
  assert.fail(`Timed out waiting for ${label}`);
}

function appCss() {
  const assets = path.join(ui, 'dist/assets');
  const cssFile = fs.readdirSync(assets).find(name => name.endsWith('.css'));
  assert.ok(cssFile, 'build the UI before running the mocked browser check');
  return fs.readFileSync(path.join(assets, cssFile), 'utf8');
}

async function bundleEditor() {
  const sourceExtensions = ['.tsx', '.ts', '.jsx', '.js', '.mjs', '.cjs', '.json'];
  const resolveFile = candidate => {
    if (fs.existsSync(candidate) && fs.statSync(candidate).isFile()) return fs.realpathSync(candidate);
    for (const extension of sourceExtensions) {
      if (fs.existsSync(candidate + extension)) return fs.realpathSync(candidate + extension);
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
          : extension === '.jsx' ? 'jsx' : extension === '.json' ? 'json'
            : extension === '.css' ? 'css' : 'js';
        return {contents: fs.readFileSync(args.path, 'utf8'), loader, resolveDir: path.dirname(args.path)};
      });
    },
  };
  const result = await esbuild.build({
    stdin: {
      contents: `
        import {createRoot} from 'react-dom/client';
        import {EditorWorkspace} from './src/editor/EditorWorkspace';
        import {useStore} from './src/stores/useStore';
        import {useEditorStore} from './src/editor/useEditorStore';
        window.appStore = useStore;
        window.editorStore = useEditorStore;
        window.mountEditor = () => createRoot(document.getElementById('root')).render(<EditorWorkspace/>);
      `,
      resolveDir: ui,
      loader: 'tsx',
    },
    bundle: true,
    write: false,
    format: 'iife',
    target: 'es2022',
    jsx: 'automatic',
    define: {'process.env.NODE_ENV': '"development"'},
    logLevel: 'silent',
    plugins: [resolver],
  });
  return result.outputFiles[0].text;
}

(async () => {
  const browser = await chromium.launch({
    headless: true,
    ...(process.platform === 'win32' ? {
      executablePath: process.env.MAESTRO_CHROME || 'C:/Program Files/Google/Chrome/Application/chrome.exe',
    } : {}),
  });
  try {
    const page = await browser.newPage({viewport: {width: 1440, height: 960}});
    const pageErrors = [];
    const requests = [];
    const saves = [];
    const exportRequests = [];
    const completion = new Set();
    let serverProject = clone(project);
    let nextCompletionGate = null;
    const firstExportSeen = deferred();
    const firstExportResponse = deferred();
    const firstCompletionFetchSeen = deferred();
    const firstCompletionResponse = deferred();
    const secondCompletionFetchSeen = deferred();
    const secondCompletionResponse = deferred();

    const summary = () => ({
      id: serverProject.id,
      name: serverProject.name,
      workspace: serverProject.workspace,
      created_at: serverProject.created_at,
      updated_at: serverProject.updated_at,
      duration: 8,
      asset_count: 0,
    });
    const json = (route, body, status = 200) => route.fulfill({status, json: body});

    page.on('pageerror', error => pageErrors.push(error.message));
    await page.route('**/*', async route => {
      const request = route.request();
      const url = new URL(request.url());
      const pathname = url.pathname;
      if (pathname.startsWith('/api/')) requests.push({method: request.method(), pathname});
      if (pathname === '/' || pathname === '/index.html') {
        return route.fulfill({contentType: 'text/html', body: '<div id="root" style="height:100vh;width:100vw"></div>'});
      }
      if (pathname === '/api/v1/workspaces') {
        return json(route, {workspaces: [{name: WORKSPACE, path: '/mock-workspace'}], active: WORKSPACE});
      }
      if (pathname === '/api/v1/editor/projects' && request.method() === 'GET') {
        return json(route, {projects: [summary()]});
      }
      if (pathname === `/api/v1/editor/projects/${PROJECT_ID}` && request.method() === 'GET') {
        if (nextCompletionGate) {
          const gate = nextCompletionGate;
          nextCompletionGate = null;
          gate.seen.resolve();
          await gate.response.promise;
        }
        return json(route, clone(serverProject));
      }
      if (pathname === `/api/v1/editor/projects/${PROJECT_ID}` && request.method() === 'PUT') {
        const body = request.postDataJSON();
        const submitted = body.project;
        if (submitted.revision !== serverProject.revision) {
          return json(route, {detail: 'This project changed on another device.'}, 409);
        }
        serverProject = {
          ...clone(submitted),
          revision: serverProject.revision + 1,
          updated_at: serverProject.updated_at + 1,
        };
        saves.push({revision: submitted.revision, itemName: submitted.tracks[0].items[0].name});
        return json(route, clone(serverProject));
      }
      if (pathname === '/api/v1/editor/media/status') return json(route, {assets: []});
      if (pathname === '/api/v1/outputs') return json(route, {outputs: [], total: 0});
      if (pathname === '/api/v1/director/pipelines') return json(route, {pipelines: []});
      if (pathname === '/api/v1/editor/export' && request.method() === 'POST') {
        const body = request.postDataJSON();
        const jobId = `mock-editor-export-${exportRequests.length + 1}`;
        exportRequests.push({
          jobId,
          revision: body.project.revision,
          queueMode: body.queue_mode,
          itemName: body.project.tracks[0].items[0].name,
        });
        if (exportRequests.length === 1) {
          firstExportSeen.resolve();
          await firstExportResponse.promise;
        }
        return json(route, {job_id: jobId, status: 'held', revision: body.project.revision});
      }
      if (pathname.startsWith('/api/v1/status/')) {
        const jobId = pathname.split('/').at(-1);
        return json(route, {
          job_id: jobId,
          kind: 'editor_export',
          status: completion.has(jobId) ? 'completed' : 'held',
          progress: completion.has(jobId) ? 100 : 0,
          step: 0,
          total_steps: 0,
          phase: completion.has(jobId) ? 'Complete' : 'Held',
          message: completion.has(jobId) ? 'Export complete' : 'Export held',
          output_files: completion.has(jobId) ? [`${jobId}.mp4`] : [],
          error: null,
        });
      }
      if (pathname.startsWith('/api/')) {
        return json(route, {items: [], outputs: [], jobs: [], pipelines: [], projects: [], status: 'available'});
      }
      return route.fulfill({status: 204, body: ''});
    });

    console.log('Building isolated Editor bundle...');
    const editorBundle = await bundleEditor();
    console.log('Launching mocked Editor workspace...');
    await page.goto('http://editor-export-revision.test');
    await page.addStyleTag({content: appCss()});
    await page.addScriptTag({content: editorBundle});
    await page.evaluate(workspace => {
      window.appStore.setState({activeWorkspace: workspace, sidebarOpen: false});
      window.mountEditor();
    }, WORKSPACE);
    await page.waitForFunction(id => (
      window.editorStore?.getState().project?.id === id
      && window.editorStore.getState().loading === false
    ), PROJECT_ID);
    console.log('Workspace initialized.');

    const renameClip = name => page.evaluate(nextName => {
      window.editorStore.getState().updateItem('clip-1', {name: nextName});
    }, name);
    const current = () => page.evaluate(() => {
      const state = window.editorStore.getState();
      return {
        revision: state.project?.revision,
        workspace: state.project?.workspace,
        dirty: state.dirty,
        saving: state.saving,
        itemName: state.project?.tracks[0].items[0].name,
        error: state.error,
        exportJobId: state.exportJobId,
        exports: state.project?.exports,
      };
    });

    await renameClip('Saved before export');
    await until(() => saves.length === 1, 'first debounced autosave');
    await page.waitForFunction(() => !window.editorStore.getState().dirty && !window.editorStore.getState().saving);
    assert.equal((await current()).revision, 2);
    console.log('Initial autosave passed.');

    await page.evaluate(() => { void window.editorStore.getState().exportProject('queue'); });
    await firstExportSeen.promise;
    console.log('First export request reached the mocked API.');
    assert.deepEqual(exportRequests.map(item => item.queueMode), ['held']);
    assert.equal(exportRequests[0].revision, 2, 'export queues the exact autosaved revision');
    assert.equal(saves.length, 1, 'a clean project is not saved a second time just to export');

    // The export request is accepted by the server but its response is delayed.
    // Autosave edits made during that gap must keep their newer revision.
    await renameClip('Autosaved while queued');
    await until(() => saves.length === 2, 'autosave while export response is pending');
    await page.waitForFunction(() => !window.editorStore.getState().dirty && !window.editorStore.getState().saving);
    assert.equal(serverProject.revision, 3);
    console.log('Autosave while export response was pending passed.');
    firstExportResponse.resolve();
    await page.waitForFunction(() => window.editorStore.getState().exportJobId === 'mock-editor-export-1');
    assert.equal((await current()).revision, 3, 'late export response cannot downgrade a newer autosave');

    // The worker writes only its export-history record to current revision 3.
    // Keep the detail response pending while the user makes a newer local edit.
    serverProject = {
      ...serverProject,
      revision: 4,
      updated_at: serverProject.updated_at + 1,
      exports: [{
        id: 'export-mock-editor-export-1', filename: 'mock-editor-export-1.mp4',
        workspace: WORKSPACE, created_at: 2, duration: 8, width: 1920, height: 1080,
        fps: 30, codec: 'h264', quality: 'balanced', encoder: 'auto',
        spatial_upsampling: '', film_grain_intensity: 0, film_grain_saturation: 0.5,
      }],
    };
    nextCompletionGate = {seen: firstCompletionFetchSeen, response: firstCompletionResponse};
    completion.add('mock-editor-export-1');
    await firstCompletionFetchSeen.promise;
    console.log('First export completion fetch reached the mock.');
    await renameClip('Newer local edit during completion');
    firstCompletionResponse.resolve();
    await page.waitForFunction(() => (
      window.editorStore.getState().project?.exports?.some(entry => entry.id === 'export-mock-editor-export-1')
      && window.editorStore.getState().project?.tracks[0].items[0].name === 'Newer local edit during completion'
    ));
    const afterFirstCompletion = await current();
    assert.equal(afterFirstCompletion.revision, 4, 'record-only server revision can be carried forward');
    assert.equal(afterFirstCompletion.dirty, true, 'completion does not clear newer unsaved edits');
    assert.equal(afterFirstCompletion.itemName, 'Newer local edit during completion');
    await until(() => saves.length === 3, 'autosave after export completion');
    await page.waitForFunction(() => !window.editorStore.getState().dirty && !window.editorStore.getState().saving);
    assert.equal(serverProject.revision, 5);
    console.log('Completion merge and follow-up save passed.');

    // A genuinely external timeline edit plus the export-history write causes
    // a two-revision gap. The Editor must keep its old revision so next save
    // detects the conflict instead of overwriting the external document.
    await page.evaluate(() => { void window.editorStore.getState().exportProject('queue'); });
    await until(() => exportRequests.length === 2, 'second export queue');
    console.log('Second export queued.');
    assert.equal(exportRequests[1].revision, 5);
    serverProject = clone(serverProject);
    serverProject.revision = 6;
    serverProject.tracks[0].items[0].name = 'External timeline edit';
    serverProject.updated_at += 1;
    serverProject.revision = 7;
    serverProject.exports = [{
      id: 'export-mock-editor-export-2', filename: 'mock-editor-export-2.mp4',
      workspace: WORKSPACE, created_at: 3, duration: 8, width: 1920, height: 1080,
      fps: 30, codec: 'h264', quality: 'balanced', encoder: 'auto',
      spatial_upsampling: '', film_grain_intensity: 0, film_grain_saturation: 0.5,
    }, ...serverProject.exports];
    nextCompletionGate = {seen: secondCompletionFetchSeen, response: secondCompletionResponse};
    completion.add('mock-editor-export-2');
    await secondCompletionFetchSeen.promise;
    console.log('External-change completion fetch reached the mock.');
    secondCompletionResponse.resolve();
    await page.waitForFunction(() => window.editorStore.getState().project?.exports?.some(
      entry => entry.id === 'export-mock-editor-export-2',
    ));
    const afterExternalCompletion = await current();
    assert.equal(afterExternalCompletion.revision, 5, 'external timeline revision is not adopted');
    assert.equal(afterExternalCompletion.itemName, 'Newer local edit during completion');
    assert.ok(requests.every(entry => !entry.pathname.includes('/generate') && !entry.pathname.includes('/upload')),
      'the fixture sends no generation or upload request');
    assert.deepEqual(pageErrors, [], `browser errors: ${pageErrors.join('; ')}`);
    console.log(`Editor export revision regression passed (${saves.length} autosaves, ${exportRequests.length} queued exports, no media/generation requests).`);
  } finally {
    await browser.close();
  }
})().catch(error => {
  console.error(error);
  process.exitCode = 1;
});
