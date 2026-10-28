// Mocked browser regression for Editor export setup. No export or generation
// endpoint is called; submit and cancel behavior is exercised through store stubs.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const {createRequire} = require('node:module');
const root = path.resolve(__dirname, '../..');
const ui = path.join(root, 'ui');
const requireFromUi = createRequire(path.join(ui, 'package.json'));
const esbuild = requireFromUi('esbuild');
const uiNodeModules = fs.realpathSync(path.join(ui, 'node_modules'));
const playwrightPath = process.env.MAESTRO_PLAYWRIGHT ||
  'C:/Users/bliza/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright-core';
const {chromium} = require(playwrightPath);

const project = {
  schema_version: 5,
  id: 'export-dialog-project',
  name: 'Export dialog fixture',
  workspace: 'export-dialog-workspace',
  revision: 3,
  created_at: 1,
  updated_at: 2,
  canvas: {width: 1920, height: 1080, fps: 24, background: '#000000'},
  assets: {},
  tracks: [{
    id: 'video-1', name: 'Video', type: 'video', z_index: 1,
    muted: false, locked: false,
    items: [{
      id: 'clip-1', name: 'Clip', start: 0, duration: 8, source_in: 0,
      speed: 1, volume: 1, opacity: 1, fit: 'contain',
      transform: {x: 0, y: 0, scale: 1, rotation: 0},
    }],
  }],
  markers: [],
  export: {
    quality: 'draft', codec: 'h264', encoder: 'auto', include_audio: true,
    resolution: '720p', frame_rate: 'project', filename: 'review-copy',
    spatial_upsampling: 'flashvsr2', film_grain_intensity: 0.15,
    film_grain_saturation: 0.5,
  },
  exports: [],
};

async function bundleDialog() {
  const sourceExtensions = ['.tsx', '.ts', '.jsx', '.js', '.mjs', '.cjs', '.json'];
  const resolveFile = candidate => {
    if (fs.existsSync(candidate) && fs.statSync(candidate).isFile()) return fs.realpathSync(candidate);
    for (const extension of sourceExtensions) {
      if (fs.existsSync(candidate + extension)) return fs.realpathSync(candidate + extension);
      const indexFile = path.join(candidate, `index${extension}`);
      if (fs.existsSync(indexFile)) return fs.realpathSync(indexFile);
    }
    return null;
  };
  const mocks = {
    name: 'editor-export-store-mocks',
    setup(build) {
      build.onResolve({filter: /.*/}, args => {
        if (args.path.startsWith('node:')) return {path: args.path, external: true};
        if (args.path === './useEditorStore' || args.path === './src/editor/useEditorStore') {
          return {path: 'mock-editor-store', namespace: 'editor-export-mock'};
        }
        if (args.path === '../stores/useStore' || args.path === './src/stores/useStore') {
          return {path: 'mock-app-store', namespace: 'editor-export-mock'};
        }
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
      build.onLoad({filter: /.*/, namespace: 'editor-export-mock'}, args => {
        if (args.path === 'mock-editor-store') return {loader: 'js', contents: `
          import {create} from 'zustand';
          window.__exportSubmits = [];
          let sequence = 0;
          export const useEditorStore = create((set, get) => ({
            project: null, exportJobId: null, exportProgress: 0, error: null,
            setExportSettings: patch => set(state => ({project: {
              ...state.project,
              export: {...state.project.export, ...patch},
            }})),
            exportProject: async mode => {
              window.__exportSubmits.push(mode);
              const id = 'mock-export-' + (++sequence);
              set({exportJobId: id, exportProgress: 0.25, error: null});
              window.appStore.setState(state => ({jobs: [{
                id, kind: 'editor_export', status: mode === 'queue' ? 'held' : 'queued',
                progress: 0.25, step: 0, totalSteps: 0, phase: mode === 'queue' ? 'Queued' : 'Preparing',
                message: mode === 'queue' ? 'Editor export held' : 'Editor export queued',
                outputFiles: [], error: null,
              }, ...state.jobs.filter(job => job.id !== id)]}));
            },
          }));
        `};
        return {loader: 'js', contents: `
          import {create} from 'zustand';
          window.__cancelCalls = [];
          export const useStore = create((set, get) => ({
            jobs: [],
            stopGeneration: jobId => {
              window.__cancelCalls.push(jobId);
              set(state => ({jobs: state.jobs.map(job => job.id === jobId
                ? {...job, status: 'cancelled', phase: 'Cancelled', message: 'Cancelled'}
                : job)}));
            },
          }));
        `};
      });
    },
  };
  const result = await esbuild.build({
    absWorkingDir: path.dirname(uiNodeModules),
    stdin: {
      contents: `
        import React, {useState} from 'react';
        import {createRoot} from 'react-dom/client';
        import {EditorExportDialog} from './src/editor/EditorExportDialog';
        import {useEditorStore} from './src/editor/useEditorStore';
        import {useStore} from './src/stores/useStore';
        window.editorStore = useEditorStore;
        window.appStore = useStore;
        function App() {
          const [open, setOpen] = useState(false);
          return <><button type="button" onClick={() => setOpen(true)}>Open export settings</button>
            <EditorExportDialog open={open} onClose={() => setOpen(false)} /></>;
        }
        const root = createRoot(document.getElementById('root'));
        window.mount = () => root.render(<App />);
      `,
      resolveDir: ui,
      loader: 'tsx',
    },
    bundle: true,
    write: false,
    nodePaths: [uiNodeModules],
    jsx: 'automatic',
    define: {'process.env.NODE_ENV': '"development"'},
    logLevel: 'silent',
    plugins: [mocks],
  });
  return result.outputFiles[0].text;
}

function appCss() {
  const assets = path.join(ui, 'dist/assets');
  const cssFile = fs.readdirSync(assets).find(name => name.endsWith('.css'));
  assert.ok(cssFile, 'the UI CSS bundle is required for responsive layout checks');
  return fs.readFileSync(path.join(assets, cssFile), 'utf8');
}

(async () => {
  const browser = await chromium.launch({
    headless: true,
    ...(process.platform === 'win32' ? {executablePath: process.env.MAESTRO_CHROME || 'C:/Program Files/Google/Chrome/Application/chrome.exe'} : {}),
  });
  try {
    const page = await browser.newPage({viewport: {width: 1180, height: 900}});
    const apiRequests = [];
    let capabilityMode = 'error';
    const capabilities = {
      recommended: 'software',
      encoders: {software: true, nvidia: false, intel: false, apple: false},
    };
    page.on('pageerror', error => { throw error; });
    await page.route('**/*', async route => {
      const url = new URL(route.request().url());
      if (url.pathname === '/' || url.pathname === '/index.html') {
        return route.fulfill({contentType: 'text/html', body: '<div id="root"></div>'});
      }
      if (url.pathname === '/api/v1/editor/export/capabilities') {
        apiRequests.push(url.pathname);
        if (capabilityMode === 'error') {
          return route.fulfill({status: 503, contentType: 'application/json', body: JSON.stringify({detail: 'FFmpeg probe is offline'})});
        }
        return route.fulfill({contentType: 'application/json', body: JSON.stringify(capabilities)});
      }
      if (url.pathname.startsWith('/api/v1/')) {
        apiRequests.push(url.pathname);
        return route.fulfill({contentType: 'application/json', body: '{}'});
      }
      return route.fulfill({status: 204, body: ''});
    });
    await page.goto('http://editor-export-dialog.test');
    await page.addStyleTag({content: appCss()});
    await page.addScriptTag({content: await bundleDialog()});
    await page.evaluate(fixture => {
      window.editorStore.setState({project: fixture, exportJobId: null, exportProgress: 0, error: null});
      window.appStore.setState({jobs: [{
        id: 'foreign-project-export', kind: 'editor_export', status: 'completed',
        progress: 1, step: 0, totalSteps: 0, phase: 'Complete',
        message: 'Unrelated project export completed', outputFiles: [], error: null,
      }]});
      window.mount();
    }, project);

    assert.deepEqual(apiRequests, [], 'a closed dialog does not probe export capability or submit a job');
    await page.getByRole('button', {name: 'Open export settings'}).click();
    const dialog = page.getByRole('dialog', {name: 'Export video'});
    await dialog.waitFor();
    const initialJobStatus = await page.getByTestId('editor-export-job-status').innerText();
    assert.match(initialJobStatus, /project stays editable/);
    assert.doesNotMatch(initialJobStatus, /Unrelated project export completed/,
      'the dialog ignores newer Editor exports owned by another project');
    const capabilityStatus = page.getByTestId('editor-export-capability-status');
    await capabilityStatus.getByText(/Encoder readiness is unavailable/).waitFor();
    assert.match(await capabilityStatus.innerText(), /Unable to inspect export encoders/);
    assert.equal(await page.getByTestId('editor-export-submit').isDisabled(), false,
      'capability inspection errors are visible without blocking safe Auto/software submission');
    capabilityMode = 'ready';
    await capabilityStatus.getByRole('button', {name: 'Check again'}).click();
    await capabilityStatus.getByText(/Ready.*Software encoder/).waitFor();
    assert.deepEqual(await page.evaluate(() => window.__exportSubmits), [], 'opening export settings never starts an export');
    assert.equal(apiRequests.filter(pathname => pathname === '/api/v1/editor/export').length, 0,
      'no export submission endpoint is contacted while configuring options');

    const summary = await page.getByTestId('editor-export-summary').innerText();
    assert.match(summary, /Project canvas[\s\S]*1920×1080/);
    assert.match(summary, /Render frame[\s\S]*1280×720/);
    assert.match(summary, /Delivered frame[\s\S]*2560×1440/);
    assert.match(await page.getByTestId('editor-export-stage-summary').innerText(), /AI upscale 2× → Film grain/);

    await page.getByTestId('editor-export-preset-balanced').click();
    const presetState = await page.evaluate(() => window.editorStore.getState().project);
    assert.deepEqual({
      resolution: presetState.export.resolution,
      codec: presetState.export.codec,
      quality: presetState.export.quality,
    }, {resolution: '1080p', codec: 'h264', quality: 'balanced'});
    assert.equal(presetState.export.frame_rate, 'project', 'preset preserves frame-rate choice');
    assert.equal(presetState.export.encoder, 'auto', 'preset preserves encoder choice');
    assert.equal(presetState.export.spatial_upsampling, 'flashvsr2', 'preset preserves source-finishing choice');
    assert.equal(presetState.export.film_grain_intensity, 0.15, 'preset preserves grain settings');
    assert.equal(presetState.canvas.width, 1920, 'delivery presets do not alter the edit canvas');

    await page.evaluate(() => window.editorStore.setState(state => ({
      project: {...state.project, export: {...state.project.export, encoder: 'nvidia'}},
    })));
    await page.getByTestId('editor-export-unsupported-encoder').waitFor();
    assert.equal(await page.getByTestId('editor-export-submit').isDisabled(), true,
      'an explicitly selected encoder is disabled when the real capability probe reports it unavailable');
    await page.evaluate(() => window.editorStore.setState(state => ({
      project: {...state.project, export: {...state.project.export, encoder: 'auto'}},
    })));
    assert.equal(await page.getByTestId('editor-export-submit').isDisabled(), false);

    await page.getByTestId('editor-export-queue').click();
    await page.getByTestId('editor-export-job-status').getByText(/Waiting in the export queue/).waitFor();
    assert.deepEqual(await page.evaluate(() => window.__exportSubmits), ['queue']);
    assert.equal(await page.getByTestId('editor-export-cancel').isVisible(), true);
    await page.getByTestId('editor-export-cancel').click();
    assert.deepEqual(await page.evaluate(() => window.__cancelCalls), ['mock-export-1'],
      'cancellation is sent only for the displayed Editor export job');
    await page.getByTestId('editor-export-job-status').getByText(/Export cancelled/).waitFor();

    await page.evaluate(() => {
      window.appStore.setState(state => ({jobs: state.jobs.map(job => ({
        ...job, status: 'running', phase: 'Upscaling', message: 'Hardware export unavailable; retrying in software', progress: 0.62,
      }))}));
      window.editorStore.setState({exportJobId: 'mock-export-1', exportProgress: 0.62});
    });
    const stageStatus = await page.getByTestId('editor-export-job-status').innerText();
    assert.match(stageStatus, /Rendering project/);
    assert.match(stageStatus, /Hardware export unavailable; retrying in software/,
      'the existing software fallback stage is visible while the job continues');

    await page.setViewportSize({width: 390, height: 844});
    const mobileMetrics = await dialog.evaluate(element => ({
      width: element.getBoundingClientRect().width,
      scrollWidth: element.scrollWidth,
      clientWidth: element.clientWidth,
      height: element.getBoundingClientRect().height,
      viewportHeight: window.innerHeight,
    }));
    assert.ok(mobileMetrics.width <= 390, `dialog fits mobile width: ${JSON.stringify(mobileMetrics)}`);
    assert.ok(mobileMetrics.scrollWidth <= mobileMetrics.clientWidth + 1, `dialog has no horizontal overflow: ${JSON.stringify(mobileMetrics)}`);
    assert.ok(mobileMetrics.height <= mobileMetrics.viewportHeight, `dialog fits the mobile viewport with internal scrolling: ${JSON.stringify(mobileMetrics)}`);
    await page.getByTestId('editor-export-cancel').waitFor();

    assert.equal(apiRequests.filter(pathname => pathname === '/api/v1/editor/export/capabilities').length, 2,
      'capability probing happens on open and explicit retry, not as an export side effect');

    await page.keyboard.press('Escape');
    await page.getByRole('dialog', {name: 'Export video'}).waitFor({state: 'detached'});
    assert.deepEqual(await page.evaluate(() => window.__exportSubmits), ['queue'],
      'the only submitted job was the explicit queue-button action');
    console.log('Editor export dialog regression passed (presets, geometry, encoder readiness, stages, cancellation, responsive mobile layout, and explicit submission).');
  } finally {
    await browser.close();
  }
})().catch(error => {
  console.error(error);
  process.exitCode = 1;
});
