// Exercise the real Editor round-trip setup against mocked HTTP. No generation
// endpoint is called; the test verifies durable intent creation and ownership.
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

const project = {
  schema_version: 1,
  revision: 3,
  id: 'editor-operation-project',
  name: 'Operation fixture',
  workspace: 'Operation fixtures',
  created_at: 1,
  updated_at: 1,
  canvas: {width: 1920, height: 1080, fps: 30, background: '#000000'},
  assets: {
    source: {
      id: 'source', name: 'source.mp4', type: 'video', origin: 'output',
      workspace: 'Operation fixtures', path: '/fixtures/source.mp4', url: '/source.mp4',
      duration: 12, width: 1280, height: 720, fps: 30, has_audio: true,
    },
  },
  tracks: [{
    id: 'video-track', name: 'Video 1', type: 'video', z_index: 1,
    muted: false, locked: false,
    items: [{
      id: 'clip-1', asset_id: 'source', name: 'Opening shot', start: 2,
      duration: 4, source_in: 1, speed: 1.25, volume: 1, opacity: 1,
      fit: 'contain', transform: {x: 0, y: 0, scale: 1, rotation: 0},
    }],
  }],
  markers: [],
  export: {
    quality: 'balanced', codec: 'h264', encoder: 'auto', include_audio: true,
    resolution: 'canvas', frame_rate: 'project', filename: 'edit.mp4',
    spatial_upsampling: '', film_grain_intensity: 0, film_grain_saturation: 0.5,
  },
  exports: [],
};
const projectSummary = {
  id: project.id, name: project.name, workspace: project.workspace,
  created_at: project.created_at, updated_at: project.updated_at,
  duration: 6, asset_count: 1,
};

async function bundleProbe() {
  const sourceExtensions = ['.tsx', '.ts', '.jsx', '.js', '.mjs', '.cjs', '.json'];
  const resolveFile = candidate => {
    if (require('node:fs').existsSync(candidate) && require('node:fs').statSync(candidate).isFile()) return require('node:fs').realpathSync(candidate);
    for (const extension of sourceExtensions) {
      const candidateWithExtension = candidate + extension;
      if (require('node:fs').existsSync(candidateWithExtension)) return require('node:fs').realpathSync(candidateWithExtension);
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
        return {path: require('node:fs').realpathSync(resolved), namespace: 'workspace-fs'};
      });
      build.onLoad({filter: /.*/, namespace: 'workspace-fs'}, args => {
        const extension = path.extname(args.path);
        const loader = extension === '.tsx' ? 'tsx' : extension === '.ts' ? 'ts'
          : extension === '.jsx' ? 'jsx' : extension === '.json' ? 'json' : 'js';
        return {contents: require('node:fs').readFileSync(args.path, 'utf8'), loader, resolveDir: path.dirname(args.path)};
      });
    },
  };
  const result = await esbuild.build({
    stdin: {
      contents: `
        import React from 'react';
        import {createRoot} from 'react-dom/client';
        import {useEditorStore} from './src/editor/useEditorStore';
        import {useStore} from './src/stores/useStore';
        window.editorStore = useEditorStore;
        window.store = useStore;
        function Probe() {
          const open = useEditorStore(state => state.aiWorkspaceOpen);
          const operation = useEditorStore(state => state.roundTrip);
          const mode = useStore(state => state.sidebarMode);
          return <div data-testid="probe" data-open={String(open)} data-mode={mode} data-operation={operation?.id || ''} />;
        }
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

(async () => {
  const browser = await chromium.launch({
    headless: true,
    ...(process.platform === 'win32' ? {
      executablePath: process.env.MAESTRO_CHROME || 'C:/Program Files/Google/Chrome/Application/chrome.exe',
    } : {}),
  });
  try {
    const page = await browser.newPage();
    const errors = [];
    const createdOperations = [];
    const createdProjectBodies = [];
    const projectReads = [];
    const savedProjects = new Map([[project.id, project]]);
    let projectSaveConflict = false;
    let generationRequests = 0;
    const generationBodies = [];
    let operationJob = null;
    page.on('pageerror', error => errors.push(error.message));
    await page.route('**/*', async route => {
      const url = new URL(route.request().url());
      if (url.pathname === '/') return route.fulfill({contentType: 'text/html', body: '<div id="root"></div>'});
      if (url.pathname === '/api/v1/editor/projects' && route.request().method() === 'POST') {
        const requestBody = JSON.parse(route.request().postData());
        createdProjectBodies.push(requestBody.project);
        const saved = {...requestBody.project, revision: 1};
        savedProjects.set(saved.id, saved);
        return route.fulfill({json: saved});
      }
      if (url.pathname.startsWith('/api/v1/editor/projects/') && route.request().method() === 'PUT') {
        if (projectSaveConflict) return route.fulfill({status: 409, json: {detail: 'stale project revision'}});
        const projectId = decodeURIComponent(url.pathname.split('/').at(-1));
        const requestBody = JSON.parse(route.request().postData());
        const saved = {...(savedProjects.get(projectId) || project), ...requestBody.project, revision: (savedProjects.get(projectId)?.revision || 0) + 1};
        savedProjects.set(projectId, saved);
        return route.fulfill({json: saved});
      }
      if (url.pathname.startsWith('/api/v1/editor/projects/') && route.request().method() === 'GET') {
        const projectId = decodeURIComponent(url.pathname.split('/').at(-1));
        projectReads.push(projectId);
        const saved = savedProjects.get(projectId);
        return saved ? route.fulfill({json: saved}) : route.fulfill({status: 404, json: {detail: 'Editor project not found'}});
      }
      if (url.pathname === '/api/v1/editor/projects') return route.fulfill({json: {projects: [projectSummary]}});
      if (url.pathname === '/api/v1/editor/operations' && route.request().method() === 'POST') {
        const record = JSON.parse(route.request().postData());
        createdOperations.push(record);
        return route.fulfill({json: {
          ...record, schema_version: 1, status: 'armed', job_id: null, job_kind: null,
          job_status: null, output_file: null, error: null, review_reason: null,
          created_at: 1, updated_at: 1,
        }});
      }
      if (url.pathname.startsWith('/api/v1/editor/operations/')) {
        const operationId = decodeURIComponent(url.pathname.split('/').at(-1));
        const record = createdOperations.find(item => item.operation_id === operationId);
        if (record) return route.fulfill({json: {
          ...record, schema_version: 1, status: operationJob?.status || 'armed',
          job_id: operationJob?.job_id || null, job_kind: operationJob?.job_kind || null,
          job_status: operationJob?.job_status || null,
          output_file: null, error: null, review_reason: null,
          created_at: 1, updated_at: 1,
        }});
      }
      if (url.pathname === '/api/v1/generate') { generationRequests += 1; generationBodies.push(route.request().postDataJSON()); return route.fulfill({json: {job_id: 'job-helper-image', status: 'queued'}}); }
      if (url.pathname.startsWith('/api/')) return route.fulfill({json: {assets: [], projects: [], outputs: [], jobs: []}});
      return route.fulfill({status: 204, body: ''});
    });
    await page.goto('http://editor-operation.test');
    await page.addScriptTag({content: await bundleProbe()});
    await page.evaluate(projectFixture => {
      window.store.setState({activeWorkspace: 'Operation fixtures', sidebarMode: 'editor'});
      window.editorStore.setState({
        workspace: 'Operation fixtures', project: projectFixture,
        selectedItemId: 'clip-1', selectedTrackId: 'video-track', dirty: false,
      });
      window.mountProbe();
    }, project);
    await page.evaluate(async () => {
      await window.editorStore.getState().beginAIRoundTrip('recast', 'replace');
    });
    await page.waitForFunction(() => document.querySelector('[data-testid="probe"]')?.dataset.open === 'true');
    const state = await page.evaluate(() => ({
      sidebarMode: window.store.getState().sidebarMode,
      operationId: window.editorStore.getState().roundTrip?.id,
      stored: JSON.parse(sessionStorage.getItem('maestro-editor-ai-round-trip-v1') || 'null'),
    }));
    assert.equal(state.sidebarMode, 'editor', 'the user remains in Editor while configuring the AI job');
    assert.ok(state.operationId && state.stored?.id === state.operationId, 'the operation id is durably kept in this browser session');
    assert.equal(createdOperations.length, 1, 'the backend received exactly one durable operation record');
    assert.equal(createdOperations[0].project_id, project.id);
    assert.equal(createdOperations[0].item_id, 'clip-1');
    assert.equal(createdOperations[0].tool, 'recast');
    assert.deepEqual(createdOperations[0].source_range, {start: 1, end: 6});
    assert.equal(createdOperations[0].client_submission_id, state.stored.clientSubmissionId);
    assert.equal(generationRequests, 0, 'opening the AI workspace does not submit or launch a job');

    // Recast can create helper reference images while the operation remains
    // armed. A generic image generation must not inherit its operation IDs.
    await page.evaluate(async () => {
      window.store.setState({
        generationMode: 'image',
        editSubMode: 'recast',
        studioImageWorkflow: 'generate',
        enabledModels: new Set(['fixture-image']),
        modelVisibilityOverrides: {},
        models: [{
          model_type: 'fixture-image', name: 'Fixture Image', family: 'fixture',
          architecture: 'fixture', is_i2v: false, is_t2v: false,
          guidance_max_phases: 1, fps: 24, nsfw_only: false,
        }],
        modelOptions: { model_type: 'fixture-image', architecture: 'fixture', fps: 24, t2v_class: true },
        servicesConfig: { nsfw_mode: false },
        params: { ...window.store.getState().params, model_type: 'fixture-image', prompt: 'Create a reference image', image_mode: 1 },
      });
      await window.store.getState().startGeneration();
    });
    assert.equal(generationRequests, 1, 'the helper image is submitted through generic generation');
    assert.equal(generationBodies.length, 1);
    assert.equal(generationBodies[0]._editor_operation_id, undefined, 'a helper image cannot claim the pending Recast operation');
    assert.equal(generationBodies[0]._client_submission_id?.length > 0, true, 'ordinary generation still has its own idempotency token');

    operationJob = {status: 'running', job_id: 'job-editor-42', job_kind: 'recast', job_status: 'running'};
    await page.reload();
    await page.addScriptTag({content: await bundleProbe()});
    await page.evaluate(async () => {
      window.store.setState({
        activeWorkspace: 'Operation fixtures', sidebarMode: 'editor',
        jobs: [{id: 'unrelated-studio-job', status: 'running', kind: 'generation'}],
      });
      await window.editorStore.getState().initialize('Operation fixtures');
      window.editorStore.setState({aiWorkspaceOpen: true});
      window.mountProbe();
    });
    await page.waitForFunction(() => window.editorStore.getState().roundTrip?.claimedJobId === 'job-editor-42');
    const recovered = await page.evaluate(() => ({
      operationId: window.editorStore.getState().roundTrip?.id,
      clientSubmissionId: window.editorStore.getState().roundTrip?.clientSubmissionId,
      claimedJobId: window.editorStore.getState().roundTrip?.claimedJobId,
      status: window.editorStore.getState().roundTrip?.status,
      sidebarMode: window.store.getState().sidebarMode,
    }));
    assert.equal(recovered.operationId, state.operationId, 'reload restores the same durable operation');
    assert.equal(recovered.clientSubmissionId, state.stored.clientSubmissionId);
    assert.equal(recovered.claimedJobId, 'job-editor-42', 'recovery follows the operation-linked job id, not an unrelated active job');
    assert.equal(recovered.status, 'running');
    assert.equal(recovered.sidebarMode, 'editor');
    assert.equal(createdOperations.length, 1, 'reload does not create a second operation');
    assert.equal(generationRequests, 1, 'reload does not launch a second helper or Editor job');

    // A failed source save must not block making a new version from the local
    // immutable snapshot, and project navigation must keep dirty work open.
    projectSaveConflict = true;
    const recovery = await page.evaluate(async () => {
      const originalOperationId = window.editorStore.getState().roundTrip?.id;
      window.editorStore.setState(state => ({
        project: {...state.project, name: 'Unsaved version snapshot'}, dirty: true,
      }));
      await window.editorStore.getState().duplicateProject();
      const duplicate = window.editorStore.getState().project;
      const duplicateId = duplicate?.id;
      window.editorStore.setState(state => ({
        project: {...state.project, name: 'Newer local edits'}, dirty: true,
      }));
      await window.editorStore.getState().loadProject('project-that-should-not-open');
      const afterBlockedNavigation = {
        projectId: window.editorStore.getState().project?.id,
        projectName: window.editorStore.getState().project?.name,
        dirty: window.editorStore.getState().dirty,
        error: window.editorStore.getState().error,
      };
      await window.editorStore.getState().loadProject(duplicateId, {discardUnsaved: true});
      return {
        duplicateId,
        duplicateDirty: window.editorStore.getState().dirty,
        afterBlockedNavigation,
        afterExplicitReload: {
          projectId: window.editorStore.getState().project?.id,
          projectName: window.editorStore.getState().project?.name,
          dirty: window.editorStore.getState().dirty,
        },
        roundTripId: window.editorStore.getState().roundTrip?.id,
        operationId: originalOperationId,
        aiWorkspaceOpen: window.editorStore.getState().aiWorkspaceOpen,
      };
    });
    assert.equal(createdProjectBodies.length, 1, 'duplicate creates a new project without requiring the stale source save');
    assert.match(createdProjectBodies[0].name, /^Unsaved version snapshot(?: v2)?$/, 'duplicate preserves the current local snapshot');
    assert.ok(recovery.duplicateId && recovery.duplicateId !== project.id);
    assert.equal(recovery.afterBlockedNavigation.projectId, recovery.duplicateId, 'failed save keeps the current project open');
    assert.equal(recovery.afterBlockedNavigation.projectName, 'Newer local edits');
    assert.equal(recovery.afterBlockedNavigation.dirty, true);
    assert.match(recovery.afterBlockedNavigation.error, /explicitly reload/i);
    assert.equal(projectReads.includes('project-that-should-not-open'), false, 'failed save prevents navigation fetch');
    assert.equal(recovery.afterExplicitReload.projectId, recovery.duplicateId);
    assert.equal(recovery.afterExplicitReload.projectName, createdProjectBodies[0].name, 'explicit same-project reload is the only path that discards local edits');
    assert.equal(recovery.afterExplicitReload.dirty, false);
    assert.equal(recovery.roundTripId, recovery.operationId, 'project recovery preserves Editor AI ownership state');
    assert.equal(recovery.aiWorkspaceOpen, true);
    assert.deepEqual(errors, [], `Editor operation setup raised browser errors: ${errors.join('; ')}`);
    console.log('Editor operation regression passed (exact job recovery, helper image isolation, stale-save versioning, and protected project navigation).');
  } finally {
    await browser.close();
  }
})().catch(error => {
  console.error(error);
  process.exitCode = 1;
});
