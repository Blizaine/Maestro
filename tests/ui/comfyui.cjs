// Run: node tests/ui/comfyui.cjs
// Checks the typed ComfyUI API payloads and the workflow dialog against mocked
// browser requests. No live ComfyUI server, GPU, model files, or generation is used.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const root = path.resolve(__dirname, '../..');
const esbuild = require(path.join(root, 'ui/node_modules/esbuild'));
let chromium = null;
let playwrightUnavailable = null;
try { ({chromium} = require(process.env.MAESTRO_PLAYWRIGHT || 'playwright')); }
catch (error) { playwrightUnavailable = error; }

async function compileApi() {
  const filename = path.join(root, 'ui/src/api/comfyui.ts');
  const built = await esbuild.build({
    stdin: {contents: fs.readFileSync(filename, 'utf8'), resolveDir: path.dirname(filename), sourcefile: filename, loader: 'ts'},
    bundle: false,
    platform: 'node',
    format: 'cjs',
    write: false,
    target: 'node18',
  });
  const module = {exports: {}};
  new Function('module', 'exports', 'require', built.outputFiles[0].text)(module, module.exports, require);
  return module.exports;
}

function response(payload, status = 200) {
  return {ok: status >= 200 && status < 300, status, json: async () => payload};
}

async function checkApiPayloads() {
  const api = await compileApi();
  const requests = [];
  const originalFetch = global.fetch;
  global.fetch = async (url, init = {}) => {
    requests.push({url: String(url), method: init.method || 'GET', body: init.body ? JSON.parse(init.body) : null});
    return response({
      enabled: false, endpoint: 'http://127.0.0.1:8188', connected: false,
      installations: [], workflows: [], templates: [],
      linked_paths: [], roots: [], files: [], truncated: false,
      valid: true, errors: [], missing_nodes: [], missing_models: [],
      deleted: true, abandoned: true, job_id: 'comfy-test-job', status: 'queued',
    });
  };
  try {
    await api.fetchComfyUIStatus();
    await api.saveComfyUISettings({enabled: true, endpoint: 'http://comfy.lan:8188', dedicated: true});
    await api.testComfyUIConnection();
    await api.discoverComfyUIConnections();
    await api.discoverComfyUIModels();
    await api.discoverComfyUIModels(['D:/ComfyUI/models']);
    await api.scanComfyUIModels(['D:/ComfyUI/models/checkpoints']);
    await api.linkComfyUIModels(['D:/ComfyUI/models/checkpoints']);
    await api.fetchComfyUIWorkflows();
    const workflow = {name: 'Starter', kind: 'image', workflow: {6: {class_type: 'CLIPTextEncode'}}, bindings: []};
    await api.createComfyUIWorkflow(workflow);
    await api.deleteComfyUIWorkflow('workflow / 1');
    await api.fetchComfyUIWorkflowOptions('workflow / 1');
    await api.validateComfyUIWorkflow('workflow / 1');
    const run = {
      client_submission_id: 'd90b76aa-9267-4cc2-961f-c97f2ce90467',
      inputs: {prompt: 'blue fox', steps: 24, enabled: true},
      workspace: 'selected-workspace',
    };
    await api.runComfyUIWorkflow('workflow / 1', run);
    await api.getComfyUIJobBySubmission(run.client_submission_id);
    await api.resumeComfyUIJobBySubmission(run.client_submission_id);
    await api.abandonComfyUISubmission(run.client_submission_id);

    assert.deepEqual(requests, [
      {url: '/api/v1/comfyui/status', method: 'GET', body: null},
      {url: '/api/v1/comfyui/settings', method: 'PUT', body: {enabled: true, endpoint: 'http://comfy.lan:8188', dedicated: true}},
      {url: '/api/v1/comfyui/connection/test', method: 'POST', body: {}},
      {url: '/api/v1/comfyui/connection/discover', method: 'POST', body: {}},
      {url: '/api/v1/comfyui/models/discover', method: 'POST', body: {}},
      {url: '/api/v1/comfyui/models/discover', method: 'POST', body: {paths: ['D:/ComfyUI/models']}},
      {url: '/api/v1/comfyui/models/scan', method: 'POST', body: {paths: ['D:/ComfyUI/models/checkpoints']}},
      {url: '/api/v1/comfyui/models/link', method: 'POST', body: {paths: ['D:/ComfyUI/models/checkpoints']}},
      {url: '/api/v1/comfyui/workflows', method: 'GET', body: null},
      {url: '/api/v1/comfyui/workflows', method: 'POST', body: workflow},
      {url: '/api/v1/comfyui/workflows/workflow%20%2F%201', method: 'DELETE', body: null},
      {url: '/api/v1/comfyui/workflows/workflow%20%2F%201/options', method: 'GET', body: null},
      {url: '/api/v1/comfyui/workflows/workflow%20%2F%201/validate', method: 'POST', body: {inputs: {}}},
      {url: '/api/v1/comfyui/workflows/workflow%20%2F%201/run', method: 'POST', body: run},
      {url: '/api/v1/comfyui/jobs/by-submission/d90b76aa-9267-4cc2-961f-c97f2ce90467', method: 'GET', body: null},
      {url: '/api/v1/comfyui/jobs/by-submission/d90b76aa-9267-4cc2-961f-c97f2ce90467/resume', method: 'POST', body: {}},
      {url: '/api/v1/comfyui/jobs/by-submission/d90b76aa-9267-4cc2-961f-c97f2ce90467/abandon', method: 'POST', body: {}},
    ], 'Each API method sends the documented route, method, and exact request body');
  } finally { global.fetch = originalFetch; }
}

async function buildDialogBundle() {
  const source = [
    "import React from 'react';",
    "import {createRoot} from 'react-dom/client';",
    "import {ComfyUIWorkflowDialog} from './src/components/Sidebar/ComfyUIWorkflowDialog';",
    'window.mountComfyUIDialog = () => createRoot(document.getElementById("root")).render(',
    '  React.createElement(ComfyUIWorkflowDialog, {open: true, workspace: "selected-workspace", onClose: () => {}}),',
    ');',
    'window.mountComfyUIDialog();',
  ].join('\n');
  const result = await esbuild.build({
    stdin: {contents: source, sourcefile: path.join(root, 'ui/comfyui-dialog-test.tsx'), resolveDir: path.join(root, 'ui'), loader: 'tsx'},
    bundle: true,
    write: false,
    jsx: 'automatic',
    define: {'process.env.NODE_ENV': '"development"'},
    logLevel: 'silent',
    plugins: [{
      name: 'mock-zustand-store',
      setup(build) {
        build.onResolve({filter: /stores\/useStore/}, () => ({path: 'fake-use-store', namespace: 'test-store'}));
        build.onLoad({filter: /.*/, namespace: 'test-store'}, () => ({
          contents: 'export const useStore = selector => selector({activeWorkspace: "selected-workspace", reconnectJobs: async () => {}});',
          loader: 'js',
        }));
      },
    }],
  });
  return result.outputFiles[0].text;
}

async function buildWorkflowHostBundle() {
  const source = [
    "import React from 'react';",
    "import {createRoot} from 'react-dom/client';",
    "import {ComfyUIWorkflowDialogHost} from './src/components/Sidebar/ComfyUIWorkflowDialog';",
    "import {requestOpenComfyUIWorkflow} from './src/api/comfyui';",
    'window.mountWorkflowHost = () => createRoot(document.getElementById("root")).render(',
    '  React.createElement(React.Fragment, null,',
    '    React.createElement("button", {onClick: () => requestOpenComfyUIWorkflow("workflow-owned", {prompt: "restored gallery prompt"})}, "Use ComfyUI settings"),',
    '    React.createElement(ComfyUIWorkflowDialogHost),',
    '  ),',
    ');',
    'window.mountWorkflowHost();',
  ].join('\n');
  const result = await esbuild.build({
    stdin: {contents: source, sourcefile: path.join(root, 'ui/comfyui-host-test.tsx'), resolveDir: path.join(root, 'ui'), loader: 'tsx'},
    bundle: true,
    write: false,
    jsx: 'automatic',
    define: {'process.env.NODE_ENV': '"development"'},
    logLevel: 'silent',
    plugins: [{
      name: 'mock-zustand-store',
      setup(build) {
        build.onResolve({filter: /stores\/useStore/}, () => ({path: 'fake-use-store', namespace: 'test-store'}));
        build.onLoad({filter: /.*/, namespace: 'test-store'}, () => ({
          contents: 'export const useStore = selector => selector({activeWorkspace: "selected-workspace", reconnectJobs: async () => {}});',
          loader: 'js',
        }));
      },
    }],
  });
  return result.outputFiles[0].text;
}

async function buildSettingsBundle() {
  const source = [
    "import React from 'react';",
    "import {createRoot} from 'react-dom/client';",
    "import {ComfyUISettingsPanel} from './src/components/SettingsDrawer/ComfyUISettingsPanel';",
    'window.mountSettings = () => createRoot(document.getElementById("root")).render(',
    '  React.createElement(ComfyUISettingsPanel),',
    ');',
    'window.mountSettings();',
  ].join('\n');
  const result = await esbuild.build({
    stdin: {contents: source, sourcefile: path.join(root, 'ui/comfyui-settings-test.tsx'), resolveDir: path.join(root, 'ui'), loader: 'tsx'},
    bundle: true,
    write: false,
    jsx: 'automatic',
    define: {'process.env.NODE_ENV': '"development"'},
    logLevel: 'silent',
    plugins: [{
      name: 'mock-zustand-store',
      setup(build) {
        build.onResolve({filter: /stores\/useStore/}, () => ({path: 'fake-use-store', namespace: 'test-store'}));
        build.onLoad({filter: /.*/, namespace: 'test-store'}, () => ({
          contents: 'export const useStore = selector => selector({activeWorkspace: "selected-workspace", reconnectJobs: async () => {}});',
          loader: 'js',
        }));
      },
    }],
  });
  return result.outputFiles[0].text;
}

function jsonResponse(route, value, status = 200) {
  return route.fulfill({status, contentType: 'application/json', body: JSON.stringify(value)});
}

async function checkDialogWorkflow(browser) {
  if (!chromium) throw playwrightUnavailable || new Error('Playwright is unavailable');
  const bundle = await buildDialogBundle();
  const state = {requests: [], runs: [], receipts: [], resumes: [], imports: [], validations: []};
    const page = await browser.newPage({viewport: {width: 390, height: 844}});
    const pageErrors = [];
    page.on('pageerror', error => pageErrors.push(error.message));
    const existingWorkflow = {
      id: 'workflow-owned', name: 'Prompted Image', kind: 'image',
      workflow: {6: {class_type: 'CLIPTextEncode', inputs: {text: 'old prompt'}}},
      bindings: [
        {name: 'prompt', label: 'Prompt', node_id: '6', input: 'text', type: 'string', required: true},
        {name: 'checkpoint', label: 'Checkpoint', node_id: '7', input: 'ckpt_name', type: 'string'},
      ],
    };
    const importedWorkflow = {
      id: 'workflow-imported', name: 'Imported Video', kind: 'video', workflow: {1: {class_type: 'SaveVideo'}}, bindings: [],
    };
    const savedWorkflows = [existingWorkflow];
    await page.route('**/*', async route => {
      const request = route.request();
      const url = new URL(request.url());
      if (url.origin !== 'http://maestro.test') return route.abort('blockedbyclient');
      state.requests.push({path: url.pathname, method: request.method()});
      if (url.pathname === '/' && request.method() === 'GET') {
        return route.fulfill({contentType: 'text/html', body: '<!doctype html><html><body><div id="root"></div><script src="/dialog.js"></script></body></html>'});
      }
      if (url.pathname === '/dialog.js' && request.method() === 'GET') return route.fulfill({contentType: 'text/javascript', body: bundle});
      if (url.pathname === '/api/v1/comfyui/workflows' && request.method() === 'GET') {
        return jsonResponse(route, {workflows: savedWorkflows, templates: []});
      }
      if (url.pathname === '/api/v1/comfyui/workflows/workflow-owned/options' && request.method() === 'GET') {
        return jsonResponse(route, {workflow_id: 'workflow-owned', choices: {checkpoint: ['model-a.safetensors', 'model-b.safetensors']}});
      }
      if (url.pathname.endsWith('/options') && request.method() === 'GET') {
        const id = url.pathname.split('/').slice(-2, -1)[0];
        return jsonResponse(route, {workflow_id: id, choices: {}});
      }
      if (url.pathname === '/api/v1/comfyui/workflows' && request.method() === 'POST') {
        const body = request.postDataJSON();
        state.imports.push(body);
        savedWorkflows.unshift(importedWorkflow);
        return jsonResponse(route, importedWorkflow);
      }
      if (url.pathname === '/api/v1/comfyui/workflows/workflow-imported/validate' && request.method() === 'POST') {
        state.validations.push({workflow_id: 'workflow-imported', body: request.postDataJSON()});
        return jsonResponse(route, {valid: true, errors: [], missing_nodes: [], missing_models: []});
      }
      if (url.pathname === '/api/v1/comfyui/workflows/workflow-owned/validate' && request.method() === 'POST') {
        state.validations.push({workflow_id: 'workflow-owned', body: request.postDataJSON()});
        return jsonResponse(route, {valid: true, errors: [], missing_nodes: [], missing_models: []});
      }
      if (url.pathname === '/api/v1/comfyui/workflows/workflow-owned/run' && request.method() === 'POST') {
        state.runs.push(request.postDataJSON());
        // Simulate an accepted run whose POST response was lost. The UI must
        // reconcile with the saved submission UUID and never retry the POST.
        return route.abort('failed');
      }
      const submission = url.pathname.match(/^\/api\/v1\/comfyui\/jobs\/by-submission\/([^/]+)$/);
      if (submission && request.method() === 'GET') {
        const id = decodeURIComponent(submission[1]);
        state.receipts.push(id);
        return jsonResponse(route, {client_submission_id: id, job_id: 'comfy-job-owned', status: 'uncertain', state: 'uncertain'});
      }
      const resumed = url.pathname.match(/^\/api\/v1\/comfyui\/jobs\/by-submission\/([^/]+)\/resume$/);
      if (resumed && request.method() === 'POST') {
        state.resumes.push({id: decodeURIComponent(resumed[1]), body: request.postDataJSON()});
        return jsonResponse(route, {client_submission_id: decodeURIComponent(resumed[1]), job_id: 'comfy-job-owned', status: 'queued'});
      }
      return jsonResponse(route, {detail: `Unexpected request ${request.method()} ${url.pathname}`}, 404);
    });
    await page.goto('http://maestro.test/');
    await page.getByRole('dialog', {name: 'ComfyUI workflow library'}).waitFor({state: 'visible'});
    await page.getByText('Ready to run').waitFor({state: 'visible'});
    await page.waitForFunction(() => Array.from(document.querySelectorAll('datalist')).some(list =>
      Array.from(list.querySelectorAll('option')).some(option => option.value === 'model-b.safetensors')));
    assert.ok(state.requests.some(item => item.path === '/api/v1/comfyui/workflows/workflow-owned/options' && item.method === 'GET'),
      'Opening a workflow loads choices only for that selection');
    assert.equal(state.runs.length, 0, 'Opening a workflow and completing preflight does not submit a run');

    await page.getByLabel('Name').fill('Imported Video');
    await page.getByLabel('Output kind').selectOption('video');
    await page.getByLabel('Workflow JSON').fill('{"1":{"class_type":"SaveVideo"}}');
    await page.getByLabel('Input bindings JSON').fill('[]');
    await page.getByRole('button', {name: 'Import workflow'}).click();
    await page.getByRole('heading', {name: 'Imported Video'}).waitFor({state: 'visible'});
    assert.deepEqual(state.imports, [{name: 'Imported Video', kind: 'video', workflow: {1: {class_type: 'SaveVideo'}}, bindings: []}],
      'Import saves the API graph and typed bindings with the selected output kind');
    assert.equal(state.runs.length, 0, 'Importing a graph never auto-submits generation');

    await page.getByRole('button', {name: /Prompted Image/}).click();
    await page.getByText('Ready to run').waitFor({state: 'visible'});
    await page.getByRole('textbox', {name: /Prompt/}).fill('a blue fox under stars');
    await page.getByText('Ready to run').waitFor({state: 'visible'});
    assert.ok(state.validations.some(item => item.workflow_id === 'workflow-owned' && item.body.inputs?.prompt === 'a blue fox under stars'),
      'Preflight validates the current typed values');
    assert.equal(state.runs.length, 0, 'Editing typed workflow inputs does not submit a run');
    await page.getByRole('button', {name: 'Generate'}).click();
    await page.getByText('Owned ComfyUI job: comfy-job-owned').waitFor({state: 'visible', timeout: 5000});
    assert.equal(state.runs.length, 1, 'The explicit Generate action submits exactly one run request');
    assert.match(state.runs[0].client_submission_id, /^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i);
    assert.deepEqual({inputs: state.runs[0].inputs, workspace: state.runs[0].workspace}, {
      inputs: {prompt: 'a blue fox under stars'}, workspace: 'selected-workspace',
    }, 'The typed prompt and selected current workspace are sent to the selected workflow');
    assert.deepEqual(state.receipts, [state.runs[0].client_submission_id], 'The lost POST response is resolved by GET using its submission UUID');
    await page.getByRole('button', {name: 'Resume monitoring'}).click();
    await page.getByText('Owned ComfyUI job: comfy-job-owned').waitFor({state: 'visible'});
    assert.deepEqual(state.resumes, [{id: state.runs[0].client_submission_id, body: {}}], 'Resume monitoring POST uses the owned UUID and empty body');
    assert.equal(state.runs.length, 1, 'Resuming monitoring never posts a second workflow run');
    assert.deepEqual(pageErrors, [], 'The workflow dialog has no browser runtime errors');
    const dialog = page.getByRole('dialog');
    assert.ok(await dialog.evaluate(node => node.scrollWidth <= innerWidth), 'The modal fits a narrow viewport horizontally');
  await page.close();
}

async function checkWorkflowHost(browser) {
  const bundle = await buildWorkflowHostBundle();
  const page = await browser.newPage({viewport: {width: 390, height: 844}});
  const state = {requests: [], validations: [], runs: []};
  const workflow = {
    id: 'workflow-owned', name: 'Prompted Image', kind: 'image',
    workflow: {6: {class_type: 'CLIPTextEncode', inputs: {text: 'old prompt'}}},
    bindings: [{name: 'prompt', label: 'Prompt', node_id: '6', input: 'text', type: 'string', required: true}],
  };
  await page.route('**/*', async route => {
    const request = route.request();
    const url = new URL(request.url());
    if (url.origin !== 'http://maestro.test') return route.abort('blockedbyclient');
    if (url.pathname === '/' && request.method() === 'GET') return route.fulfill({contentType: 'text/html', body: '<!doctype html><html><body><div id="root"></div><script src="/host.js"></script></body></html>'});
    if (url.pathname === '/host.js' && request.method() === 'GET') return route.fulfill({contentType: 'text/javascript', body: bundle});
    state.requests.push({path: url.pathname, method: request.method(), body: request.postDataJSON?.() ?? null});
    if (url.pathname === '/api/v1/comfyui/workflows' && request.method() === 'GET') return jsonResponse(route, {workflows: [workflow], templates: []});
    if (url.pathname === '/api/v1/comfyui/workflows/workflow-owned/validate' && request.method() === 'POST') {
      state.validations.push(request.postDataJSON());
      return jsonResponse(route, {valid: true, errors: [], missing_nodes: [], missing_models: []});
    }
    if (url.pathname.endsWith('/run') && request.method() === 'POST') {
      state.runs.push(request.postDataJSON());
      return jsonResponse(route, {job_id: 'unexpected-run'});
    }
    return jsonResponse(route, {detail: 'Unexpected request ' + request.method() + ' ' + url.pathname}, 404);
  });
  await page.goto('http://maestro.test/');
  await page.getByRole('button', {name: 'Use ComfyUI settings'}).click();
  await page.getByRole('dialog', {name: 'ComfyUI workflow library'}).waitFor({state: 'visible'});
  assert.equal(await page.getByRole('textbox', {name: /Prompt/}).inputValue(), 'restored gallery prompt', 'The gallery handoff selects the saved workflow and restores its saved inputs');
  await page.getByText('Ready to run').waitFor({state: 'visible'});
  assert.ok(state.validations.some(body => body.inputs?.prompt === 'restored gallery prompt'), 'The restored settings receive current preflight');
  assert.equal(state.runs.length, 0, 'Opening restored settings never submits generation');
  await page.close();
}

async function checkSettingsAndDedicatedToggle(browser) {
  const bundle = await buildSettingsBundle();
  const page = await browser.newPage({viewport: {width: 430, height: 900}});
  const state = {requests: [], settings: [], tests: [], resumes: [], abandons: [], cancellations: []};
  const pending = [
    {submission_id: 'd90b76aa-9267-4cc2-961f-c97f2ce90467', state: 'prepared', job_id: null},
    {submission_id: 'a90b76aa-9267-4cc2-961f-c97f2ce90468', state: 'prepared', job_id: 'prepared-maestro-job'},
    {submission_id: 'b90b76aa-9267-4cc2-961f-c97f2ce90469', state: 'uncertain', job_id: 'uncertain-maestro-job', error: 'Needs monitoring'},
  ];
  let config = {enabled: false, dedicated: false, endpoint: 'http://127.0.0.1:8188', connected: false, pending_submissions: pending};
  await page.route('**/*', async route => {
    const request = route.request();
    const url = new URL(request.url());
    if (url.origin !== 'http://maestro.test') return route.abort('blockedbyclient');
    if (url.pathname === '/' && request.method() === 'GET') return route.fulfill({contentType: 'text/html', body: '<!doctype html><html><body><div id="root"></div><script src="/settings.js"></script></body></html>'});
    if (url.pathname === '/settings.js' && request.method() === 'GET') return route.fulfill({contentType: 'text/javascript', body: bundle});
    const body = request.postDataJSON?.() ?? null;
    state.requests.push({path: url.pathname, method: request.method(), body});
    if (url.pathname === '/api/v1/comfyui/status' && request.method() === 'GET') return jsonResponse(route, config);
    if (url.pathname === '/api/v1/comfyui/settings' && request.method() === 'PUT') {
      state.settings.push(body);
      config = {...config, ...body};
      return jsonResponse(route, config);
    }
    if (url.pathname === '/api/v1/comfyui/connection/discover' && request.method() === 'POST') {
      return jsonResponse(route, {candidates: [
        {endpoint: 'http://localhost:8188', connected: true, node_count: 12},
        {endpoint: 'http://localhost:8189', connected: false, error: 'Not reachable'},
      ]});
    }
    if (url.pathname === '/api/v1/comfyui/connection/test' && request.method() === 'POST') {
      state.tests.push(body);
      return jsonResponse(route, {connected: true, connection_error: null, node_count: 12});
    }
    const resumePrefix = '/api/v1/comfyui/jobs/by-submission/';
    const resumeId = request.method() === 'POST' && url.pathname.startsWith(resumePrefix) && url.pathname.endsWith('/resume')
      ? decodeURIComponent(url.pathname.slice(resumePrefix.length, -'/resume'.length))
      : '';
    if (resumeId) {
      state.resumes.push({id: resumeId, body});
      config = {...config, pending_submissions: config.pending_submissions.filter(item => item.submission_id !== resumeId)};
      return jsonResponse(route, {client_submission_id: resumeId, job_id: 'uncertain-maestro-job', status: 'queued'});
    }
    const abandonId = request.method() === 'POST' && url.pathname.startsWith(resumePrefix) && url.pathname.endsWith('/abandon')
      ? decodeURIComponent(url.pathname.slice(resumePrefix.length, -'/abandon'.length))
      : '';
    if (abandonId) {
      state.abandons.push({id: abandonId, body});
      config = {...config, pending_submissions: config.pending_submissions.filter(item => item.submission_id !== abandonId)};
      return jsonResponse(route, {abandoned: true});
    }
    const cancelPrefix = '/api/v1/cancel/';
    const cancelId = request.method() === 'POST' && url.pathname.startsWith(cancelPrefix)
      ? decodeURIComponent(url.pathname.slice(cancelPrefix.length))
      : '';
    if (cancelId) {
      state.cancellations.push({job: cancelId});
      config = {...config, pending_submissions: config.pending_submissions.filter(item => item.job_id !== cancelId)};
      return jsonResponse(route, {});
    }
    if (url.pathname === '/api/v1/comfyui/models/discover' && request.method() === 'POST') return jsonResponse(route, {installations: []});
    return jsonResponse(route, {detail: 'Unexpected request ' + request.method() + ' ' + url.pathname}, 404);
  });
  await page.goto('http://maestro.test/');
  await page.getByRole('button', {name: /ComfyUI integration/}).click();
  const integration = page.getByRole('checkbox', {name: /Enable ComfyUI integration/});
  const dedicated = page.getByRole('checkbox', {name: /Dedicated ComfyUI server/});
  await integration.waitFor({state: 'visible'});
  assert.equal(await integration.isChecked(), false, 'The integration remains off by default');
  assert.equal(await dedicated.isChecked(), false, 'The dedicated server option remains off by default');
  await page.getByText('Next action: discover a server or enter its address, then save and test. Testing never enables integration.').waitFor({state: 'visible'});
  await page.getByRole('button', {name: 'Discover endpoints'}).click();
  await page.getByText('http://localhost:8188').waitFor({state: 'visible'});
  await page.getByRole('button', {name: 'Use endpoint'}).first().click();
  assert.equal(await page.getByLabel('ComfyUI endpoint').inputValue(), 'http://localhost:8188',
    'Selecting a discovered endpoint only fills the saved endpoint field');
  assert.equal(await integration.isChecked(), false, 'Endpoint discovery never enables integration');
  await dedicated.check();
  await page.getByRole('button', {name: 'Save settings'}).click();
  await page.waitForFunction(() => Array.from(document.querySelectorAll('button')).some(button =>
    button.textContent?.includes('Test connection') && !button.disabled));
  assert.deepEqual(state.settings, [{enabled: false, endpoint: 'http://localhost:8188', dedicated: true}],
    'The dedicated option persists independently while integration stays disabled');
  const testButton = page.getByRole('button', {name: 'Test connection'});
  assert.equal(await testButton.isEnabled(), true, 'An explicitly saved endpoint can be tested while integration is disabled');
  await testButton.click();
  await page.getByText('ComfyUI is reachable').waitFor({state: 'visible'});
  assert.deepEqual(state.tests, [{}], 'Connection testing is explicit and does not enable integration');
  assert.equal(await integration.isChecked(), false);
  await page.getByText('Pending ComfyUI submissions').waitFor({state: 'visible'});
  await page.getByRole('button', {name: 'Resume monitoring'}).click();
  await page.getByText('b90b76aa-9267-4cc2-961f-c97f2ce90469').waitFor({state: 'detached'});
  assert.deepEqual(state.resumes, [{id: 'b90b76aa-9267-4cc2-961f-c97f2ce90469', body: {}}],
    'Pending uncertain receipts resume through the safe monitor endpoint');
  await page.getByRole('button', {name: 'Dismiss unsubmitted intent'}).first().click();
  assert.deepEqual(state.abandons, [{id: 'd90b76aa-9267-4cc2-961f-c97f2ce90467', body: {}}],
    'Prepared receipts without a Maestro job can be dismissed through the abandon endpoint');
  await page.getByRole('button', {name: 'Dismiss unsubmitted intent'}).click();
  assert.deepEqual(state.cancellations, [{job: 'prepared-maestro-job'}],
    'Prepared receipts with an existing Maestro job use its normal cancel endpoint');
  await page.close();
}

async function main() {
  await checkApiPayloads();
  if (chromium) {
    const browser = await chromium.launch({headless: true, ...(process.platform === 'win32' ? {
      executablePath: process.env.MAESTRO_CHROME || 'C:/Program Files/Google/Chrome/Application/chrome.exe',
    } : {})});
    try {
      await checkDialogWorkflow(browser);
      await checkWorkflowHost(browser);
      await checkSettingsAndDedicatedToggle(browser);
    } finally { await browser.close(); }
    console.log('ComfyUI API payloads, import/preflight, explicit generation, receipt recovery, gallery settings restore, dedicated settings, and mobile dialog checks passed.');
  } else {
    console.log(`ComfyUI API payloads passed. Browser interaction checks skipped: ${playwrightUnavailable?.code || 'Playwright unavailable'}.`);
  }
}

main().catch(error => {
  console.error(error);
  process.exitCode = 1;
});
