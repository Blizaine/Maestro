// Run after the UI dependencies are installed: node tests/ui/editor_project_tools.cjs
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const esbuild = require(path.join(__dirname, '../../ui/node_modules/esbuild'));
const playwright = require(process.env.MAESTRO_PLAYWRIGHT || 'playwright');
const uiRoot = path.resolve(__dirname, '../../ui');
const uiNodeModules = fs.realpathSync(path.join(uiRoot, 'node_modules'));

function makeProject(id = 'project-source', revision = 1) {
  return {
    schema_version: 5,
    id,
    name: id === 'project-source' ? 'Source edit' : 'Imported edit',
    workspace: 'project-tools-workspace',
    revision,
    created_at: 1,
    updated_at: 2,
    canvas: {width: 1920, height: 1080, fps: 24, background: '#000000'},
    assets: {},
    tracks: [],
    markers: [],
    export: {quality: 'high', codec: 'h264', encoder: 'auto', include_audio: true, resolution: 'canvas', frame_rate: 'project', filename: ''},
    exports: [],
  };
}

(async () => {
  const bundle = await esbuild.build({
    absWorkingDir: path.dirname(uiNodeModules),
    stdin: {
      contents: `
        import React, {useState} from 'react';
        import {createRoot} from 'react-dom/client';
        import {EditorProjectTools} from './src/editor/EditorProjectTools';
        import {useEditorStore} from './src/editor/useEditorStore';
        window.editorStore = useEditorStore;
        function App() {
          const [open, setOpen] = useState(false);
          return <><button type="button" onClick={() => setOpen(true)}>Open Project tools</button>
            <EditorProjectTools open={open} onClose={() => setOpen(false)} /></>;
        }
        const root = createRoot(document.getElementById('root'));
        window.mount = () => root.render(<App />);
      `,
      resolveDir: uiRoot,
      loader: 'tsx',
    },
    bundle: true,
    write: false,
    nodePaths: [uiNodeModules],
    jsx: 'automatic',
    define: {'process.env.NODE_ENV': '"development"'},
    logLevel: 'silent',
  });

  const launchOptions = {headless: true};
  if (process.env.MAESTRO_CHROME) launchOptions.executablePath = process.env.MAESTRO_CHROME;
  else if (process.platform === 'win32') launchOptions.channel = 'chrome';
  const browser = await playwright.chromium.launch(launchOptions);
  const page = await browser.newPage({viewport: {width: 1100, height: 900}});
  const requests = [];
  const serverProjects = {'project-source': makeProject('project-source', 3)};
  let failSave = false;
  let package1Polls = 0;
  let package2Cancelled = false;
  let restoreBody = null;
  let packageBody = null;
  let importBody = '';

  try {
    await page.route('**/*', async route => {
      const request = route.request();
      const url = new URL(request.url());
      const method = request.method();
      const json = (body, status = 200) => route.fulfill({status, contentType: 'application/json', body: JSON.stringify(body)});

      if (url.pathname === '/' || url.pathname === '/index.html') {
        return route.fulfill({contentType: 'text/html', body: '<div id="root"></div>'});
      }
      if (url.pathname.startsWith('/api/v1/')) {
        requests.push({method, path: url.pathname, search: url.search});
        const base = '/api/v1/editor/projects/';
        if (url.pathname === `${base}project-source/revisions` || url.pathname === `${base}project-imported/revisions`) {
          return json({revisions: [
            {revision: 2, updated_at: 1730000000, name: 'Saved edit'},
            {revision: 1, updated_at: 1720000000},
          ]});
        }
        if (url.pathname === `${base}project-source` && method === 'PUT') {
          if (failSave) return json({detail: 'Mock save rejected'}, 500);
          serverProjects['project-source'] = {...serverProjects['project-source'], revision: serverProjects['project-source'].revision + 1};
          return json(serverProjects['project-source']);
        }
        if (url.pathname === `${base}project-source/restore` && method === 'POST') {
          restoreBody = request.postDataJSON();
          serverProjects['project-source'] = {...makeProject('project-source', serverProjects['project-source'].revision + 1), name: 'Restored snapshot'};
          return json(serverProjects['project-source']);
        }
        if (url.pathname === `${base}project-source/package` && method === 'POST') {
          packageBody = request.postDataJSON();
          const number = requests.filter(entry => entry.path === `${base}project-source/package`).length;
          return json({job_id: number === 1 ? 'package-job-1' : 'package-job-2', status: 'queued'});
        }
        if (url.pathname === `${base}project-imported` && method === 'GET') return json(serverProjects['project-imported']);
        if (url.pathname === `${base}project-source` && method === 'GET') return json(serverProjects['project-source']);
        if (url.pathname === '/api/v1/status/package-job-1') {
          package1Polls += 1;
          return json({
            job_id: 'package-job-1', status: package1Polls === 1 ? 'running' : 'completed',
            progress: package1Polls === 1 ? 0.5 : 1, phase: 'Packaging', message: 'Writing project ZIP',
            output_files: package1Polls === 1 ? [] : ['portable/source-project.zip'], error: null,
          });
        }
        if (url.pathname === '/api/v1/status/package-job-2') {
          return json({job_id: 'package-job-2', status: package2Cancelled ? 'cancelled' : 'queued', progress: 0, phase: 'Queued', message: 'Waiting', output_files: [], error: null});
        }
        if (url.pathname === '/api/v1/cancel/package-job-2' && method === 'POST') {
          package2Cancelled = true;
          return route.fulfill({status: 200, body: ''});
        }
        if (url.pathname === '/api/v1/editor/projects/import' && method === 'POST') {
          importBody = request.postDataBuffer()?.toString('utf8') || '';
          serverProjects['project-imported'] = makeProject('project-imported', 1);
          return json(serverProjects['project-imported']);
        }
        if (url.pathname === '/api/v1/editor/media/status') return json({assets: []});
        return json({});
      }
      if (url.pathname.endsWith('.zip')) return route.fulfill({status: 200, contentType: 'application/zip', body: ''});
      return route.fulfill({status: 204, body: ''});
    });

    await page.goto('http://editor-project-tools.test');
    await page.addScriptTag({content: bundle.outputFiles[0].text});
    await page.evaluate(project => {
      window.editorStore.setState({workspace: project.workspace, project, dirty: false, saving: false, error: null});
      window.mount();
    }, makeProject('project-source', 3));

    assert.deepEqual(requests, [], 'Mounting a closed dialog does not make project-tool requests');
    await page.getByRole('button', {name: 'Open Project tools'}).click();
    const dialog = page.getByRole('dialog', {name: 'Project tools'});
    await dialog.waitFor();
    await dialog.getByRole('button', {name: 'Restore revision 1'}).waitFor();
    assert.equal(requests.filter(entry => entry.path.endsWith('/revisions')).length, 1, 'History is fetched only when opened');
    assert.equal(new URLSearchParams(requests.find(entry => entry.path.endsWith('/revisions')).search).get('workspace'), 'project-tools-workspace');

    await page.evaluate(() => window.editorStore.setState({dirty: true}));
    await dialog.getByRole('button', {name: 'Restore revision 1'}).click();
    await page.waitForFunction(() => window.editorStore.getState().project?.revision === 5);
    assert.deepEqual(restoreBody, {workspace: 'project-tools-workspace', revision: 1, expected_revision: 4},
      'Restore saves dirty state first and uses the saved revision for optimistic concurrency');
    const saveIndex = requests.findIndex(entry => entry.method === 'PUT' && entry.path.endsWith('/project-source'));
    const restoreIndex = requests.findIndex(entry => entry.method === 'POST' && entry.path.endsWith('/project-source/restore'));
    const loadIndex = requests.findIndex(entry => entry.method === 'GET' && entry.path.endsWith('/project-source'));
    assert.ok(saveIndex >= 0 && saveIndex < restoreIndex && restoreIndex < loadIndex,
      'The dirty project is saved before restore, then the restored project is reloaded');

    await dialog.getByRole('tab', {name: 'Portable project'}).click();
    await dialog.getByRole('button', {name: 'Create package'}).click();
    await dialog.getByRole('link', {name: 'Download source-project.zip'}).waitFor({timeout: 7000});
    assert.deepEqual(packageBody, {workspace: 'project-tools-workspace', revision: 5}, 'Packaging uses the saved revision');
    const packageLink = dialog.getByRole('link', {name: 'Download source-project.zip'});
    assert.match(await packageLink.getAttribute('href'), /\/api\/v1\/file\/portable\/source-project\.zip\?workspace=project-tools-workspace/,
      'Completed packages use the native file URL with the owning workspace');
    assert.equal(await packageLink.getAttribute('download'), 'source-project.zip', 'The package is downloaded through a native anchor');

    await dialog.getByRole('button', {name: 'Create package'}).click();
    await dialog.getByRole('button', {name: 'Cancel package'}).waitFor();
    await dialog.getByRole('button', {name: 'Cancel package'}).click();
    await dialog.getByText('Package cancelled.').waitFor({timeout: 5000});
    assert.ok(requests.some(entry => entry.method === 'POST' && entry.path === '/api/v1/cancel/package-job-2'),
      'Only the owned package job is cancelled through the existing job endpoint');

    const archive = {name: 'portable-edit.zip', mimeType: 'application/zip', buffer: Buffer.from('mock zip')};
    await dialog.getByLabel('Choose project ZIP').setInputFiles(archive);
    await dialog.getByRole('button', {name: 'Import ZIP'}).click();
    await page.waitForFunction(() => window.editorStore.getState().project?.id === 'project-imported', null, {timeout: 7000});
    assert.match(importBody, /portable-edit\.zip/, 'The selected archive is submitted as multipart form data');
    assert.match(importBody, /project-tools-workspace/, 'The target workspace travels with the archive upload');
    await dialog.getByText(/Opened imported project/).waitFor();
    assert.equal(await dialog.getByText(/Opened imported project/).count(), 1, 'A successful import opens the newly returned project ID');
    assert.equal(requests.some(entry => entry.method === 'DELETE'), false, 'Import never deletes the source project or assets');

    // A failed save must stop restore before the server-side restore endpoint is called.
    failSave = true;
    await page.evaluate(project => window.editorStore.setState({project, dirty: true, saving: false, error: null}), makeProject('project-source', 2));
    await dialog.getByRole('tab', {name: 'History'}).click();
    await dialog.getByRole('button', {name: 'Restore revision 1'}).waitFor();
    const restoresBeforeFailedSave = requests.filter(entry => entry.path.endsWith('/restore')).length;
    await dialog.getByRole('button', {name: 'Restore revision 1'}).click();
    await dialog.getByRole('alert').filter({hasText: 'Mock save rejected'}).waitFor();
    assert.equal(requests.filter(entry => entry.path.endsWith('/restore')).length, restoresBeforeFailedSave,
      'A failed dirty-project save prevents the restore mutation');

    console.log('Project history restore, dirty-save protection, portable ZIP progress/download/cancellation and ZIP import passed');
    console.log('All editor project API activity was routed through mocked HTTP endpoints; no archive was downloaded');
  } finally {
    await browser.close();
  }
})().catch(error => {
  console.error(error);
  process.exitCode = 1;
});
