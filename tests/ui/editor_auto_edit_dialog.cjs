// Isolated Auto-edit proposal UI regression harness. All HTTP is mocked.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const esbuild = require(path.join(__dirname, '../../ui/node_modules/esbuild'));
const playwright = require(process.env.MAESTRO_PLAYWRIGHT || 'playwright');
const uiRoot = path.resolve(__dirname, '../../ui');
const uiNodeModules = fs.realpathSync(path.join(uiRoot, 'node_modules'));

function makeProject(id = 'auto-edit-project', revision = 4) {
  return {
    schema_version: 5,
    id,
    name: id === 'auto-edit-project' ? 'Auto-edit fixture' : 'Other project',
    workspace: 'auto-edit-workspace',
    revision,
    created_at: 1,
    updated_at: 2,
    canvas: {width: 1920, height: 1080, fps: 24, background: '#000000'},
    assets: {
      shotA: {id: 'shotA', name: 'Opening shot.mp4', type: 'video', origin: 'output', workspace: 'Media', path: '/outputs/opening.mp4', url: '/media/opening.mp4', duration: 10, width: 1920, height: 1080, fps: 24, has_audio: true, size: 100},
      shotB: {id: 'shotB', name: 'Detail shot.mp4', type: 'video', origin: 'output', workspace: 'Media', path: '/outputs/detail.mp4', url: '/media/detail.mp4', duration: 10, width: 1920, height: 1080, fps: 24, has_audio: true, size: 120},
    },
    tracks: [{
      id: 'video-track', name: 'Video', type: 'video', z_index: 0, muted: false, locked: false, volume: 1,
      items: [
        {id: 'clip-a', asset_id: 'shotA', name: 'Opening shot', start: 0, duration: 1.5, source_in: 0, speed: 1, volume: 1, opacity: 1, fit: 'contain', transform: {x: 0, y: 0, scale: 1, rotation: 0}},
        {id: 'clip-b', asset_id: 'shotB', name: 'Detail shot', start: 4, duration: 1.5, source_in: 0, speed: 1, volume: 1, opacity: 1, fit: 'contain', transform: {x: 0, y: 0, scale: 1, rotation: 0}},
      ],
    }],
    markers: [],
    export: {quality: 'high', codec: 'h264', encoder: 'auto', include_audio: true, resolution: 'canvas', frame_rate: 'project', filename: ''},
    exports: [],
  };
}

function makeProposal(project, id, request) {
  return {
    schema_version: 1,
    proposal_id: id,
    project_id: project.id,
    workspace: project.workspace,
    base_revision: request.revision,
    client_signature: request.client_signature,
    mode: request.mode,
    status: 'ready',
    summary: 'Move the opening into the first beat and tighten the detail shot.',
    actions: [
      {id: `${id}-act-001`, type: 'move', item_id: 'clip-a', start: 0.5, reason: 'Create a brief lead-in.'},
      {id: `${id}-act-002`, type: 'trim', item_id: 'clip-b', duration: 1.25, source_in: 0.25, reason: 'Keep the strongest moment.'},
    ].map(action => ({item_id:null,asset_id:null,track_id:null,start:null,duration:null,source_in:null,volume:null,seconds:null,reason:null,...action})),
    warnings: ['Review the transition into the second shot.'],
    missing_media: ['Optional aerial establishing shot', {type: 'audio', description: 'Soft room tone'}],
    drafts: [{phase: 'initial', raw: '{"summary":"A short, measured opening"}'}],
    errors: [],
    created_at: 1800000000,
    updated_at: 1800000001,
  };
}

(async () => {
  const bundle = await esbuild.build({
    absWorkingDir: path.dirname(uiNodeModules),
    stdin: {
      contents: `
        import React, {useState} from 'react';
        import {createRoot} from 'react-dom/client';
        import {EditorAutoEditDialog} from './src/editor/EditorAutoEditDialog';
        import {editorAutoEditSignature} from './src/editor/editorAutoEdit';
        import {useEditorStore} from './src/editor/useEditorStore';
        window.editorStore = useEditorStore;
        window.editorSignature = editorAutoEditSignature;
        function App() {
          const [open, setOpen] = useState(false);
          return <><button type="button" onClick={() => setOpen(true)}>Open Auto-edit</button>
            <EditorAutoEditDialog open={open} onClose={() => setOpen(false)} /></>;
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
  const plans = [];
  const fetchedProposalIds = [];
  const savedProjects = {'auto-edit-project': makeProject()};
  const proposalRecords = new Map();
  let writerAvailable = false;
  let failSave = false;
  let loseNextPlanReceipt = true;

  try {
    await page.route('**/*', async route => {
      const request = route.request();
      const url = new URL(request.url());
      const method = request.method();
      const json = (body, status = 200) => route.fulfill({status, contentType: 'application/json', body: JSON.stringify(body)});

      if (url.pathname === '/' || url.pathname === '/index.html') {
        return route.fulfill({contentType: 'text/html', body: '<meta name="viewport" content="width=device-width,initial-scale=1"><div id="root"></div>'});
      }
      if (url.pathname.startsWith('/api/v1/')) {
        requests.push({method, path: url.pathname, search: url.search});
        if (url.pathname === '/api/v1/editor/auto-edit/capabilities' && method === 'GET') {
          return json({
            director_available: true,
            writer_available: writerAvailable,
            writer_provider: writerAvailable ? 'Local writer' : null,
            writer_model: writerAvailable ? 'installed-model' : null,
            reason: writerAvailable ? '' : 'No local proposal writer is installed.',
            content_basis: 'project asset and timeline metadata only',
          });
        }
        if (url.pathname === '/api/v1/editor/auto-edit/proposals' && method === 'GET') {
          const projectId = url.searchParams.get('project_id');
          const workspace = url.searchParams.get('workspace');
          assert.ok(projectId);
          assert.equal(workspace, 'auto-edit-workspace');
          const proposals = [...proposalRecords.values()].filter(entry => entry.project_id === projectId && entry.workspace === workspace).map(entry => ({
            proposal_id: entry.proposal_id, project_id: entry.project_id, workspace: entry.workspace,
            base_revision: entry.base_revision, mode: entry.mode, status: entry.status, summary: entry.summary,
            created_at: entry.created_at, updated_at: entry.updated_at,
          }));
          proposals.push({proposal_id: 'foreign-record', project_id: 'another-project', workspace: 'foreign', base_revision: 1, mode: 'director', status: 'ready', summary: 'Foreign project proposal'});
          return json({proposals});
        }
        if (url.pathname === '/api/v1/editor/projects/auto-edit-project' && method === 'PUT') {
          if (failSave) return json({detail: 'Mock save rejected'}, 500);
          const body = request.postDataJSON();
          const current = savedProjects['auto-edit-project'];
          savedProjects['auto-edit-project'] = {...body.project, revision: current.revision + 1};
          return json(savedProjects['auto-edit-project']);
        }
        if (url.pathname === '/api/v1/editor/projects' && method === 'GET') return json({projects: []});
        if (url.pathname === '/api/v1/editor/auto-edit/plan' && method === 'POST') {
          const body = request.postDataJSON();
          plans.push(body);
          const saved = savedProjects[body.project_id];
          assert.ok(saved, 'The proposal request uses a saved project');
          const record = makeProposal(saved, body.proposal_id, body);
          proposalRecords.set(body.proposal_id, record);
          if (loseNextPlanReceipt) {
            loseNextPlanReceipt = false;
            return json({detail: 'Mock receipt connection dropped after persistence'}, 502);
          }
          return json(record);
        }
        if (url.pathname.startsWith('/api/v1/editor/auto-edit/proposals/')) {
          const proposalId = decodeURIComponent(url.pathname.split('/').at(-1));
          fetchedProposalIds.push(proposalId);
          const record = proposalRecords.get(proposalId);
          return record ? json(record) : json({detail: 'Proposal not found'}, 404);
        }
        if (url.pathname === '/api/v1/media/status' || url.pathname === '/api/v1/editor/media/status') return json({assets: []});
        if (/generation|director|export|download/i.test(url.pathname)) {
          throw new Error(`Forbidden non-planning request: ${method} ${url.pathname}`);
        }
        return json({});
      }
      return route.fulfill({status: 204, body: ''});
    });

    await page.goto('http://editor-auto-edit.test');
    await page.addScriptTag({content: bundle.outputFiles[0].text});
    await page.evaluate(project => {
      window.editorStore.setState({workspace: project.workspace, project, dirty: false, saving: false, history: [], future: [], error: null});
      window.mount();
    }, makeProject());

    assert.deepEqual(requests, [], 'Mounting the closed dialog performs no API calls');
    await page.getByRole('button', {name: 'Open Auto-edit'}).click();
    const dialog = page.getByRole('dialog', {name: 'Auto-edit proposal'});
    await dialog.waitFor();
    await dialog.getByText('No local proposal writer is installed.').waitFor();
    assert.equal(requests.filter(entry => entry.path.endsWith('/capabilities')).length, 1, 'Opening checks writer capability');
    assert.equal(requests.filter(entry => entry.path === '/api/v1/editor/auto-edit/proposals').length, 1, 'Opening loads compact proposal summaries');
    assert.equal(plans.length, 0, 'Opening never generates a proposal');
    assert.equal(await dialog.getByRole('option', {name: 'Director'}).isDisabled(), false, 'Director remains available without a writer');
    assert.equal(await dialog.getByLabel('Planning mode').locator('option').nth(1).evaluate(option => option.disabled), true, 'Writer modes are disabled when the writer is unavailable');

    await dialog.getByRole('button', {name: 'Close Auto-edit proposal'}).click();
    writerAvailable = true;
    await page.getByRole('button', {name: 'Open Auto-edit'}).click();
    await dialog.getByText('Local writer · installed-model').waitFor();
    await dialog.getByLabel('Planning mode').selectOption('revise');
    await dialog.getByLabel('Auto-edit brief').fill('Give the opening a little more room, then tighten the detail shot.');
    await dialog.getByLabel('Target duration seconds').fill('4.25');

    await page.evaluate(() => window.editorStore.setState({dirty: true, error: null}));
    failSave = true;
    await dialog.getByRole('button', {name: 'Generate proposal'}).click();
    await dialog.getByRole('alert').filter({hasText: 'Mock save rejected'}).waitFor();
    assert.equal(plans.length, 0, 'A failed project save blocks proposal planning');
    failSave = false;

    await dialog.getByRole('button', {name: 'Generate proposal'}).click();
    await dialog.getByRole('heading', {name: 'Proposal review'}).waitFor({timeout: 7000});
    await dialog.getByText(/Recovered the existing proposal by request ID/).waitFor();
    assert.equal(plans.length, 1, 'A lost receipt is never recovered by resubmitting the plan');
    assert.equal(fetchedProposalIds.length, 1, 'A lost receipt is recovered with GET for the same proposal ID');
    assert.equal(fetchedProposalIds[0], plans[0].proposal_id);
    assert.match(plans[0].proposal_id, /^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i, 'Each user request has a UUID idempotency key');
    assert.equal(plans[0].project_id, 'auto-edit-project');
    assert.equal(plans[0].workspace, 'auto-edit-workspace');
    assert.equal(plans[0].revision, 5, 'Planning uses the revision returned by save');
    assert.equal(plans[0].mode, 'revise');
    assert.equal(plans[0].brief, 'Give the opening a little more room, then tighten the detail shot.');
    assert.equal(plans[0].target_seconds, 4.25);
    assert.equal(plans[0].client_signature, await page.evaluate(() => window.editorSignature(window.editorStore.getState().project)),
      'The server echoes the signature of the saved project content');
    assert.equal(await dialog.getByRole('checkbox', {name: 'Include action 1'}).isChecked(), true);
    assert.equal(await dialog.getByRole('checkbox', {name: 'Include action 2'}).isChecked(), true);
    await dialog.getByText('Review the transition into the second shot.').waitFor();
    await dialog.getByText('Optional aerial establishing shot').waitFor();
    await dialog.getByText(/Show writer draft and provenance/).click();
    await dialog.getByText(/A short, measured opening/).waitFor();
    assert.equal(await dialog.getByText('Foreign project proposal').count(), 0, 'History identity checks exclude other projects');

    await dialog.getByRole('checkbox', {name: 'Include action 2'}).uncheck();
    await dialog.getByText('Validated preview · 1 change').waitFor();
    const beforeApply = await page.evaluate(() => ({
      history: window.editorStore.getState().history.length,
      firstStart: window.editorStore.getState().project.tracks[0].items[0].start,
      secondDuration: window.editorStore.getState().project.tracks[0].items[1].duration,
    }));
    await dialog.getByRole('button', {name: 'Apply selected actions'}).click();
    const afterApply = await page.evaluate(() => {
      const state = window.editorStore.getState();
      return {
        history: state.history.length,
        firstStart: state.project.tracks[0].items.find(item => item.id === 'clip-a').start,
        secondDuration: state.project.tracks[0].items.find(item => item.id === 'clip-b').duration,
        actionIds: state.project.auto_edit_history.at(-1).action_ids,
      };
    });
    assert.equal(afterApply.firstStart, 0.5, 'The selected move action is applied');
    assert.equal(afterApply.secondDuration, beforeApply.secondDuration, 'The unchecked trim action is left unchanged');
    assert.deepEqual(afterApply.actionIds, [`${plans[0].proposal_id}-act-001`], 'The one-step project history records only selected action IDs');
    assert.equal(afterApply.history, beforeApply.history + 1, 'Applying selected actions is one undo step');
    await dialog.getByText(/Applied 1 selected action in one undo step/).waitFor();
    assert.equal(await dialog.getByRole('button', {name: 'Apply selected actions'}).isDisabled(), true, 'An applied proposal cannot be applied twice');
    await page.evaluate(() => window.editorStore.getState().undo());
    assert.equal(await page.evaluate(() => window.editorStore.getState().project.tracks[0].items[0].start), 0,
      'One undo restores the original timeline');

    await dialog.getByRole('tab', {name: 'History'}).click();
    await dialog.getByRole('button', {name: 'Refresh'}).click();
    await dialog.getByRole('button', {name: `Resume proposal ${plans[0].proposal_id}`}).waitFor();
    const getCountBeforeResume = fetchedProposalIds.length;
    await dialog.getByRole('button', {name: `Resume proposal ${plans[0].proposal_id}`}).click();
    await dialog.getByText('Proposal review').waitFor();
    assert.equal(fetchedProposalIds.length, getCountBeforeResume + 1, 'Resuming history retrieves one full proposal by ID');

    await page.evaluate(() => window.editorStore.getState().updateItem('clip-a', {start: 0.25}));
    await dialog.getByText(/Preview unavailable: This auto-edit proposal is stale/).waitFor();
    assert.equal(await dialog.getByRole('button', {name: 'Apply selected actions'}).isDisabled(), true,
      'Changing project content after proposal creation prevents stale apply');

    await dialog.getByRole('button', {name: 'Close Auto-edit proposal'}).click();
    await page.evaluate(project => window.editorStore.setState({project, dirty: false}), makeProject('other-project', 1));
    await page.getByRole('button', {name: 'Open Auto-edit'}).click();
    await dialog.waitFor();
    assert.equal(await dialog.getByRole('region', {name: 'Proposal review'}).count(), 0,
      'Switching projects clears the active proposal from the prior project');
    assert.equal(plans.length, 1, 'No new plan is generated during project changes or dialog reopen');
    const forbidden = requests.filter(entry => /generation|director|export|download/i.test(entry.path));
    assert.deepEqual(forbidden, [], `The dialog makes no generation/export/download requests: ${forbidden.map(entry => entry.path).join(', ')}`);
    console.log('Auto-edit writer gating, save-before-plan, receipt recovery, compact history, partial preview/apply/undo, stale rejection, and project isolation passed');
    console.log('Every API request was mocked; no generation, export, or download request was sent');
  } finally {
    await browser.close();
  }
})().catch(error => {
  console.error(error);
  process.exitCode = 1;
});
