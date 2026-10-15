// Mount the real virtualized MainContent and real MediaFeedItem cards while
// intercepting every network request. This never contacts Maestro or starts a
// render. Run after `npm run build` with the bundled Node/Playwright runtime.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const root = path.resolve(__dirname, '../..');
const esbuild = require(path.join(root, 'ui/node_modules/esbuild'));
const {chromium} = require(process.env.MAESTRO_PLAYWRIGHT || 'playwright');
const artifacts = path.join(root, '.codex-tmp/director-gallery-groups-browser');
fs.mkdirSync(artifacts, {recursive: true});

const bundleSource = [
  "import React from 'react'; import {createRoot} from 'react-dom/client';",
  "import {useStore} from './src/stores/useStore'; import {MainContent} from './src/components/MainContent/MainContent';",
  'window.store = useStore;',
  'window.mountGallery = () => { window.reactRoot = createRoot(document.getElementById("root")); window.reactRoot.render(<MainContent/>); };',
].join('\n');

const parentOutput = (name = 'joined-final.mp4', clipCount = 4) => ({
  id: `Saved/${name}`, name, workspace: 'Saved', path: `Saved/${name}`,
  url: `/api/v1/file/${name}?workspace=Saved`, type: 'video', mode: 'video',
  size: 2048, created_at: 300, favorite: false, metadata_ready: true,
  director_group: {pipeline_id: 'director-pipeline-1', clip_count: clipCount},
});
const ordinaryOutput = {
  id: 'Saved/ordinary.png', name: 'ordinary.png', workspace: 'Saved', path: 'Saved/ordinary.png',
  url: '/api/v1/file/ordinary.png?workspace=Saved', type: 'image', mode: 'image',
  size: 512, created_at: 200, favorite: false, metadata_ready: true,
};
const shot = (name, index, createdAt, overrides = {}) => ({
  id: `Saved/${name}`, name, workspace: 'Saved', path: `Saved/${name}`,
  url: `/api/v1/file/${name}?workspace=Saved`, type: 'video', mode: 'video',
  size: 1024, created_at: createdAt, favorite: false, metadata_ready: true,
  director_pipeline_id: 'director-pipeline-1', director_clip_index: index,
  ...overrides,
});
const baseShots = [
  shot('shot-1.mp4', 0, 100),
  shot('shot-2-new.mp4', 1, 190),
  shot('shot-2-old.mp4', 1, 150),
  shot('shot-3.mp4', 2, 110),
];

async function runViewport(browser, viewport, label, bundleText, css) {
  const page = await browser.newPage({viewport, hasTouch: viewport.width < 900, isMobile: viewport.width < 900});
  page.setDefaultTimeout(7000);
  const errors = [];
  const requests = [];
  let shots = [...baseShots];
  let joinedCount = 0;
  const deletedNames = new Set();
  let holdMemberFetch = false;
  let releaseMemberFetch = null;
  page.on('pageerror', error => errors.push(error.message));

  await page.route('**/*', async route => {
    const url = new URL(route.request().url());
    const method = route.request().method();
    requests.push({path: url.pathname, search: url.search, method});
    const json = body => route.fulfill({contentType: 'application/json', body: JSON.stringify(body)});
    if (url.pathname === '/') {
      return route.fulfill({contentType: 'text/html', body:
        '<meta name="viewport" content="width=device-width,initial-scale=1"><div id="root" style="height:100dvh;width:100vw;display:flex"></div>'});
    }
    if (url.pathname === '/api/v1/outputs') {
      const workspace = url.searchParams.get('workspace') || 'Saved';
      const pipelineId = url.searchParams.get('director_pipeline_id');
      if (pipelineId) {
        if (holdMemberFetch) await new Promise(resolve => {releaseMemberFetch = resolve;});
        const offset = Number(url.searchParams.get('offset') || 0);
        const page = shots.slice(offset, offset + 100);
        return json({outputs: page, total: shots.length, next_cursor: null});
      }
      if (workspace === 'Other') return json({outputs: [], total: 0, next_cursor: null});
      const currentName = joinedCount ? `joined-final-${joinedCount}.mp4` : 'joined-final.mp4';
      const currentParent = deletedNames.has(currentName) ? null : parentOutput(currentName, shots.length);
      const previousName = joinedCount > 1 ? `joined-final-${joinedCount - 1}.mp4` : 'joined-final.mp4';
      const currentWasDeleted = deletedNames.has(currentName);
      const previousFinal = joinedCount && !deletedNames.has(previousName)
        ? {...parentOutput(previousName, currentWasDeleted ? shots.length : 0)} : null;
      if (previousFinal && currentWasDeleted) {
        previousFinal.director_group = {pipeline_id: 'director-pipeline-1', clip_count: shots.length};
      } else if (previousFinal) {
        delete previousFinal.director_group;
      }
      const outputs = [currentParent, previousFinal, !deletedNames.has(ordinaryOutput.name) ? ordinaryOutput : null].filter(Boolean);
      return json({outputs, total: outputs.length, next_cursor: null});
    }
    if (url.pathname.endsWith('/metadata')) {
      const filename = decodeURIComponent(url.pathname.split('/').at(-2) || 'media');
      const clip = shots.find(item => item.name === filename);
      return json({source: 'sidecar', timestamp: 100, params: {
        prompt: `Prompt for ${filename}`, model_type: 'minimax_h3_fused_turbo',
        ...(clip ? {multi_clip_info: {group_id: 'legacy-group', index: clip.director_clip_index, total: 3}} : {}),
      }});
    }
    if (url.pathname.startsWith('/api/v1/media-info/')) return json({duration_seconds: 2, width: 640, height: 360});
    if (url.pathname.startsWith('/api/v1/thumbnail/')) {
      return route.fulfill({contentType: 'image/svg+xml', body: '<svg xmlns="http://www.w3.org/2000/svg" width="640" height="360"><rect width="640" height="360" fill="#334155"/></svg>'});
    }
    if (url.pathname.startsWith('/api/v1/file/')) {
      const filename = decodeURIComponent(url.pathname.split('/').at(-1) || '');
      if (/\.(?:png|jpe?g|webp)$/i.test(filename)) {
        return route.fulfill({contentType: 'image/svg+xml', body: '<svg xmlns="http://www.w3.org/2000/svg" width="640" height="360"><rect width="640" height="360" fill="#475569"/></svg>'});
      }
      return route.fulfill({contentType: 'video/mp4', body: ''});
    }
    if (url.pathname.startsWith('/api/v1/favorites/') && method === 'POST') {
      return json({name: decodeURIComponent(url.pathname.split('/').at(-1)), favorite: true});
    }
    if (url.pathname.startsWith('/api/v1/outputs/') && method === 'DELETE') {
      deletedNames.add(decodeURIComponent(url.pathname.split('/').at(-1)));
      return json({ok: true});
    }
    if (url.pathname.startsWith('/api/v1/outputs/') && url.pathname.endsWith('/move') && method === 'POST') {
      const filename = decodeURIComponent(url.pathname.split('/').at(-2));
      shots = shots.filter(file => file.name !== filename);
      return json({ok: true});
    }
    if (url.pathname === '/api/v1/director/pipelines/director-pipeline-1/rejoin' && method === 'POST') {
      joinedCount += 1;
      return json({filename: `joined-final-${joinedCount}.mp4`});
    }
    if (url.pathname === '/api/v1/director/pipelines/director-pipeline-1') {
      return json({pipeline_id: 'director-pipeline-1', status: 'completed', clips: []});
    }
    if (url.pathname === '/api/v1/director/pipelines') return json({pipelines: []});
    if (url.pathname.startsWith('/api/')) return json({});
    return route.fulfill({status: 204, body: ''});
  });

  await page.goto('http://director-gallery.test');
  await page.addStyleTag({content: css});
  await page.addScriptTag({content: bundleText});
  await page.evaluate(({parent, ordinary, initialShots}) => {
    const idle = async () => {};
    window.store.setState({
      loadWorkspaces: idle, loadModels: idle, loadServicesConfig: idle,
      loadLlmStatus: idle, loadPipelineList: idle, reconnectJobs: idle,
      loadSettingsFromOutput: async () => {
        const state = window.store.getState();
        window.loadedSettingsOutput = state.filteredOutputs()[state.selectedOutput]?.name || null;
      },
      activeWorkspace: 'Saved', browsingAllFolders: false, browsingUploads: false,
      mediaFilter: 'all', outputSearchQuery: '', outputs: [parent, ordinary],
      outputsTotal: 2, outputsCursor: null, outputsLoading: false, selectedOutput: 0,
      selectedOutputMeta: null, directorGalleryGroups: {}, jobs: [], pipelineId: null,
      pipelineStatus: null, models: [], servicesConfig: {}, workspaces: [
        {name: 'default', path: 'default'}, {name: 'Saved', path: 'Saved'}, {name: 'Other', path: 'Other'},
      ], isGenerating: false, isEnhancing: false, generationMode: 'video',
    });
    window.fixtureParent = parent;
    window.fixtureOrdinary = ordinary;
    window.fixtureShots = initialShots;
  }, {parent: parentOutput(), ordinary: ordinaryOutput, initialShots: baseShots});
  await page.evaluate(() => window.mountGallery());

  const parentCard = () => page.locator('[data-output-name="joined-final.mp4"]');
  const assertCollapsedGalleryLayout = async phase => {
    await page.waitForFunction(phaseLabel => {
      const cards = [...document.querySelectorAll('[data-output-name]')];
      const parent = cards.find(card => card.dataset.outputName === 'joined-final.mp4');
      const ordinary = cards.find(card => card.dataset.outputName === 'ordinary.png');
      if (!parent || !ordinary || parent.innerText.includes('Loading...') || ordinary.innerText.includes('Loading...')) return false;
      const parentRect = parent.getBoundingClientRect();
      const ordinaryRect = ordinary.getBoundingClientRect();
      const snapshot = [parentRect.top, parentRect.bottom, ordinaryRect.top, ordinaryRect.bottom]
        .map(value => Math.round(value * 10) / 10);
      const previous = window.__directorCollapsedLayout;
      const stableFrames = previous?.label === phaseLabel
        && snapshot.every((value, index) => Math.abs(value - previous.snapshot[index]) < 0.5)
        ? previous.stableFrames + 1 : 0;
      window.__directorCollapsedLayout = {label: phaseLabel, snapshot, stableFrames};
      return stableFrames >= 5;
    }, phase, {polling: 'raf', timeout: 10000});
    const layout = await page.evaluate(() => {
      const parent = document.querySelector('[data-output-name="joined-final.mp4"]');
      const ordinary = document.querySelector('[data-output-name="ordinary.png"]');
      const button = parent?.querySelector('button[aria-expanded="false"]');
      if (!parent || !ordinary || !button) return null;
      const parentRect = parent.getBoundingClientRect();
      const ordinaryRect = ordinary.getBoundingClientRect();
      const buttonRect = button.getBoundingClientRect();
      const hitTarget = document.elementFromPoint(buttonRect.left + buttonRect.width / 2, buttonRect.top + buttonRect.height / 2);
      return {
        parentBottom: parentRect.bottom,
        ordinaryTop: ordinaryRect.top,
        buttonVisible: buttonRect.width > 0 && buttonRect.height > 0
          && buttonRect.top >= parentRect.top && buttonRect.bottom <= parentRect.bottom,
        buttonReceivesPointer: hitTarget === button || button.contains(hitTarget),
      };
    });
    assert.ok(layout, `${label} ${phase}: collapsed cards and expander are mounted`);
    assert.ok(layout.parentBottom <= layout.ordinaryTop + 1,
      `${label} ${phase}: next card must start after the parent bottom (parent=${layout.parentBottom}, next=${layout.ordinaryTop})`);
    assert.ok(layout.buttonVisible, `${label} ${phase}: Show clips control is fully visible inside its parent`);
    assert.ok(layout.buttonReceivesPointer, `${label} ${phase}: Show clips control receives pointer hits without an overlay`);
    console.log(`${label} ${phase} bounds: parent bottom=${layout.parentBottom.toFixed(1)}, next card top=${layout.ordinaryTop.toFixed(1)}, expander pointer target=${layout.buttonReceivesPointer}`);
  };
  await parentCard().waitFor();
  assert.equal(await page.locator('[data-feed-index]').count(), 2, 'the default gallery shows the final and ordinary media only');
  assert.equal(requests.some(request => request.path === '/api/v1/outputs' && request.search.includes('director_pipeline_id=')), false,
    'individual clips are loaded only after expansion');

  await assertCollapsedGalleryLayout('initial mount');
  const expand = await parentCard().getByRole('button', {name: 'Show 4 clips'});
  assert.equal(await expand.getAttribute('aria-expanded'), 'false');
  await expand.click({trial: true});
  await expand.click();
  await page.waitForFunction(() => {
    const state = window.store.getState();
    return state.filteredOutputs().length === 6
      && Object.values(state.directorGalleryGroups).some(group => group.loaded && group.clips.length === 4);
  });
  assert.equal(await parentCard().getByRole('button', {name: 'Hide clips'}).getAttribute('aria-expanded'), 'true');
  const sequence = await page.evaluate(() => window.store.getState().filteredOutputs().map(file => file.name));
  assert.deepEqual(sequence, ['joined-final.mp4', 'shot-1.mp4', 'shot-2-new.mp4', 'shot-2-old.mp4', 'shot-3.mp4', 'ordinary.png'],
    'expanded shot takes appear below the final in clip-index/newest-take order');
  assert.ok(requests.some(request => request.path === '/api/v1/outputs'
    && request.search.includes('director_pipeline_id=director-pipeline-1')
    && request.search.includes('workspace=Saved')
    && request.search.includes('limit=100')),
  'member request is pipeline and workspace qualified');
  const memberCards = page.locator('[data-director-pipeline-id="director-pipeline-1"]');
  assert.equal(await memberCards.count(), 4, 'each take is its own normal gallery card');
  assert.equal(await memberCards.nth(0).getAttribute('data-director-clip-index'), '0');
  assert.equal(await memberCards.nth(2).getAttribute('data-director-clip-index'), '1');
  await memberCards.nth(2).getByText('Clip 2', {exact: true}).waitFor();

  await page.screenshot({path: path.join(artifacts, `${label}-expanded.png`)});
  assert.ok(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth), `${label} gallery stays within viewport width`);

  // Existing favorite and Load Settings controls must act on the selected
  // flattened member identity instead of indexing the raw parent page.
  await memberCards.nth(1).getByRole('button', {name: 'Add to favorites'}).click();
  await page.waitForFunction(() => Object.values(window.store.getState().directorGalleryGroups)
    .some(group => group.clips.some(file => file.name === 'shot-2-new.mp4' && file.favorite)));
  await memberCards.nth(1).getByRole('button', {name: 'Load settings'}).click();
  await page.waitForFunction(() => window.loadedSettingsOutput === 'shot-2-new.mp4');

  await page.evaluate(() => window.store.getState().setSelectedOutput(2));
  assert.equal(await page.evaluate(() => {
    const state = window.store.getState(); return state.filteredOutputs()[state.selectedOutput]?.name;
  }), 'shot-2-new.mp4', 'selection indexes address flattened child cards');
  await parentCard().getByRole('button', {name: 'Hide clips'}).click();
  await page.waitForFunction(() => Object.values(window.store.getState().directorGalleryGroups).some(group => !group.expanded));
  assert.equal(await page.evaluate(() => {
    const state = window.store.getState(); return state.filteredOutputs()[state.selectedOutput]?.name;
  }), 'joined-final.mp4', 'collapsing a selected child moves selection to its final parent');
  assert.equal(await page.locator('[data-feed-index]').count(), 2);
  await assertCollapsedGalleryLayout('after collapse');
  await page.screenshot({path: path.join(artifacts, `${label}-collapsed.png`)});

  // Re-expand and delete one of two takes for the same shot. The parent count
  // counts files/takes, while the surviving take remains visible.
  await parentCard().getByRole('button', {name: 'Show 4 clips'}).click();
  await page.waitForFunction(() => window.store.getState().filteredOutputs().length === 6);
  const latestTake = page.locator('[data-output-name="shot-2-new.mp4"]');
  await latestTake.getByRole('button', {name: 'More clip actions'}).click();
  const menu = page.getByRole('menu', {name: 'Clip actions'});
  await menu.getByRole('menuitem', {name: 'Delete output', exact: true}).click();
  await menu.getByRole('menuitem', {name: 'Click again to delete output'}).click();
  await page.waitForFunction(() => {
    const state = window.store.getState();
    return state.filteredOutputs().length === 5
      && state.filteredOutputs().find(file => file.name === 'joined-final.mp4')?.director_group?.clip_count === 3;
  });
  assert.equal(await page.locator('[data-output-name="shot-2-old.mp4"]').count(), 1, 'deleting one take preserves the other take for that shot');
  shots = shots.filter(file => file.name !== 'shot-2-new.mp4');
  await page.evaluate(() => { window.fixtureShots = window.fixtureShots.filter(file => file.name !== 'shot-2-new.mp4'); });

  const movedTake = page.locator('[data-output-name="shot-1.mp4"]');
  await movedTake.getByRole('button', {name: 'More clip actions'}).click();
  const moveMenu = page.getByRole('menu', {name: 'Clip actions'});
  await moveMenu.getByRole('menuitem', {name: 'Move to workspace'}).click();
  await moveMenu.getByRole('menuitem', {name: 'Other', exact: true}).click();
  await page.waitForFunction(() => {
    const state = window.store.getState();
    return state.filteredOutputs().length === 4
      && state.filteredOutputs().find(file => file.director_group)?.director_group?.clip_count === 2;
  });
  assert.equal(await page.locator('[data-output-name="shot-1.mp4"]').count(), 0,
    'moving a cached child removes it from the expanded source workspace');
  await page.evaluate(() => { window.fixtureShots = window.fixtureShots.filter(file => file.name !== 'shot-1.mp4'); });

  // Dashboard shortcut selects the saved pipeline, and both parent and child
  // rejoin actions use the current Director pipeline endpoint rather than the
  // filesystem-order generic multiclip rejoin endpoint.
  await parentCard().getByRole('button', {name: 'Edit Director shots'}).click();
  await page.waitForFunction(() => window.store.getState().dashboardOpen
    && window.store.getState().dashboardSelectedPipeline?.pipeline_id === 'director-pipeline-1');
  const latestTakeCard = page.locator('[data-output-name="shot-2-old.mp4"]');
  await latestTakeCard.getByRole('button', {name: 'More clip actions'}).click();
  await page.getByRole('menu', {name: 'Clip actions'}).getByRole('menuitem', {name: 'Rejoin Director shots'}).click();
  await page.waitForFunction(() => window.store.getState().outputs.some(file => file.name === 'joined-final-1.mp4'));
  assert.ok(requests.some(request => request.path.endsWith('/director-pipeline-1/rejoin') && request.method === 'POST'));
  assert.equal(requests.some(request => request.path === '/api/v1/outputs/rejoin'), false,
    'Director shot takes never use the generic filesystem-order rejoin route');

  const latestGroupCard = () => page.locator('[data-feed-index]').filter({has: page.locator('[data-director-group-id="director-pipeline-1"]')});
  await latestGroupCard().getByRole('button', {name: 'Rejoin'}).click();
  await page.waitForFunction(() => window.store.getState().outputs.some(file => file.name === 'joined-final-2.mp4'));

  // Reconcile after deleting the joined final: a refresh can surface restored
  // standalone shots or a recovered parent without losing the selected media.
  const rootDeleteTarget = page.locator('[data-output-name="joined-final-2.mp4"]');
  if (await rootDeleteTarget.count()) {
    await rootDeleteTarget.getByRole('button', {name: 'More clip actions'}).click();
    const deleteMenu = page.getByRole('menu', {name: 'Clip actions'});
    await deleteMenu.getByRole('menuitem', {name: 'Delete output', exact: true}).click();
    await deleteMenu.getByRole('menuitem', {name: 'Click again to delete output'}).click();
    await page.waitForFunction(() => !window.store.getState().outputs.some(file => file.name === 'joined-final-2.mp4'));
    assert.ok(requests.some(request => request.path === '/api/v1/outputs' && request.search.includes('group_director_clips=true')),
      'deleting a joined parent refreshes the grouped root listing');
  }

  // A request begun in Saved must not publish after Saved→Other→Saved, even
  // though the final query string equals the initial query.
  joinedCount = 0;
  deletedNames.delete('joined-final.mp4');
  await page.evaluate(() => {
    const state = window.store.getState();
    window.store.setState({activeWorkspace: 'Saved', browsingUploads: false, browsingAllFolders: false,
      outputs: [{...window.fixtureParent, director_group: {pipeline_id: 'director-pipeline-1', clip_count: window.fixtureShots.length}},
        window.fixtureOrdinary], outputsTotal: 2, selectedOutput: 0,
      directorGalleryGroups: {}});
  });
  holdMemberFetch = true;
  await page.locator('[data-output-name="joined-final.mp4"]').getByRole('button', {name: /Show \d+ clips/}).click();
  while (!releaseMemberFetch) await new Promise(resolve => setTimeout(resolve, 10));
  await page.evaluate(async () => {
    window.store.setState({activeWorkspace: 'Other'});
    await window.store.getState().loadOutputs();
    window.store.setState({activeWorkspace: 'Saved'});
    await window.store.getState().loadOutputs();
  });
  releaseMemberFetch();
  holdMemberFetch = false;
  await page.waitForFunction(() => Object.values(window.store.getState().directorGalleryGroups)
    .some(group => group.error && !group.loaded && !group.loading));
  assert.equal(await page.locator('[data-output-name="joined-final.mp4"]')
    .getByRole('button', {name: 'Retry loading clips'}).count(), 1,
  'a stale member request leaves a one-click retry control');
  await page.locator('[data-output-name="joined-final.mp4"]')
    .getByRole('button', {name: 'Retry loading clips'}).click();
  await page.waitForFunction(expected => Object.values(window.store.getState().directorGalleryGroups)
    .some(group => group.loaded && group.clips.length === expected), shots.length);

  assert.deepEqual(errors, [], `${label} has no browser errors`);
  console.log(`PASS Director gallery groups ${label}: final-first expansion, sorted takes, flattened actions, identity, stale scope and safe rejoin`);
  await page.close();
}

(async () => {
  const bundle = await esbuild.build({absWorkingDir: path.join(root, 'ui'), stdin: {contents: bundleSource, resolveDir: path.join(root, 'ui'), loader: 'tsx'},
    bundle: true, write: false, jsx: 'automatic', define: {'process.env.NODE_ENV': '"development"'}, logLevel: 'silent'});
  const assets = path.join(root, 'ui/dist/assets');
  const css = fs.readFileSync(path.join(assets, fs.readdirSync(assets).find(name => name.endsWith('.css'))), 'utf8');
  const browser = await chromium.launch({headless: true, ignoreDefaultArgs: ['--hide-scrollbars'],
    ...(process.platform === 'win32' ? {executablePath: process.env.MAESTRO_CHROME || 'C:/Program Files/Google/Chrome/Application/chrome.exe'} : {})});
  try {
    for (const [label, viewport] of [
      ['desktop', {width: 1280, height: 900}],
      ['mobile', {width: 390, height: 844}],
    ]) {
      await runViewport(browser, viewport, label, bundle.outputFiles[0].text, css);
    }
  } finally {
    await browser.close();
  }
})().catch(error => {console.error(error); process.exitCode = 1;});
