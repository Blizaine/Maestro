const assert = require('node:assert/strict');
const path = require('node:path');

async function assertGalleryLibrary(page, output) {
  const calls = [];
  const items = ['Folder-A', 'Folder-B'].map((workspace, i) => ({
    name: 'same.png', id: `${workspace}/same.png`, workspace, path: `C:/test/${workspace}/same.png`,
    url: `/api/v1/file/same.png?workspace=${workspace}`, type: 'image', created_at: 2-i, size: 50,
    mode: 'image', favorite: false, metadata_ready: true,
  }));
  let releaseSlow;
  await page.route('**/api/v1/**', async route => {
    const url = new URL(route.request().url());
    calls.push({url: url.pathname + url.search, method: route.request().method()});
    const json = body => route.fulfill({contentType: 'application/json', body: JSON.stringify(body)});
    if (url.pathname === '/api/v1/outputs') {
      if (url.searchParams.get('search') === 'slow') {
        await new Promise(resolve => {releaseSlow = resolve;});
        return json({outputs: [{...items[0], name: 'stale.png'}], total: 1});
      }
      return json({outputs: items, total: 202, next_cursor: 'page-two'});
    }
    if (url.pathname.includes('/metadata')) return json({source: 'sidecar', params: {prompt: `Scene ${url.searchParams.get('workspace')}`, seed: 4}});
    if (url.pathname.includes('/favorites/')) return json({favorite: true});
    if (url.pathname.includes('/file/')) return route.fulfill({contentType: 'image/svg+xml', body: '<svg xmlns="http://www.w3.org/2000/svg" width="640" height="360"><rect width="640" height="360" fill="#141c2a"/></svg>'});
    return route.fallback();
  });
  await page.evaluate(() => {
    window.reactRoot.unmount();
    document.getElementById('root').style.cssText = 'height:100vh;width:100vw;display:flex';
    window.store.setState({outputs: [], outputsTotal: 0, jobs: [], pipelineStatus: null,
      activeWorkspace: 'Destination', browsingUploads: false, browsingAllFolders: false,
      mediaFilter: 'all', outputSearchQuery: '', isGenerating: false,
      workspaces: ['default', 'Destination', 'Folder-A', 'Folder-B'].map(name => ({name, path:name}))});
    window.mountGallery();
  });
  await page.getByTitle('Switch workspace').click();
  await page.getByRole('button', {name: 'All folders', exact: true}).click();
  await page.waitForFunction(() => window.store.getState().outputs.length === 2);
  assert.equal(await page.evaluate(() => window.store.getState().activeWorkspace), 'Destination');
  assert.equal(calls.some(call => call.url.includes('/workspaces/active')), false);
  assert.equal(await page.locator('[data-feed-index]').count(), 2, 'duplicate basenames render independently');
  await page.getByTitle('Open Folder-B').waitFor();
  await page.evaluate(async () => {
    const store = window.store.getState();
    await store.toggleFavorite('same.png', 'Folder-B');
    await store.loadOutputMetadata('same.png', 'Folder-B');
  });
  assert.deepEqual(await page.evaluate(() => window.store.getState().outputs.map(o => o.favorite)), [false, true]);
  assert.equal(await page.evaluate(() => window.store.getState().selectedOutputMeta.params.prompt), 'Scene Folder-B');
  await page.evaluate(() => window.store.getState().setOutputSearchQuery('old harbor'));
  await page.waitForFunction(() => !window.store.getState().outputsLoading);
  await page.evaluate(() => window.store.getState().setMediaFilter('images'));
  await page.waitForFunction(() => !window.store.getState().outputsLoading);
  await page.evaluate(() => window.store.getState().loadMoreOutputs());
  assert.ok(calls.some(call => /cursor=page-two/.test(call.url) && /media_filter=images/.test(call.url) && /search=old\+harbor/.test(call.url) && /workspace=__all__/.test(call.url)));
  await page.evaluate(() => window.store.getState().setOutputSearchQuery('slow'));
  while (!releaseSlow) await new Promise(resolve => setTimeout(resolve, 10));
  await page.evaluate(() => window.store.getState().setOutputSearchQuery('new'));
  await page.waitForFunction(() => !window.store.getState().outputsLoading);
  releaseSlow();
  await page.waitForTimeout(80);
  assert.equal(await page.evaluate(() => window.store.getState().outputs[0].name), 'same.png');
  await page.setViewportSize({width: 390, height: 844});
  await page.screenshot({path: path.join(output, 'mobile-all-folders.png')});
  assert.ok(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth), 'gallery stays within mobile width');
  assert.ok(calls.filter(call => call.url.startsWith('/api/v1/outputs?')).every(call => /limit=100/.test(call.url)), 'browser never asks for an unbounded library');
  console.log('All folders: identity, destination, filters, pagination and stale-query checks passed');
  await assertGalleryRefreshOrder(page);
}

async function assertGalleryRefreshOrder(page) {
  // Exercise the real store used by running-job polls, without mounting the
  // viewport observer (which deliberately selects the item being browsed).
  await page.evaluate(() => window.reactRoot.unmount());
  const media = (name, created_at, workspace = 'Folder-A') => ({
    name, created_at, workspace, id: `${workspace}/${name}`, type: 'image',
    path: `C:/test/${workspace}/${name}`, url: `/api/v1/file/${name}?workspace=${workspace}`,
    size: 50, mode: 'image', favorite: false, metadata_ready: true,
  });
  let reply;
  await page.route('**/api/v1/outputs?*', async route => {
    const snapshot = reply;
    snapshot.arrived?.();
    if (snapshot.wait) await snapshot.wait;
    return route.fulfill({contentType: 'application/json', body: JSON.stringify(snapshot.data)});
  });
  const prime = (outputs, selectedOutput = 0, total = outputs.length, cursor = null) => page.evaluate(
    ({outputs, selectedOutput, total, cursor}) => {
      window.store.setState({outputs, outputsTotal: total, outputsCursor: cursor,
        selectedOutput, outputsLoading: false, selectedOutputMeta: null,
        activeWorkspace: 'Folder-A', browsingAllFolders: false, browsingUploads: false,
        mediaFilter: 'all', outputSearchQuery: '', isGenerating: true,
        jobs: [{id: 'running-test', status: 'running', outputFiles: []}]});
    }, {outputs, selectedOutput, total, cursor});
  const refresh = async (outputs, total = outputs.length, cursor = null) => {
    reply = {data: {outputs, total, next_cursor: cursor}};
    await page.evaluate(() => window.store.getState().refreshOutputs());
  };
  const names = () => page.evaluate(() => window.store.getState().outputs.map(item => item.name));
  const selected = () => page.evaluate(() => {
    const state = window.store.getState();
    return state.filteredOutputs()[state.selectedOutput]?.name;
  });

  const newest = media('newest.png', 300), middle = media('middle.png', 200), old = media('old.png', 100);
  await prime([newest, middle], 1);
  await page.evaluate(() => {
    window.gallerySelections = [];
    window.stopGallerySelectionAudit = window.store.subscribe(state => {
      window.gallerySelections.push(state.filteredOutputs()[state.selectedOutput]?.name);
    });
  });
  await refresh([newest, middle, old]);
  assert.deepEqual(await names(), ['newest.png', 'middle.png', 'old.png'],
    'a newly discovered OLD output must not jump above newer generations during a running-job refresh');
  assert.equal(await selected(), 'middle.png');
  const latest = media('latest.png', 400);
  await refresh([latest, newest, middle, old]);
  await refresh([latest, newest, middle, old]);
  assert.deepEqual(await names(), ['latest.png', 'newest.png', 'middle.png', 'old.png']);
  assert.ok((await page.evaluate(() => window.gallerySelections)).every(name => name === 'middle.png'),
    'each published store state must retain the selected identity while a new result arrives');
  await page.evaluate(() => window.stopGallerySelectionAudit());

  // Updated file timestamps and removed partial windows must follow the same
  // authoritative ordering/visibility as a full reload.
  const updated = {...old, created_at: 500};
  await refresh([updated, latest, newest, middle]);
  assert.deepEqual(await names(), ['old.png', 'latest.png', 'newest.png', 'middle.png']);
  await refresh([latest, newest, middle]);
  assert.deepEqual(await names(), ['latest.png', 'newest.png', 'middle.png'], 'removed/hidden items must not persist after a complete refresh');
  assert.equal(await selected(), 'middle.png');

  // Only the first page is refreshed. Keep already browsed older pages and
  // their cursor, but discard a removed item within the authoritative head.
  const catalogue = Array.from({length: 115}, (_, i) => media(`item-${i}.png`, 1000 - i));
  await prime(catalogue.slice(0, 105), 102, 115, 'loaded-tail');
  const incoming = media('incoming.png', 1100);
  const remaining = catalogue.filter((_, i) => i !== 20);
  await refresh([incoming, ...remaining.slice(0, 99)], 115, 'head-cursor');
  assert.deepEqual(await names(), [incoming, ...catalogue.slice(0, 105).filter((_, i) => i !== 20)].map(item => item.name));
  assert.equal(await selected(), 'item-102.png');
  assert.equal(await page.evaluate(() => window.store.getState().outputsCursor), 'loaded-tail');
  reply = {data: {outputs: catalogue.slice(102), total: 115, next_cursor: null}};
  await page.evaluate(() => window.store.getState().loadMoreOutputs());
  assert.deepEqual(await names(), [incoming, ...remaining].map(item => item.name), 'pagination overlap must not duplicate or reorder existing output');

  const tiedA = media('same.png', 700, 'Folder-A'), tiedZ = media('same.png', 700, 'Folder-Z');
  const tail = media('older.png', 699);
  await prime([tiedA, tail], 0, 3, 'tail-cursor');
  await page.evaluate(() => window.store.setState({browsingAllFolders: true}));
  await refresh([tiedZ, tiedA], 3, 'head-cursor');
  assert.deepEqual(await page.evaluate(() => window.store.getState().outputs.map(item => item.id)),
    ['Folder-Z/same.png', 'Folder-A/same.png', 'Folder-A/older.png'], 'equal timestamps and duplicate filenames retain folder identity');
  assert.equal(await page.evaluate(() => {
    const state = window.store.getState(); return state.outputs[state.selectedOutput].workspace;
  }), 'Folder-A');

  // Slow older polls must not resurrect a stale snapshot after a later poll
  // has already returned the current first page.
  await prime([newest, middle, old], 1);
  let releaseOlder, markArrived;
  const arrived = new Promise(resolve => { markArrived = resolve; });
  reply = {data: {outputs: [newest, middle, old], total: 3, next_cursor: null},
    arrived: markArrived, wait: new Promise(resolve => { releaseOlder = resolve; })};
  await page.evaluate(() => {
    window.galleryOlderRefreshDone = false;
    void window.store.getState().refreshOutputs().finally(() => { window.galleryOlderRefreshDone = true; });
  });
  await arrived;
  await refresh([latest, newest, middle, old]);
  releaseOlder();
  await page.waitForFunction(() => window.galleryOlderRefreshDone);
  assert.deepEqual(await names(), ['latest.png', 'newest.png', 'middle.png', 'old.png'], 'late old refresh must not roll back the latest response');
  assert.equal(await selected(), 'middle.png');
  console.log('Gallery running-job refresh: chronology, atomic selection, authoritative removals, pagination and overlapping polls passed');
}

module.exports = {assertGalleryLibrary};
