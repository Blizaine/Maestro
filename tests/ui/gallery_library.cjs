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
}

module.exports = {assertGalleryLibrary};
