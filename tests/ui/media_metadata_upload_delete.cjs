// Focused gallery metadata and upload-delete UI check. All API calls are
// intercepted; the test never touches the running Maestro service or media.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const root = path.resolve(__dirname, '../..');
const esbuild = require(path.join(root, 'ui/node_modules/esbuild'));
const playwright = require(process.env.MAESTRO_PLAYWRIGHT ||
  'C:/Users/bliza/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');

const processedAt = 1780000012;
const uploadedAt = 1780000456;
const fileDate = 1780000789;
const files = {
  processed: {name: 'upscaled.mp4', workspace: 'Workspace-A', id: 'Workspace-A/upscaled.mp4',
    type: 'video', mode: 'video', path: 'C:/test/upscaled.mp4', url: '/api/v1/file/upscaled.mp4?workspace=Workspace-A',
    size: 100, created_at: 1779999999, favorite: false, metadata_ready: true},
  legacy: {name: 'legacy-upscale.png', workspace: 'Workspace-A', id: 'Workspace-A/legacy-upscale.png',
    type: 'image', mode: 'image', path: 'C:/test/legacy-upscale.png', url: '/api/v1/file/legacy-upscale.png?workspace=Workspace-A',
    size: 200, created_at: fileDate, favorite: false, metadata_ready: true},
  legacyLanczos: {name: 'legacy-lanczos.png', workspace: 'Workspace-A', id: 'Workspace-A/legacy-lanczos.png',
    type: 'image', mode: 'image', path: 'C:/test/legacy-lanczos.png', url: '/api/v1/file/legacy-lanczos.png?workspace=Workspace-A',
    size: 200, created_at: fileDate, favorite: false, metadata_ready: true},
  upload: {name: 'held-upload.mp4', workspace: '__uploads__', id: '__uploads__/held-upload.mp4',
    type: 'video', mode: null, path: 'C:/test/held-upload.mp4', url: '/api/v1/file/held-upload.mp4?workspace=__uploads__',
    size: 300, created_at: uploadedAt, favorite: false, metadata_ready: true},
};
const metadata = {
  'upscaled.mp4': {source: 'sidecar', params: null, timestamp: {value: processedAt, kind: 'processed'},
    media_info: {width: 640, height: 360, fps: 48, frames: 48, duration_seconds: 1, size_bytes: 123456},
    processing: {method: 'dlss5*2', method_label: 'DLSS 5 Neural Rendering', multiplier: 2,
      temporal_method: 'dlss_frame_generation', temporal_label: 'DLSS Frame Generation', frame_multiplier: 2,
      source_name: 'inputs/source.mp4', input: {width: 320, height: 180, fps: 24, frames: 24},
      output: {width: 640, height: 360, fps: 48, frames: 48}, elapsed_seconds: 8,
      completed_at: processedAt, options: {dlss_intensity: 0.7, dlss_depth: 'half', dlss_motion: 'original'}}},
  'legacy-upscale.png': {source: 'sidecar', params: {edit_sub_mode: 'upscale', method: 'flashvsr2pass4'},
    tool: 'upscale', tool_source: 'uploads/original.png'},
  'legacy-lanczos.png': {source: 'sidecar', params: {edit_sub_mode: 'upscale', method: 'lanczos1.5'}, tool: 'upscale'},
  'held-upload.mp4': {source: 'sidecar', params: null,
    timestamp: {value: uploadedAt, kind: 'uploaded'}, media_info: {width: 640, height: 360, size_bytes: 300}},
};

(async () => {
  const bundle = await esbuild.build({stdin: {contents: `
    import React from 'react'; import {createRoot} from 'react-dom/client';
    import {useStore} from './src/stores/useStore';
    import {MediaFeedItem} from './src/components/MainContent/MediaFeedItem';
    window.store=useStore; const root=createRoot(document.getElementById('root'));
    window.mount=(file,uploads)=>{
      useStore.setState({outputs:[file],outputsTotal:1,selectedOutput:0,activeWorkspace:'Workspace-A',
        browsingUploads:uploads,browsingAllFolders:false,mediaFilter:'all',outputSearchQuery:'',workspaces:[],models:[],
        jobs:[],isGenerating:false,isEnhancing:false,servicesConfig:{}});
      root.render(<MediaFeedItem key={file.name} file={file} index={0} isActive={true} onActivate={()=>{}}
        onPlaybackStart={()=>{}} onMeasured={()=>{}}/>);
    };`, resolveDir: path.join(root, 'ui'), loader: 'tsx'}, bundle: true, write: false,
    jsx: 'automatic', define: {'process.env.NODE_ENV': '"development"'}, logLevel: 'silent'});
  const assets = path.join(root, 'ui/dist/assets');
  const cssName = fs.readdirSync(assets).find(name => name.endsWith('.css'));
  assert.ok(cssName, 'UI build CSS exists for gallery card layout');
  const css = fs.readFileSync(path.join(assets, cssName), 'utf8');
  const browser = await playwright.chromium.launch({headless: true,
    ...(process.platform === 'win32' ? {executablePath: process.env.MAESTRO_CHROME || 'C:/Program Files/Google/Chrome/Application/chrome.exe'} : {})});
  try {
    const page = await browser.newPage({viewport: {width: 1000, height: 900}});
    page.setDefaultTimeout(5000);
    const errors = [], mutations = [];
    page.on('pageerror', error => errors.push(error.message));
    await page.addInitScript(() => {
      // Give the failure path a deterministic active media element so the UI's
      // pause/unload/restore behavior can be checked without a real media file.
      Object.defineProperty(HTMLMediaElement.prototype, 'paused', {configurable: true,
        get() { return this.dataset.testPlaying !== 'true'; }});
      HTMLMediaElement.prototype.play = function() { this.dataset.testPlaying = 'true'; return Promise.resolve(); };
      HTMLMediaElement.prototype.pause = function() { this.dataset.testPlaying = 'false'; };
      HTMLMediaElement.prototype.load = function() {
        if (this.getAttribute('src')) setTimeout(() => this.dispatchEvent(new Event('loadedmetadata')), 0);
      };
    });
    await page.route('**/*', async route => {
      const request = route.request(), url = new URL(request.url());
      const json = body => route.fulfill({json: body});
      if (url.pathname.endsWith('/metadata')) {
        const name = decodeURIComponent(url.pathname.split('/').at(-2));
        return json(metadata[name]);
      }
      if (request.method() === 'DELETE') {
        mutations.push({method: request.method(), path: url.pathname, query: url.search});
        return route.fulfill({status: 409, json: {detail: 'This upload is in use by a running job.'}});
      }
      if (url.pathname.startsWith('/api/v1/thumbnail/')) return route.fulfill({contentType: 'image/svg+xml', body: '<svg xmlns="http://www.w3.org/2000/svg" width="2" height="2"/>'});
      if (url.pathname.startsWith('/api/v1/file/')) return route.fulfill({status: 200, contentType: 'video/mp4', body: ''});
      if (url.pathname === '/') return route.fulfill({contentType: 'text/html', body: '<meta name="viewport" content="width=device-width,initial-scale=1"><div id="root" style="height:100vh"></div>'});
      return route.fulfill({json: {}});
    });
    await page.goto('http://media-details.test');
    await page.addStyleTag({content: css});
    await page.addScriptTag({content: bundle.outputFiles[0].text});

    await page.evaluate(file => window.mount(file, false), files.processed);
    const processedCard = page.locator('[data-feed-index="0"]');
    const expectedProcessedDate = await page.evaluate(value => new Intl.DateTimeFormat(undefined,
      {dateStyle: 'medium', timeStyle: 'short'}).format(new Date(value * 1000)), processedAt);
    await page.waitForFunction(expected => document.querySelector('[data-feed-index="0"]')?.textContent?.includes(expected), expectedProcessedDate);
    assert.ok((await processedCard.innerText()).includes(expectedProcessedDate), 'every media card shows local date and time');
    await processedCard.getByRole('button', {name: 'Show media details'}).click();
    await processedCard.getByText('DLSS 5 Neural Rendering', {exact: true}).waitFor();
    await processedCard.getByText('Resolution change', {exact: true}).waitFor();
    assert.ok((await processedCard.innerText()).includes('640 × 360'), 'Info shows before and after resolution');
    assert.ok((await processedCard.innerText()).includes('48 fps'), 'Info shows frame rate');
    assert.ok((await processedCard.innerText()).includes('24 fps → 48 fps'), 'Processing shows recorded frame-rate change');
    assert.ok((await processedCard.innerText()).includes('DLSS Frame Generation'), 'Info shows the temporal method');
    assert.ok((await processedCard.innerText()).includes('Source\nsource.mp4'), 'Info shows the source name');
    const screenshotPath = path.join(root, '.codex-tmp/media-metadata-ui/mobile-info.png');
    fs.mkdirSync(path.dirname(screenshotPath), {recursive: true});
    await page.setViewportSize({width: 390, height: 844});
    await processedCard.screenshot({path: screenshotPath});
    await processedCard.getByText('Processing options', {exact: true}).click();
    assert.ok((await processedCard.innerText()).includes('Enhancement strength'), `DLSS options use friendly labels\n${await processedCard.innerText()}`);

    await page.evaluate(file => window.mount(file, false), files.legacy);
    const legacyCard = page.locator('[data-feed-index="0"]');
    const expectedFileDate = await page.evaluate(value => new Intl.DateTimeFormat(undefined,
      {dateStyle: 'medium', timeStyle: 'short'}).format(new Date(value * 1000)), fileDate);
    assert.ok((await legacyCard.innerText()).includes(expectedFileDate), 'legacy media uses the file date fallback');
    await legacyCard.getByRole('button', {name: 'Show media details'}).click();
    await legacyCard.getByText('FlashVSR two-pass 4×', {exact: true}).waitFor();
    assert.ok((await legacyCard.innerText()).includes('original.png'), 'legacy upscale source stays visible');

    await page.evaluate(file => window.mount(file, false), files.legacyLanczos);
    const lanczosCard = page.locator('[data-feed-index="0"]');
    await lanczosCard.getByRole('button', {name: 'Show media details'}).click();
    await lanczosCard.getByText('Lanczos 1.5×', {exact: true}).waitFor();

    await page.evaluate(file => window.mount(file, true), files.upload);
    const uploadCard = page.locator('[data-feed-index="0"]');
    await uploadCard.getByRole('button', {name: 'More clip actions'}).click();
    const uploadMenu = uploadCard.getByRole('menu', {name: 'Clip actions'});
    await uploadMenu.getByRole('menuitem', {name: 'Delete upload', exact: true}).waitFor();
    assert.ok(await uploadMenu.getByText('Removes this source file from Uploads. Completed outputs stay.').isVisible());
    assert.equal(await uploadMenu.getByRole('menuitem', {name: 'Move to workspace'}).count(), 0, 'uploads still cannot be moved');
    assert.equal(await uploadCard.getByRole('button', {name: 'Add to favorites'}).count(), 0, 'uploads still cannot be favorited');
    const video = uploadCard.locator('video');
    await video.evaluate(element => element.play());
    await uploadMenu.getByRole('menuitem', {name: 'Delete upload', exact: true}).click();
    await uploadMenu.getByRole('menuitem', {name: 'Click again to delete upload', exact: true}).click();
    await uploadCard.getByRole('alert').getByText('This upload is in use by a running job.').waitFor();
    assert.equal(await video.getAttribute('src'), files.upload.url, 'failed deletion restores the media URL');
    assert.equal(await video.getAttribute('data-test-playing'), 'true', 'failed deletion resumes prior playback');
    assert.deepEqual(mutations, [{method: 'DELETE', path: '/api/v1/uploads/held-upload.mp4', query: ''}],
      'upload delete uses the constrained upload endpoint without a workspace query');
    assert.deepEqual(errors, [], 'metadata and delete error display without browser exceptions');
    console.log('PASS gallery media facts, processing metadata, timestamps, legacy upscale details, and upload delete recovery');
  } finally {
    await browser.close();
  }
})().catch(error => {console.error(error); process.exitCode = 1;});
