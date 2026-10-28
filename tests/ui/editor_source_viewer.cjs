// Isolated Editor source-preview and trim-range regression harness.
// Every HTTP request is mocked; no media is generated or downloaded.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const esbuild = require(path.join(__dirname, '../../ui/node_modules/esbuild'));
const playwright = require(process.env.MAESTRO_PLAYWRIGHT || 'playwright');
const uiRoot = path.resolve(__dirname, '../../ui');
const uiNodeModules = fs.realpathSync(path.join(uiRoot, 'node_modules'));

function makeProject() {
  return {
    schema_version: 5,
    id: 'source-viewer-project',
    name: 'Source viewer fixture',
    workspace: 'Project workspace',
    revision: 1,
    created_at: 1,
    updated_at: 1,
    canvas: {width: 1920, height: 1080, fps: 24, background: '#000000'},
    assets: {},
    tracks: [
      {id: 'selected-video-track', name: 'Selected video', type: 'video', z_index: 0, muted: false, locked: false, volume: 1, items: []},
      {id: 'other-video-track', name: 'Other video', type: 'video', z_index: 1, muted: false, locked: false, volume: 1, items: []},
      {id: 'audio-track', name: 'Audio', type: 'audio', z_index: 0, muted: false, locked: false, volume: 1, items: []},
    ],
    markers: [],
    export: {quality: 'high', codec: 'h264', encoder: 'auto', include_audio: true, resolution: 'canvas', frame_rate: 'project', filename: ''},
    exports: [],
  };
}

function makeAsset(id, name, type, duration) {
  return {
    id,
    name,
    display_name: name,
    type,
    origin: 'output',
    workspace: 'Source workspace',
    path: `/outputs/${name}`,
    url: `/media/${name}`,
    duration,
    width: type === 'image' ? 640 : 1280,
    height: type === 'image' ? 480 : 720,
    fps: type === 'video' ? 24 : 0,
    has_audio: type === 'video',
    size: 100,
  };
}

(async () => {
  const bundle = await esbuild.build({
    absWorkingDir: path.dirname(uiNodeModules),
    stdin: {
      contents: `
        import React, {useState} from 'react';
        import {createRoot} from 'react-dom/client';
        import {EditorSourceViewer} from './src/editor/EditorSourceViewer';
        import {useEditorStore} from './src/editor/useEditorStore';
        window.editorStore = useEditorStore;
        const video = {id:'unknown-video',name:'unknown.mp4',display_name:'unknown.mp4',type:'video',origin:'output',workspace:'Source workspace',path:'/outputs/unknown.mp4',url:'/media/unknown.mp4',duration:0,width:1280,height:720,fps:24,has_audio:true,size:100};
        const image = {id:'still-image',name:'still.png',display_name:'still.png',type:'image',origin:'output',workspace:'Source workspace',path:'/outputs/still.png',url:'/media/still.png',duration:0,width:640,height:480,fps:0,has_audio:false,size:80};
        function App() {
          const [open, setOpen] = useState(false);
          const [asset, setAsset] = useState(video);
          return <main>
            <button type="button" onClick={() => {setAsset(video); setOpen(true)}}>Open video source</button>
            <button type="button" onClick={() => {setAsset(image); setOpen(true)}}>Open image source</button>
            {open && <EditorSourceViewer key={asset.id} asset={asset} onClose={() => setOpen(false)} />}
          </main>;
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
  const page = await browser.newPage({viewport: {width: 1100, height: 850}});
  const probes = [];
  const apiRequests = [];
  const pageErrors = [];

  try {
    await page.addInitScript(() => {
      const state = new WeakMap();
      const mediaState = media => {
        if (!state.has(media)) state.set(media, {currentTime: 0});
        return state.get(media);
      };
      Object.defineProperties(HTMLMediaElement.prototype, {
        duration: {configurable: true, get: () => Number.NaN},
        currentTime: {
          configurable: true,
          get() { return mediaState(this).currentTime; },
          set(value) { mediaState(this).currentTime = Number(value); },
        },
      });
    });
    page.on('pageerror', error => pageErrors.push(error.message));
    await page.route('**/*', async route => {
      const request = route.request();
      const url = new URL(request.url());
      const json = (body, status = 200) => route.fulfill({status, contentType: 'application/json', body: JSON.stringify(body)});

      if (url.pathname === '/' || url.pathname === '/index.html') {
        return route.fulfill({contentType: 'text/html', body: '<meta name="viewport" content="width=device-width,initial-scale=1"><div id="root"></div>'});
      }
      if (url.pathname.startsWith('/api/')) {
        apiRequests.push({method: request.method(), path: url.pathname});
        if (url.pathname === '/api/v1/editor/media/probe') {
          const body = request.postDataJSON();
          probes.push(body);
          return json({
            name: body.asset.name,
            type: body.asset.name.endsWith('.png') ? 'image' : 'video',
            workspace: body.workspace,
            path: body.asset.path,
            duration: body.asset.name.endsWith('.png') ? 0 : 12,
            width: body.asset.name.endsWith('.png') ? 640 : 1280,
            height: body.asset.name.endsWith('.png') ? 480 : 720,
            fps: body.asset.name.endsWith('.png') ? 0 : 24,
            has_audio: !body.asset.name.endsWith('.png'),
            size: 100,
          });
        }
        throw new Error(`Unexpected API request: ${request.method()} ${url.pathname}`);
      }
      if (url.pathname.endsWith('.mp4') || url.pathname.endsWith('.png')) return route.fulfill({status: 204, body: ''});
      return route.fulfill({status: 204, body: ''});
    });

    await page.goto('http://editor-source-viewer.test');
    await page.addScriptTag({content: bundle.outputFiles[0].text});
    await page.evaluate(project => {
      window.editorStore.setState({
        workspace: project.workspace,
        project,
        selectedItemId: null,
        selectedItemIds: [],
        selectedTrackId: 'selected-video-track',
        playhead: 4.75,
        dirty: false,
        history: [],
        future: [],
        error: null,
      });
      window.mount();
    }, makeProject());

    const snapshot = () => page.evaluate(() => JSON.stringify({
      project: window.editorStore.getState().project,
      dirty: window.editorStore.getState().dirty,
      history: window.editorStore.getState().history,
      future: window.editorStore.getState().future,
      selectedItemId: window.editorStore.getState().selectedItemId,
      selectedTrackId: window.editorStore.getState().selectedTrackId,
      playhead: window.editorStore.getState().playhead,
    }));
    const beforeOpen = await snapshot();
    const videoLauncher = page.getByRole('button', {name: 'Open video source'});
    await videoLauncher.click();
    const dialog = page.getByRole('dialog', {name: 'Source preview'});
    await dialog.waitFor();
    await page.waitForFunction(() => document.querySelector('[aria-label="Source out"]')?.value === '12');
    assert.equal(probes.length, 1, 'Unknown-duration video triggers one metadata probe');
    assert.equal(probes[0].workspace, 'Project workspace', 'Probe is performed in the current project workspace');
    assert.equal(probes[0].asset.workspace, 'Source workspace', 'Probe retains the asset source workspace');
    assert.equal(await dialog.getByLabel('Source in').inputValue(), '0');
    assert.equal(await dialog.getByLabel('Source out').inputValue(), '12', 'The probe result supplies the unknown source duration');

    assert.equal(await page.evaluate(() => document.activeElement?.getAttribute('aria-label')), 'Close source preview', 'Opening moves focus into the dialog');
    await page.keyboard.press('Shift+Tab');
    assert.equal(await page.evaluate(() => document.activeElement?.textContent?.trim()), 'Insert at playhead', 'Shift+Tab wraps from the first control to the last');
    await page.keyboard.press('Tab');
    assert.equal(await page.evaluate(() => document.activeElement?.getAttribute('aria-label')), 'Close source preview', 'Tab wraps from the last control to the first');
    await page.keyboard.press('Escape');
    await dialog.waitFor({state: 'detached'});
    assert.equal(await page.evaluate(() => document.activeElement?.textContent?.trim()), 'Open video source', 'Closing restores focus to the launcher');
    assert.equal(await snapshot(), beforeOpen, 'Opening and closing the source viewer changes no project or timeline state');
    assert.equal(probes.length, 1, 'Opening and closing probes metadata only; it does not add or trim media');

    await videoLauncher.click();
    await page.waitForFunction(() => document.querySelector('[aria-label="Source out"]')?.value === '12');
    const reopenedDialog = page.getByRole('dialog', {name: 'Source preview'});
    await reopenedDialog.getByRole('button', {name: 'Mark Out', exact: true}).click();
    assert.equal(await reopenedDialog.getByLabel('Source out').inputValue(), '0', 'Mark Out preserves a real playhead position of zero');
    assert.equal(await reopenedDialog.getByRole('button', {name: 'Insert at playhead'}).isDisabled(), true, 'A zero-length range cannot be inserted');

    await reopenedDialog.getByLabel('Source in').fill('3');
    await reopenedDialog.getByLabel('Source out').fill('3');
    assert.equal(await reopenedDialog.getByRole('button', {name: 'Insert at playhead'}).isDisabled(), true, 'Equal in/out values are rejected');
    await reopenedDialog.getByLabel('Source in').fill('2.25');
    await reopenedDialog.getByLabel('Source out').fill('5.75');
    assert.equal(await reopenedDialog.getByRole('button', {name: 'Insert at playhead'}).isDisabled(), false, 'A valid source range can be inserted');
    await reopenedDialog.getByRole('button', {name: 'Insert at playhead'}).click();
    await reopenedDialog.waitFor({state: 'detached'});
    const insertedVideo = await page.evaluate(() => {
      const state = window.editorStore.getState();
      const track = state.project.tracks.find(candidate => candidate.id === 'selected-video-track');
      return {
        item: track.items[0],
        selectedTrackId: state.selectedTrackId,
        selectedItemId: state.selectedItemId,
        dirty: state.dirty,
        historyLength: state.history.length,
      };
    });
    assert.equal(probes.length, 3, 'Insert re-probes the source before adding it');
    assert.deepEqual({start: insertedVideo.item.start, duration: insertedVideo.item.duration, source_in: insertedVideo.item.source_in},
      {start: 4.75, duration: 3.5, source_in: 2.25}, 'Insert passes the chosen source in/out range into the actual timeline clip');
    assert.equal(insertedVideo.selectedTrackId, 'selected-video-track', 'Insert uses the selected compatible track');
    assert.equal(insertedVideo.selectedItemId, insertedVideo.item.id, 'The new clip becomes selected');
    assert.equal(insertedVideo.dirty, true, 'Adding the source is a project edit');
    assert.ok(insertedVideo.historyLength > 0, 'Adding the source is undoable');

    await page.getByRole('button', {name: 'Open image source'}).click();
    const imageDialog = page.getByRole('dialog', {name: 'Source preview'});
    await imageDialog.waitFor();
    assert.equal(await imageDialog.getByLabel('Source in').isDisabled(), true, 'Still images use a duration control rather than a source in');
    assert.equal(await imageDialog.getByLabel('Source out').inputValue(), '5', 'Still images default to a five-second timeline duration');
    assert.equal(await imageDialog.getByRole('button', {name: 'Mark Out', exact: true}).count(), 0, 'Still images have no source-playhead marks');
    assert.equal(probes.length, 3, 'Opening a still image does not probe a nonexistent source duration');
    await imageDialog.getByLabel('Source out').fill('3.25');
    await imageDialog.getByRole('button', {name: 'Insert at playhead'}).click();
    await imageDialog.waitFor({state: 'detached'});
    const insertedImage = await page.evaluate(() => {
      const track = window.editorStore.getState().project.tracks.find(candidate => candidate.id === 'selected-video-track');
      return track.items.find(item => item.asset_id && window.editorStore.getState().project.assets[item.asset_id]?.type === 'image');
    });
    assert.equal(insertedImage.duration, 3.25, 'The editable still-image duration controls its timeline length');
    assert.equal(insertedImage.source_in, 0, 'Still images always start at source time zero');
    assert.equal(probes.length, 4, 'Adding the still uses the normal media metadata probe');

    const generationRequests = apiRequests.filter(request => /generation|director|maestro|export/i.test(request.path));
    assert.deepEqual(generationRequests, [], `Source preview and insert send no generation requests: ${generationRequests.map(request => request.path).join(', ')}`);
    assert.deepEqual(pageErrors, [], `No browser errors occurred: ${pageErrors.join('; ')}`);
    console.log('Source preview no-mutation, unknown-duration probe, exact range insertion, invalid range, Mark Out at zero, focus handling, image duration, and no-generation checks passed');
  } finally {
    await browser.close();
  }
})().catch(error => {
  console.error(error);
  process.exitCode = 1;
});
