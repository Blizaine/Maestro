// Isolated Editor Audio tools guard and undo regression harness.
// Every HTTP route is mocked; no audio analysis, transcription, downloads, or jobs run.
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
    id: 'audio-tools-project',
    name: 'Audio tools fixture',
    workspace: 'Audio tools workspace',
    revision: 7,
    created_at: 1,
    updated_at: 1,
    canvas: {width: 1920, height: 1080, fps: 24, background: '#000000'},
    assets: {
      music: {
        id: 'music', name: 'music.wav', type: 'audio', origin: 'output', workspace: 'Audio source',
        path: '/outputs/music.wav', url: '/media/music.wav', duration: 12, width: 0, height: 0, fps: 0, has_audio: true, size: 100,
      },
      alternateMusic: {
        id: 'alternate-music', name: 'alternate.wav', type: 'audio', origin: 'output', workspace: 'Audio source',
        path: '/outputs/alternate.wav', url: '/media/alternate.wav', duration: 12, width: 0, height: 0, fps: 0, has_audio: true, size: 110,
      },
      narration: {
        id: 'narration', name: 'narration.wav', type: 'audio', origin: 'output', workspace: 'Audio source',
        path: '/outputs/narration.wav', url: '/media/narration.wav', duration: 12, width: 0, height: 0, fps: 0, has_audio: true, size: 120,
      },
    },
    tracks: [
      {
        id: 'music-track', name: 'Music', type: 'audio', z_index: 0, muted: false, locked: false, volume: 1,
        items: [{
          id: 'music-item', asset_id: 'music', name: 'Music bed', start: 0, duration: 6, source_in: 0,
          speed: 1, volume: 0.75, opacity: 1, fit: 'contain',
          transform: {x: 0, y: 0, scale: 1, rotation: 0},
          keyframes: {volume: [
            {time: 0, value: 0.5, easing: 'linear'},
            {time: 6, value: 0.7, easing: 'linear'},
          ]},
        }],
      },
      {
        id: 'narration-track', name: 'Narration', type: 'audio', z_index: 1, muted: false, locked: false, volume: 1,
        items: [{
          id: 'narration-item', asset_id: 'narration', name: 'Narration', start: 1, duration: 3, source_in: 0,
          speed: 1, volume: 1, opacity: 1, fit: 'contain', transform: {x: 0, y: 0, scale: 1, rotation: 0},
        }],
      },
    ],
    markers: [],
    export: {quality: 'high', codec: 'h264', encoder: 'auto', include_audio: true, resolution: 'canvas', frame_rate: 'project', filename: ''},
    exports: [],
  };
}

function makeDeferred() {
  let resolve;
  const promise = new Promise(done => { resolve = done; });
  return {promise, resolve};
}

(async () => {
  const bundle = await esbuild.build({
    absWorkingDir: path.dirname(uiNodeModules),
    stdin: {
      contents: `
        import React, {useState} from 'react';
        import {createRoot} from 'react-dom/client';
        import {EditorAudioTools} from './src/editor/EditorAudioTools';
        import {useEditorStore} from './src/editor/useEditorStore';
        window.audioStore = useEditorStore;
        function App() {
          const [open, setOpen] = useState(false);
          return <><button type="button" onClick={() => setOpen(true)}>Open Audio tools</button>
            <EditorAudioTools open={open} onClose={() => setOpen(false)} /></>;
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
  const page = await browser.newPage({viewport: {width: 1000, height: 850}});
  const requests = [];
  const analyzeBodies = [];
  const transcribeBodies = [];
  const downloads = [];
  let serverRevision = 7;
  let serverProject = makeProject();
  let failSaves = false;
  let transcriptionAvailable = false;
  let analyzeGate = null;
  let analyzeStartedResolve = null;
  let transcribeGate = null;
  let transcribeStartedResolve = null;

  try {
    page.on('download', download => downloads.push(download.suggestedFilename()));
    await page.route('**/*', async route => {
      const request = route.request();
      const url = new URL(request.url());
      const method = request.method();
      const json = (body, status = 200) => route.fulfill({status, contentType: 'application/json', body: JSON.stringify(body)});

      if (url.pathname === '/' || url.pathname === '/index.html') {
        return route.fulfill({contentType: 'text/html', body: '<meta name="viewport" content="width=device-width,initial-scale=1"><div id="root"></div>'});
      }
      if (url.pathname.startsWith('/api/')) {
        requests.push({method, path: url.pathname});
        if (url.pathname === '/api/v1/editor/audio/capabilities' && method === 'GET') {
          return json({transcription_available: transcriptionAvailable, reason: transcriptionAvailable ? '' : 'Local transcription support is not installed.'});
        }
        if (url.pathname === '/api/v1/editor/projects/audio-tools-project' && method === 'PUT') {
          if (failSaves) return json({detail: 'Mock project save rejected'}, 500);
          const body = request.postDataJSON();
          serverRevision += 1;
          serverProject = {...body.project, revision: serverRevision};
          return json(serverProject);
        }
        if (url.pathname === '/api/v1/editor/projects' && method === 'GET') return json({projects: []});
        if (url.pathname === '/api/v1/editor/audio/analyze' && method === 'POST') {
          analyzeBodies.push(request.postDataJSON());
          if (analyzeGate) {
            const gate = analyzeGate;
            analyzeGate = null;
            analyzeStartedResolve?.();
            await gate.promise;
          }
          return json({integrated_lufs: -18.5, peak_db: -1.2, gain_db: 3.5, gain: 1.5});
        }
        if (url.pathname === '/api/v1/editor/audio/transcribe' && method === 'POST') {
          transcribeBodies.push(request.postDataJSON());
          if (transcribeGate) {
            const gate = transcribeGate;
            transcribeGate = null;
            transcribeStartedResolve?.();
            await gate.promise;
          }
          return json({
            offset: 0.25,
            cues: [{
              start: 0.5, end: 1.5, text: 'Reviewed phrase',
              words: [{start: 0.5, end: 0.9, text: 'Reviewed'}, {start: 0.9, end: 1.5, text: 'phrase'}],
            }],
          });
        }
        throw new Error(`Unexpected Audio tools API request: ${method} ${url.pathname}`);
      }
      return route.fulfill({status: 204, body: ''});
    });

    await page.goto('http://editor-audio-tools.test');
    await page.addScriptTag({content: bundle.outputFiles[0].text});
    await page.evaluate(project => {
      window.audioStore.setState({
        workspace: project.workspace,
        project,
        selectedItemId: 'music-item',
        selectedItemIds: ['music-item'],
        selectedTrackId: 'music-track',
        playhead: 0,
        dirty: false,
        saving: false,
        history: [],
        future: [],
        error: null,
      });
      window.mount();
    }, makeProject());

    assert.deepEqual(requests, [], 'Mounting closed Audio tools makes no request');
    await page.getByRole('button', {name: 'Open Audio tools'}).click();
    const dialog = page.getByRole('dialog', {name: 'Audio tools'});
    await dialog.waitFor();
    await dialog.getByText('Local transcription support is not installed.').waitFor();
    assert.equal(requests.filter(request => request.path.endsWith('/capabilities')).length, 1, 'Opening checks installed transcription capability once');
    assert.equal(requests.some(request => request.path.endsWith('/analyze') || request.path.endsWith('/transcribe')), false,
      'Opening Audio tools does not analyze or transcribe');
    assert.equal(await dialog.getByRole('button', {name: 'Transcribe selected clip'}).count(), 0,
      'Unavailable transcription has a reason and no active transcription action');
    await page.evaluate(() => window.audioStore.getState().setPlayhead(1));
    assert.equal(requests.filter(request => request.path.endsWith('/capabilities')).length, 1,
      'Ordinary store rerenders reuse the open dialog capability result');

    await dialog.getByRole('button', {name: 'Close audio tools'}).click();
    transcriptionAvailable = true;
    await page.getByRole('button', {name: 'Open Audio tools'}).click();
    await dialog.getByRole('button', {name: 'Transcribe selected clip'}).waitFor();
    assert.equal(requests.filter(request => request.path.endsWith('/capabilities')).length, 2,
      'Reopening refreshes the cached capability result');

    failSaves = true;
    await dialog.getByRole('button', {name: 'Analyze selected source range'}).click();
    await dialog.getByRole('alert').filter({hasText: 'Mock project save rejected'}).waitFor();
    assert.equal(analyzeBodies.length, 0, 'A failed project save blocks the analysis POST');
    await dialog.getByRole('button', {name: 'Transcribe selected clip'}).click();
    await dialog.getByRole('alert').filter({hasText: 'Mock project save rejected'}).waitFor();
    assert.equal(transcribeBodies.length, 0, 'A failed project save blocks the transcription POST');

    failSaves = false;
    let analyzeStartedResolveLocal;
    const analyzeStarted = new Promise(resolve => { analyzeStartedResolveLocal = resolve; });
    analyzeStartedResolve = analyzeStartedResolveLocal;
    const pendingAnalyze = makeDeferred();
    analyzeGate = pendingAnalyze;
    await dialog.getByRole('button', {name: 'Analyze selected source range'}).click();
    await analyzeStarted;
    assert.deepEqual(analyzeBodies[0], {
      workspace: 'Audio tools workspace',
      project_id: 'audio-tools-project',
      revision: 8,
      item_id: 'music-item',
    }, 'Analysis is tied to the saved project revision and selected clip');
    await page.evaluate(() => window.audioStore.getState().updateItem('music-item', {source_in: 0.25}));
    pendingAnalyze.resolve();
    await dialog.getByRole('button', {name: 'Analyze selected source range'}).isEnabled();
    await page.waitForTimeout(50);
    assert.equal(await dialog.getByRole('button', {name: 'Apply clip gain and envelope gain'}).count(), 0,
      'A response for the old source range cannot be applied after retiming');

    const analyzeButton = dialog.getByRole('button', {name: 'Analyze selected source range'});
    await analyzeButton.click();
    await dialog.getByRole('button', {name: 'Apply clip gain and envelope gain'}).waitFor();
    assert.equal(analyzeBodies.length, 2, 'A fresh analysis can be requested for the new range');
    assert.equal(analyzeBodies[1].item_id, 'music-item');
    assert.equal(analyzeBodies[1].revision, 9, 'A fresh analysis carries the revision saved after the range edit');
    const beforeNormalize = await page.evaluate(() => ({
      history: window.audioStore.getState().history.length,
      keyframes: window.audioStore.getState().project.tracks[0].items[0].keyframes.volume,
    }));
    await dialog.getByRole('button', {name: 'Apply clip gain and envelope gain'}).click();
    const normalized = await page.evaluate(() => {
      const state = window.audioStore.getState();
      const item = state.project.tracks[0].items[0];
      return {volume: item.volume, frames: item.keyframes.volume, history: state.history.length};
    });
    assert.equal(normalized.volume, 1.5, 'Analysis recommendation is applied as clip gain');
    assert.deepEqual(normalized.frames.map(frame => frame.value), [1, 1.4], 'Envelope points scale with the recommended gain');
    assert.equal(normalized.history, beforeNormalize.history + 1, 'Normalization is one undo step');
    await page.evaluate(() => window.audioStore.getState().undo());
    assert.equal(await page.evaluate(() => window.audioStore.getState().project.tracks[0].items[0].volume), 0.75,
      'One undo restores the clip gain');
    assert.deepEqual(await page.evaluate(() => window.audioStore.getState().project.tracks[0].items[0].keyframes.volume), beforeNormalize.keyframes,
      'One undo restores the prior volume envelope');

    const beforeDuck = await page.evaluate(() => ({
      history: window.audioStore.getState().history.length,
      keyframes: window.audioStore.getState().project.tracks[0].items[0].keyframes.volume,
      selectedItemId: window.audioStore.getState().selectedItemId,
      selectedTrackId: window.audioStore.getState().selectedTrackId,
    }));
    console.log('duck diagnostic', beforeDuck, await dialog.getByRole('combobox', {name: 'Narration track'}).inputValue(), await dialog.getByRole('button', {name: 'Apply duck envelope'}).isDisabled());
    await dialog.getByRole('button', {name: 'Apply duck envelope'}).click();
    const ducked = await page.evaluate(() => {
      const state = window.audioStore.getState();
      const item = state.project.tracks[0].items[0];
      return {frames: item.keyframes.volume, history: state.history.length};
    });
    assert.ok(Math.min(...ducked.frames.map(frame => frame.value)) < Math.min(...beforeDuck.keyframes.map(frame => frame.value)),
      'Ducking creates a lower music envelope over the narration overlap');
    assert.equal(ducked.history, beforeDuck.history + 1, 'Ducking is one undo step');
    await page.evaluate(() => window.audioStore.getState().undo());
    assert.deepEqual(await page.evaluate(() => window.audioStore.getState().project.tracks[0].items[0].keyframes.volume), beforeDuck.keyframes,
      'One undo restores the original music envelope');

    let transcribeStartedResolveLocal;
    const transcribeStarted = new Promise(resolve => { transcribeStartedResolveLocal = resolve; });
    transcribeStartedResolve = transcribeStartedResolveLocal;
    const pendingTranscription = makeDeferred();
    transcribeGate = pendingTranscription;
    await dialog.getByRole('button', {name: 'Transcribe selected clip'}).click();
    await transcribeStarted;
    assert.deepEqual(transcribeBodies[0], {
      workspace: 'Audio tools workspace',
      project_id: 'audio-tools-project',
      revision: 11,
      item_id: 'music-item',
    }, 'Transcription is tied to the saved project revision and selected clip');
    await page.evaluate(() => window.audioStore.getState().updateItem('music-item', {asset_id: 'alternateMusic'}));
    pendingTranscription.resolve();
    await dialog.getByRole('button', {name: 'Transcribe selected clip'}).isEnabled();
    await page.waitForTimeout(50);
    assert.equal(await dialog.getByRole('button', {name: 'Add reviewed transcript as captions'}).count(), 0,
      'A transcript returned for the prior source cannot be applied after a source change');

    await dialog.getByRole('button', {name: 'Transcribe selected clip'}).click();
    await dialog.getByRole('button', {name: 'Add reviewed transcript as captions'}).waitFor();
    assert.equal(transcribeBodies.length, 2, 'The changed source can be transcribed independently');
    const beforeCaptions = await page.evaluate(() => window.audioStore.getState().history.length);
    await dialog.getByRole('button', {name: 'Add reviewed transcript as captions'}).click();
    await dialog.getByText(/Added 1 transcript cues to the Captions track in one undo step/).waitFor();
    const imported = await page.evaluate(() => {
      const state = window.audioStore.getState();
      const track = state.project.tracks.find(candidate => candidate.name === 'Captions');
      return {history: state.history.length, item: track?.items[0]};
    });
    assert.equal(imported.history, beforeCaptions + 1, 'Caption import is one undo step');
    assert.deepEqual({start: imported.item.start, duration: imported.item.duration, text: imported.item.text, caption: imported.item.caption},
      {start: 0.75, duration: 1, text: 'Reviewed phrase', caption: true}, 'Reviewed transcript cues use the returned offset and caption styling metadata');
    await page.evaluate(() => window.audioStore.getState().undo());
    assert.equal(await page.evaluate(() => window.audioStore.getState().project.tracks.some(track => track.name === 'Captions')), false,
      'One undo removes the imported caption track');

    const forbidden = requests.filter(request => /generation|director|export|download/i.test(request.path));
    assert.deepEqual(forbidden, [], `No generation/export/download API requests were sent: ${forbidden.map(request => request.path).join(', ')}`);
    assert.deepEqual(downloads, [], 'Audio tools do not trigger browser downloads');
    console.log('Audio tools capability gating, save-before-analysis/transcription, stale-result protection, normalization, ducking, caption import, and undo checks passed');
    console.log('Every API request was mocked; no analysis, transcription, generation, export, or download was performed');
  } finally {
    await browser.close();
  }
})().catch(error => {
  console.error(error);
  process.exitCode = 1;
});
