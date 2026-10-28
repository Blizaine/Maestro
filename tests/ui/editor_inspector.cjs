// Run after the UI dependencies are installed: node tests/ui/editor_inspector.cjs
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
    id: 'editor-inspector-test',
    name: 'Inspector test',
    workspace: 'inspector-test-workspace',
    created_at: 1,
    updated_at: 1,
    canvas: {width: 1920, height: 1080, fps: 24, background: '#000000'},
    assets: {
      original: {
        id: 'original', name: 'original.mp4', type: 'video', origin: 'output',
        workspace: 'inspector-test-workspace', path: '/outputs/original.mp4', url: '/media/original.mp4',
        duration: 12, width: 1280, height: 720, fps: 24, has_audio: true, size: 100,
      },
      alternate: {
        id: 'alternate', name: 'alternate.mp4', type: 'video', origin: 'output',
        workspace: 'inspector-test-workspace', path: '/outputs/alternate.mp4', url: '/media/alternate.mp4',
        duration: 12, width: 1280, height: 720, fps: 24, has_audio: true, size: 110,
      },
      still: {
        id: 'still', name: 'still.png', type: 'image', origin: 'upload',
        workspace: 'inspector-test-workspace', path: '/uploads/still.png', url: '/media/still.png',
        duration: 0, width: 1000, height: 1000, fps: 0, has_audio: false, size: 120,
      },
    },
    tracks: [
      {
        id: 'video-track', name: 'Main video', type: 'video', z_index: 0,
        muted: false, locked: false, volume: 1,
        items: [{
          id: 'video-item', asset_id: 'original', name: 'Shot 1', start: 0,
          duration: 4, source_in: 1.5, speed: 1.25, volume: 1, opacity: 1,
          fit: 'cover', transform: {x: 42, y: -12, scale: 1.4, rotation: 8},
          muted: false, disabled: false, take_asset_ids: ['original', 'alternate', 'still'],
          take_states: {
            original: {source_in: 1.5, speed: 1.25},
            alternate: {source_in: 0.25, speed: 0.8},
            still: {source_in: 2, speed: 1},
          },
          director: {
            pipeline_id: 'inspector-pipeline', clip_index: 0, pipeline_type: 'music_video',
            workspace: 'inspector-test-workspace', video_prompt: 'Rendered prompt',
            window_prompts: ['Rendered prompt'],
          },
        }],
      },
      {id: 'audio-track', name: 'Music', type: 'audio', z_index: 0, muted: false, locked: false, volume: 1, items: []},
      {
        id: 'text-track', name: 'Titles', type: 'text', z_index: 10, muted: false, locked: false, volume: 1,
        items: [{
          id: 'text-item', name: 'Title', start: 0, duration: 2, source_in: 0, speed: 1,
          volume: 1, opacity: 1, fit: 'contain', transform: {x: 0, y: 0, scale: 1, rotation: 0},
          text: 'Title text', style: {x: 0, y: 0, font_size: 64, color: '#ffffff', text_align: 'center'},
        }],
      },
    ],
    markers: [],
    export: {
      quality: 'high', codec: 'h264', encoder: 'auto', include_audio: true,
      resolution: 'canvas', frame_rate: 'project', filename: '',
      spatial_upsampling: '', film_grain_intensity: 0, film_grain_saturation: 0.5,
    },
    exports: [],
  };
}

(async () => {
  const bundle = await esbuild.build({
    absWorkingDir: path.dirname(uiNodeModules),
    stdin: {
      contents: `
        import React from 'react';
        import {createRoot} from 'react-dom/client';
        import {EditorInspector} from './src/editor/EditorInspector';
        import {useEditorStore} from './src/editor/useEditorStore';
        window.editorStore = useEditorStore;
        function App() { return <div style={{width: 360, height: 900}}><EditorInspector compact /></div>; }
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
  const page = await browser.newPage({viewport: {width: 1080, height: 900}});
  const apiRequests = [];
  const playCalls = [];

  try {
    await page.addInitScript(() => {
      const mediaState = new WeakMap();
      window.__mediaPlayCalls = [];
      const stateFor = media => {
        if (!mediaState.has(media)) mediaState.set(media, {currentTime: 0, playbackRate: 1, paused: true});
        return mediaState.get(media);
      };
      Object.defineProperties(HTMLMediaElement.prototype, {
        duration: {configurable: true, get: () => 12},
        currentTime: {
          configurable: true,
          get() { return stateFor(this).currentTime; },
          set(value) { stateFor(this).currentTime = Number(value); },
        },
        playbackRate: {
          configurable: true,
          get() { return stateFor(this).playbackRate; },
          set(value) { stateFor(this).playbackRate = Number(value); },
        },
        paused: {configurable: true, get() { return stateFor(this).paused; }},
      });
      HTMLMediaElement.prototype.play = function () {
        stateFor(this).paused = false;
        window.__mediaPlayCalls.push({tag: this.tagName, muted: this.muted});
        return Promise.resolve();
      };
      HTMLMediaElement.prototype.pause = function () { stateFor(this).paused = true; };
    });

    await page.route('**/*', async route => {
      const request = route.request();
      const url = new URL(request.url());
      const json = (body, status = 200) => route.fulfill({
        status,
        contentType: 'application/json',
        body: JSON.stringify(body),
      });

      if (url.pathname === '/' || url.pathname === '/index.html') {
        return route.fulfill({contentType: 'text/html', body: '<div id="root"></div>'});
      }
      if (url.pathname.startsWith('/api/')) {
        apiRequests.push(`${request.method()} ${url.pathname}`);
        if (url.pathname === '/api/v1/editor/media/preview') {
          return json({thumbnail_url: '/preview-contact-sheet.png'});
        }
        return json({});
      }
      if (url.pathname.endsWith('.mp4')) return route.fulfill({status: 200, contentType: 'video/mp4', body: ''});
      if (url.pathname.endsWith('.png')) return route.fulfill({status: 204, body: ''});
      return route.fulfill({status: 204, body: ''});
    });

    await page.goto('http://editor-inspector.test');
    await page.addScriptTag({content: bundle.outputFiles[0].text});
    await page.evaluate(project => {
      window.editorStore.setState({
        workspace: project.workspace,
        project,
        selectedItemId: 'video-item',
        selectedItemIds: ['video-item'],
        selectedTrackId: 'video-track',
        dirty: false,
        error: null,
        roundTrip: null,
      });
      window.mount();
    }, makeProject());

    assert.equal(await page.getByRole('tab', {name: 'Basic'}).getAttribute('aria-selected'), 'true',
      'Basic is the initial inspector section');
    assert.equal(await page.getByRole('tab', {name: 'Audio'}).count(), 1, 'Audio appears for a video with audio');
    assert.equal(await page.getByRole('tab', {name: 'AI'}).count(), 1, 'AI appears for a Director video clip');
    assert.equal(await page.getByLabel('Start').inputValue(), '0', 'Timing controls are immediately available');
    assert.equal(await page.getByLabel('Scale').count(), 1, 'Transform controls are immediately available');
    assert.equal(await page.getByRole('button', {name: 'Reset transform'}).count(), 1, 'Transform reset is labeled');
    assert.equal(await page.getByRole('radio', {name: 'Take 1: original.mp4'}).isChecked(), true,
      'The active source is exposed as a checked radio choice');

    await page.getByRole('button', {name: 'Reset transform'}).click();
    assert.deepEqual(await page.evaluate(() => window.editorStore.getState().project.tracks[0].items[0].transform),
      {x: 0, y: 0, scale: 1, rotation: 0}, 'Reset transform restores neutral framing');

    const missingProject = makeProject();
    missingProject.assets.original.missing = true;
    await page.evaluate(project => window.editorStore.setState({project}), missingProject);
    assert.equal(await page.getByRole('button', {name: 'Relink source file'}).count(), 1, 'Offline active media keeps its relink action');
    assert.equal(await page.getByLabel('Choose replacement source file').getAttribute('accept'), 'video/*',
      'Relink accepts the current source media type');
    const onlineProject = makeProject();
    await page.evaluate(project => window.editorStore.setState({project}), onlineProject);

    await page.getByRole('button', {name: 'Compare current take with Take 2'}).click();
    const dialog = page.getByRole('dialog', {name: 'Compare takes'});
    await dialog.waitFor();
    assert.equal(await dialog.locator('video').count(), 2, 'Video takes appear side by side');
    assert.equal(await dialog.locator('img').count(), 0);
    assert.equal(await page.evaluate(() => window.__mediaPlayCalls.length), 0, 'Opening comparison never starts playback');
    assert.deepEqual(await dialog.locator('video').evaluateAll(videos => videos.map(video => ({muted: video.muted, autoplay: video.autoplay}))),
      [{muted: false, autoplay: false}, {muted: true, autoplay: false}], 'Only the current take is the audio master');
    assert.equal(await dialog.getByText('In 1.50 s · 1.25×').count(), 1, 'Current take uses its saved source in and speed');
    assert.equal(await dialog.getByText('In 0.25 s · 0.80×').count(), 1, 'Alternative take uses its saved source in and speed');

    await dialog.getByRole('button', {name: 'Play comparison'}).click();
    await page.waitForFunction(() => window.__mediaPlayCalls.length === 2);
    assert.deepEqual(await page.evaluate(() => window.__mediaPlayCalls.map(call => call.muted)), [false, true],
      'Play starts both videos while only one audio track is audible');
    await dialog.getByRole('slider', {name: 'Comparison position'}).evaluate(node => {
      const setValue = Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value').set;
      setValue.call(node, '1.5');
      node.dispatchEvent(new Event('input', {bubbles: true}));
      node.dispatchEvent(new Event('change', {bubbles: true}));
    });
    await page.waitForFunction(() => {
      const times = [...document.querySelectorAll('[role="dialog"] video')].map(video => video.currentTime);
      return Math.abs(times[0] - 3.375) < 0.001 && Math.abs(times[1] - 1.45) < 0.001;
    });
    await dialog.getByRole('button', {name: 'Pause comparison'}).click();
    assert.deepEqual(await dialog.locator('video').evaluateAll(videos => videos.map(video => video.paused)), [true, true],
      'Pause affects both videos together');

    await dialog.getByRole('button', {name: 'Use Take 2 on timeline'}).click();
    await dialog.waitFor({state: 'detached'});
    const switched = await page.evaluate(() => {
      const item = window.editorStore.getState().project.tracks[0].items[0];
      return {assetId: item.asset_id, sourceIn: item.source_in, speed: item.speed, original: item.take_states.original, takeIds: item.take_asset_ids};
    });
    assert.equal(switched.assetId, 'alternate', 'Selecting a take uses the store action to make it active');
    assert.equal(switched.sourceIn, 0.25);
    assert.equal(switched.speed, 0.8);
    assert.deepEqual(switched.original, {source_in: 1.5, speed: 1.25}, 'The original source state is preserved');
    assert.deepEqual(switched.takeIds, ['original', 'alternate', 'still'], 'The original and alternate assets remain available');
    assert.equal(await page.getByRole('radio', {name: 'Take 2: alternate.mp4'}).isChecked(), true,
      'The new active take is announced as the current selection');

    await page.getByRole('button', {name: 'Compare current take with Take 3'}).click();
    const imageDialog = page.getByRole('dialog', {name: 'Compare takes'});
    await imageDialog.waitFor();
    assert.equal(await imageDialog.locator('video').count(), 1, 'A video can be compared with an image');
    assert.equal(await imageDialog.locator('img').count(), 1, 'Image takes render as stills');
    assert.equal(await imageDialog.locator('video').first().evaluate(video => video.muted), false,
      'The only source with audio remains the master');
    assert.equal(await page.evaluate(() => window.__mediaPlayCalls.length), 2, 'Opening an image comparison does not autoplay the video');
    await imageDialog.getByRole('button', {name: 'Close take comparison'}).click();

    await page.getByRole('tab', {name: 'AI'}).click();
    assert.equal(await page.getByLabel('Video prompt draft').inputValue(), 'Rendered prompt',
      'Director prompt editing remains reachable in the AI section');
    assert.equal(await page.getByRole('button', {name: 'Re-run Director shot'}).count(), 1,
      'Director rerun remains available');
    await page.getByRole('button', {name: 'Edit clip with Maestro AI'}).click();
    assert.equal(await page.getByRole('button', {name: /Retake/}).count(), 1, 'Existing Maestro AI actions remain available');

    await page.evaluate(() => window.editorStore.setState({selectedItemId: 'text-item', selectedItemIds: ['text-item'], selectedTrackId: 'text-track'}));
    assert.equal(await page.getByRole('tab', {name: 'Audio'}).count(), 0, 'Text clips do not show an inapplicable Audio section');
    assert.equal(await page.getByRole('tab', {name: 'AI'}).count(), 0, 'Text clips do not show an inapplicable AI section');
    assert.equal(await page.getByLabel('Text', {exact: true}).inputValue(), 'Title text', 'Text clip controls remain available');

    await page.evaluate(() => window.editorStore.setState({selectedItemId: null, selectedItemIds: [], selectedTrackId: 'video-track'}));
    assert.equal(await page.getByLabel('Track name').inputValue(), 'Main video', 'Track settings remain available');
    await page.evaluate(() => window.editorStore.setState({selectedTrackId: null}));
    assert.equal(await page.getByLabel('Width').inputValue(), '1920', 'Project canvas settings remain available');
    assert.equal(await page.getByLabel('Include audio').isChecked(), true, 'Project export settings remain available');

    const generationRequests = apiRequests.filter(request => /generation|maestro|director/i.test(request));
    assert.deepEqual(generationRequests, [], `No generation or Director endpoint was called: ${generationRequests.join(', ')}`);
    console.log('Contextual inspector, take selection, aligned A/B playback, Director controls, relinking and track/project settings passed');
    console.log('No generation or Director request was sent during comparison');
  } finally {
    await browser.close();
  }
})().catch(error => {
  console.error(error);
  process.exitCode = 1;
});
