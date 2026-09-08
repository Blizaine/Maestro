// Real media + React/Zustand; all application API calls are isolated from the user's server.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const {execFileSync} = require('node:child_process');
const root = path.resolve(__dirname, '../..');
const esbuild = require(path.join(root, 'ui/node_modules/esbuild'));
const playwright = require(process.env.MAESTRO_PLAYWRIGHT ||
  'C:/Users/bliza/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');

(async () => {
  const temporary = fs.mkdtempSync(path.join(os.tmpdir(), 'maestro-viggle-ui-'));
  let browser;
  try {
    const videoFile = path.join(temporary, 'source.mp4');
    execFileSync('ffmpeg', ['-v', 'error', '-y', '-f', 'lavfi', '-i', 'testsrc2=size=320x180:rate=24',
      '-t', '8', '-c:v', 'libx264', '-pix_fmt', 'yuv420p', videoFile], {windowsHide: true});
    const media = fs.readFileSync(videoFile);
    const picture = Buffer.from('iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+jK1sAAAAASUVORK5CYII=', 'base64');
    const bundle = await esbuild.build({stdin: {contents: `
      import React from 'react'; import {createRoot} from 'react-dom/client';
      import {useStore} from './src/stores/useStore';
      import {ViggleControls} from './src/components/Sidebar/ViggleControls';
      import {DurationSlider} from './src/components/Sidebar/DurationSlider';
      import {VideoTimelineSelector} from './src/components/shared/VideoTimelineSelector';
      window.store = useStore;
      function App() { const mode = useStore(s => s.generationMode);
        return <main className="mx-auto max-w-sm p-4 bg-bg-primary text-text-primary">
          {mode === 'video' ? <><ViggleControls/><div hidden><DurationSlider/></div></> : <p>Image editor</p>}
        </main>; }
      const root = createRoot(document.getElementById('root'));
      window.mount = () => root.render(<React.StrictMode><App/></React.StrictMode>);
      function Legacy() {const [start, setStart] = React.useState(0); const [end, setEnd] = React.useState(8);
        return <div className="mx-auto max-w-sm p-4"><VideoTimelineSelector videoUrl="/source.mp4" duration={8}
          startTime={start} endTime={end} onStartChange={setStart} onEndChange={setEnd}/></div>}
      window.mountLegacy = () => root.render(<Legacy/>);
    `, resolveDir: path.join(root, 'ui'), loader: 'tsx'}, bundle: true, write: false,
      jsx: 'automatic', define: {'process.env.NODE_ENV': '"development"'}, logLevel: 'silent'});
    const assets = path.join(root, 'ui/dist/assets');
    const css = fs.readFileSync(path.join(assets, fs.readdirSync(assets).find(name => name.endsWith('.css'))), 'utf8');
    const options = {model_type: 'viggle_animate', architecture: 'viggle_animate', fps: 24,
      frames_minimum: 124, frames_maximum: 124, frames_steps: 17, sliding_window: true,
      sliding_window_defaults: {window_min: 124, window_max: 124, window_default: 124, overlap_default: 18}};
    const models = [{...options, name: 'Viggle Animate', family: 'minimax_h3'},
      {model_type: 'flux2_klein_9b', architecture: 'flux2_klein_9b', name: 'Klein', family: 'flux2',
        supports_image_edit: true, supports_ref_images: true, image_only: true}];
    browser = await playwright.chromium.launch({headless: true,
      ...(process.env.MAESTRO_CHROME || process.platform === 'win32'
        ? {executablePath: process.env.MAESTRO_CHROME || 'C:/Program Files/Google/Chrome/Application/chrome.exe'} : {})});
    const page = await browser.newPage({viewport: {width: 390, height: 844}});
    const errors = [], jobs = [], extracts = [];
    page.on('pageerror', error => errors.push(error.message));
    await page.route('**/*', route => {
      const request = route.request(), url = new URL(request.url());
      const json = data => route.fulfill({contentType: 'application/json', body: JSON.stringify(data)});
      if (url.pathname.endsWith('.mp4')) {
        const range = request.headers().range?.match(/bytes=(\d+)-(\d*)/);
        if (!range) return route.fulfill({contentType: 'video/mp4', body: media});
        const start = Number(range[1]), end = range[2] ? Number(range[2]) : media.length - 1;
        return route.fulfill({status: 206, contentType: 'video/mp4', body: media.subarray(start, end + 1),
          headers: {'Content-Range': `bytes ${start}-${end}/${media.length}`, 'Accept-Ranges': 'bytes'}});
      }
      if (url.pathname.endsWith('.png')) return route.fulfill({contentType: 'image/png', body: picture});
      if (url.pathname === '/api/v1/upload') return json(request.postData()?.includes('new.mp4')
        ? {path: '/uploads/new.mp4', url: '/new.mp4'} : {path: '/uploads/edited.png', url: '/edited.png'});
      if (url.pathname === '/api/v1/generate') {jobs.push(request.postDataJSON()); return json({job_id: String(jobs.length), status: 'held'});}
      if (url.pathname === '/api/v1/extract-frames') {extracts.push(request.postDataJSON()); return json({start_path: '/uploads/frame.png', start_url: '/frame.png'});}
      if (url.pathname.startsWith('/api/v1/model-options/')) return json(url.pathname.endsWith('viggle_animate') ? options
        : {model_type: 'flux2_klein_9b', architecture: 'flux2_klein_9b', image_only: true, fps: 1,
          max_image_refs: 3, image_ref_choices: {choices: [['References', 'KI']]}});
      if (url.pathname.includes('/loras')) return json({loras: [], presets: []});
      if (url.pathname.startsWith('/api/')) return json({});
      return route.fulfill({contentType: 'text/html', body: '<meta name="viewport" content="width=device-width,initial-scale=1"><div id="root"></div>'});
    });
    await page.goto('http://maestro.test'); await page.addStyleTag({content: css});
    await page.addScriptTag({content: bundle.outputFiles[0].text});
    await page.evaluate(({options, models}) => {
      window.store.setState({generationMode: 'video', studioVideoWorkflow: 'animate', models, modelOptions: options,
        params: {...window.store.getState().params, model_type: 'viggle_animate', _studio_video_workflow: 'animate',
          video_guide: '/uploads/source.mp4', _duration_planning_mode: 'auto', image_mode: 0,
          _viggle_edited_frame: '/uploads/edited.png', video_length: 124, resolution: 'auto_480p'},
        durationSeconds: 124 / 24, slidingWindowSeconds: 124 / 24, slidingWindowOverlap: 18}); window.mount();
    }, {options, models});
    await page.waitForFunction(() => window.store.getState().params._viggle_source_seconds === 8);
    await page.waitForFunction(() => window.store.getState().durationSeconds === 8);
    const timeline = page.getByLabel('Video trim timeline', {exact: true});
    await timeline.locator('img').first().waitFor();
    assert.equal(await timeline.locator('img').count(), 10, 'Filmstrip decoded');
    const start = page.getByRole('slider', {name: 'Trim start', exact: true});
    const end = page.getByRole('slider', {name: 'Trim end', exact: true});
    const frame = page.getByRole('slider', {name: 'Frame to edit', exact: true});
    async function drag(handle, time) {
      await timeline.scrollIntoViewIfNeeded();
      const track = await timeline.boundingBox(), knob = await handle.boundingBox();
      await page.mouse.move(knob.x + knob.width / 2, knob.y + knob.height / 2);
      await page.mouse.down(); await page.mouse.move(track.x + track.width * time / 8, knob.y + knob.height / 2, {steps: 8}); await page.mouse.up();
    }
    await drag(start, 2); await drag(end, 6); await drag(frame, 4);
    const selection = () => page.evaluate(() => {const p = window.store.getState().params;
      return [p._viggle_trim_start, p._viggle_trim_end, p._viggle_frame_seconds, window.store.getState().durationSeconds];});
    assert.deepEqual(await selection(), [2, 6, 4, 4]);
    await page.waitForFunction(() => Math.abs(document.querySelector('video').currentTime - 4) < 0.01);
    // Playback/native seeking also moves the edit playhead, and cannot leave the trim.
    await page.getByLabel('Control video preview').evaluate(video => {video.currentTime = 5;});
    await page.waitForFunction(() => Math.abs(window.store.getState().params._viggle_frame_seconds - 5) < 0.01);
    await frame.press('Home'); assert.equal((await selection())[2], 2);
    await frame.press('ArrowRight'); assert.ok(Math.abs((await selection())[2] - 2 - 1/24) < 1e-6);
    // Touch pointer dragging: start/end handles do not steal the playhead.
    const rect = await timeline.boundingBox();
    await frame.dispatchEvent('pointerdown', {pointerType: 'touch', pointerId: 10, clientX: rect.x + rect.width / 2, clientY: rect.y - 5});
    await page.evaluate(({x,y}) => {
      window.dispatchEvent(new PointerEvent('pointermove', {pointerType: 'touch', pointerId: 10, clientX: x, clientY: y}));
      window.dispatchEvent(new PointerEvent('pointerup', {pointerType: 'touch', pointerId: 10}));
    }, {x: rect.x + rect.width * 0.5, y: rect.y - 5});
    assert.equal((await selection())[2], 4);
    await page.getByRole('spinbutton', {name: 'Trim start seconds'}).fill('4.5');
    assert.equal((await selection())[2], 4.5, 'Excluded frame moves into new trim');
    await page.getByRole('spinbutton', {name: 'Trim start seconds'}).fill('2');
    await page.getByRole('spinbutton', {name: 'Source frame seconds'}).fill('4');
    await page.getByRole('button', {name: 'Use a character', exact: true}).click();
    await page.getByLabel('Upload image', {exact: true}).setInputFiles({name: 'person.png', mimeType: 'image/png', buffer: picture});
    await page.getByRole('img', {name: 'Character reference', exact: true}).waitFor();
    await page.evaluate(() => window.store.setState({params: {...window.store.getState().params,
      _viggle_prepared: {signature: 'frame-4', image_url: '/edited.png', image_path: '/uploads/edited.png', width: 320, height: 180},
      _viggle_edited_frame: '/uploads/edited.png'}}));
    await page.getByRole('spinbutton', {name: 'Trim end seconds'}).fill('5');
    assert.equal(await page.getByRole('img', {name: 'Prepared character frame'}).count(), 1, 'Unchanged selected frame retains preview');
    await page.getByRole('spinbutton', {name: 'Trim end seconds'}).fill('3');
    assert.equal(await page.getByRole('img', {name: 'Prepared character frame'}).count(), 0, 'Clamped selected frame invalidates preview');
    await page.getByRole('spinbutton', {name: 'Trim end seconds'}).fill('6');
    await page.getByRole('spinbutton', {name: 'Source frame seconds'}).fill('4');
    await page.evaluate(() => window.store.getState().startGeneration('queue'));
    assert.equal(jobs.at(-1).viggle_character.frame_seconds, 4, 'Klein receives original source time');
    assert.equal(jobs.at(-1)._viggle_prepared, undefined);
    await page.getByRole('button', {name: 'Use an edited frame', exact: true}).click();
    await page.getByLabel('Upload edited frame', {exact: true}).setInputFiles({name: 'edit.png', mimeType: 'image/png', buffer: picture});
    await page.getByRole('img', {name: 'Edited reference frame'}).waitFor();
    await page.evaluate(() => window.store.getState().startGeneration('queue'));
    assert.equal(jobs.at(-1).video_length, 96); assert.equal(jobs.at(-1)._viggle_trim_start, 2);
    assert.equal(jobs.at(-1)._viggle_trim_end, 6); assert.equal(jobs.at(-1)._viggle_frame_seconds, 4);
    assert.equal(jobs.at(-1).video_guide, '/uploads/source.mp4');
    const saved = jobs.at(-1);
    await page.getByRole('button', {name: 'Edit current video frame in Maestro', exact: true}).click();
    await page.getByText('Image editor', {exact: true}).waitFor();
    assert.deepEqual(extracts.at(-1), {video_path: '/uploads/source.mp4', start_time: 4}, 'Original timestamp, not trim-relative');
    await page.evaluate(() => window.store.getState().skipAnchorPhase());
    await timeline.waitFor(); assert.deepEqual(await selection(), [2, 6, 4, 4], 'Return preserves selection');
    await page.getByRole('button', {name: 'Edit current video frame in Maestro', exact: true}).click();
    await page.getByText('Image editor', {exact: true}).waitFor();
    await page.evaluate(async () => {window.store.setState({outputs: [{name: 'new-edit.png', type: 'image', url: '/new-edit.png'}]}); await window.store.getState().applyOutputAsAnchor();});
    await timeline.waitFor(); assert.deepEqual(await selection(), [2, 6, 4, 4], 'Apply edited image preserves selection');
    await page.evaluate(async saved => {
      window.store.setState({params: {...window.store.getState().params, _viggle_trim_start: 0, _viggle_trim_end: 8, _viggle_frame_seconds: 0},
        selectedOutputMeta: {params: saved, generation_mode: 'video'}});
      await window.store.getState().loadSettingsFromOutput();
    }, saved);
    assert.deepEqual(await selection(), [2, 6, 4, 4], 'Load Settings restores trim and playhead');
    for (const [width, height, theme] of [[390, 844, 'golden-hour'], [1440, 900, 'daylight']]) {
      await page.setViewportSize({width, height}); await page.evaluate(theme => document.documentElement.dataset.theme = theme, theme);
      await page.evaluate(() => window.scrollTo(0, 0));
      await timeline.locator('img').first().waitFor();
      await page.waitForFunction(() => {const video = document.querySelector('video'); return video.readyState >= 2 && !video.seeking;});
      assert.ok(await page.locator('#root').evaluate(el => el.scrollWidth <= el.clientWidth), 'No horizontal overflow');
      if (process.env.MAESTRO_UI_SCREENSHOT_DIR) {
        fs.mkdirSync(process.env.MAESTRO_UI_SCREENSHOT_DIR, {recursive: true});
        await page.screenshot({path: path.join(process.env.MAESTRO_UI_SCREENSHOT_DIR, `viggle-timeline-${width}.png`), fullPage: true});
      }
    }
    await page.getByLabel('Remove control video', {exact: true}).click();
    const removed = await page.evaluate(() => window.store.getState().params);
    assert.equal(removed._viggle_trim_start, undefined); assert.equal(removed._viggle_trim_end, undefined);
    await page.getByLabel('Upload control video', {exact: true}).setInputFiles({name: 'new.mp4', mimeType: 'video/mp4', buffer: media});
    await page.waitForFunction(() => window.store.getState().params._viggle_source_seconds === 8);
    assert.deepEqual(await selection(), [0, 8, 0, 8], 'Replacement starts with its full range');
    // Existing Retake/Repaint/Recast behavior: the handle itself scrubs video, no frame playhead required.
    await page.evaluate(() => window.mountLegacy());
    await drag(page.getByRole('slider', {name: 'Trim start', exact: true}), 1);
    assert.equal(await page.getByRole('slider', {name: 'Frame to edit'}).count(), 0);
    await page.waitForFunction(() => Math.abs(document.querySelector('video').currentTime - 1) < 0.01);
    assert.deepEqual(errors, [], 'No React errors or resize/update loops');
    console.log('Animate timeline: real preview/filmstrip, mouse/touch/keyboard scrub, trim clamp, Auto timing, queue payload, Image round trip, Load Settings, source reset, themes/mobile and legacy preview passed');
  } finally {
    if (browser) await browser.close();
    fs.rmSync(temporary, {recursive: true, force: true}); // Owned mkdtemp directory only.
  }
})().catch(error => {console.error(error); process.exitCode = 1;});
