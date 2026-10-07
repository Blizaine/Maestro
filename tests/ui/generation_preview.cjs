// Exercises the generation-preview selector and job reconnect flow in a real
// browser while every backend request is served by a small isolated fixture.
// Run: node tests/ui/generation_preview.cjs
const assert = require('node:assert/strict');
const path = require('node:path');
const root = path.resolve(__dirname, '../..');
const esbuild = require(path.join(root, 'ui/node_modules/esbuild'));
const playwright = require(process.env.MAESTRO_PLAYWRIGHT ||
  'playwright');

const jsonResponse = (route, body) => route.fulfill({json: body});
const preview = (id, kind, revision = 1) => ({
  url: `/api/v1/previews/${id}?revision=${revision}`,
  kind,
  revision,
  mode: kind === 'video' ? 'tiny_vae_video' : 'rgb',
  window: 2,
  total_windows: 5,
  clip: 1,
  total_clips: 2,
});
const status = (id, extra = {}) => ({
  job_id: id,
  status: 'running',
  progress: 45,
  step: 4,
  total_steps: 10,
  phase: 'Rendering clip 1',
  message: 'Generating...',
  output_files: [],
  error: null,
  show_in_gallery: true,
  current_clip: 1,
  total_clips: 2,
  current_window: 2,
  total_windows: 5,
  window_eta_seconds: 25,
  generation_eta_seconds: 50,
  ...extra,
});
const model = (model_type, name, preview_support) => ({
  model_type,
  name,
  family: 'fixture',
  architecture: 'fixture',
  is_i2v: false,
  is_t2v: true,
  guidance_max_phases: 1,
  fps: 24,
  ...(preview_support ? {preview_support} : {}),
});
const previewModels = [
  model('minimax_h3_fused_turbo', 'MiniMax H3 Fused Turbo — a deliberately long model display name for narrow screens', {
    rgb: false,
    tiny_vae_frames: true,
    tiny_vae_video: true,
  }),
  model('image-preview-model', 'Supported Image Model', {
    rgb: true,
    tiny_vae_frames: true,
    tiny_vae_video: true,
  }),
  model('rgb-only-model', 'RGB Only Model', {
    rgb: true,
    tiny_vae_frames: false,
    tiny_vae_video: false,
  }),
  model('legacy-model', 'Model Without Preview Metadata'),
];

(async () => {
  const entry = [
    "import React from 'react'; import {createRoot} from 'react-dom/client';",
    "import {useStore} from './src/stores/useStore';",
    "import {GenerationPreviewSetting} from './src/components/SettingsDrawer/SystemSettingsPanel';",
    "import {JobPlaceholder} from './src/components/MainContent/MainContent';",
    'function JobHarness() {',
    '  const jobs = useStore(state => state.jobs);',
    '  return <div>{jobs.map(job => <JobPlaceholder key={job.id} job={job} onDismiss={() => {}} onStop={() => {}} />)}</div>;',
    '}',
    'const root = createRoot(document.getElementById("root"));',
    'window.store = useStore;',
    'window.mountSettings = () => root.render(<GenerationPreviewSetting />);',
    'window.mountJobs = () => root.render(<JobHarness />);',
    'window.reconnectJobs = () => useStore.getState().reconnectJobs();',
  ].join('\n');
  const bundle = await esbuild.build({
    stdin: {contents: entry, resolveDir: path.join(root, 'ui'), loader: 'tsx'},
    bundle: true,
    write: false,
    jsx: 'automatic',
    define: {'process.env.NODE_ENV': '"development"'},
    logLevel: 'silent',
  });

  const browser = await playwright.chromium.launch({headless: true, ...(process.platform === 'win32' ? {
    executablePath: process.env.MAESTRO_CHROME || 'C:/Program Files/Google/Chrome/Application/chrome.exe',
  } : {})});
  try {
    let savedConfig = {attention_mode: 'auto'};
    const jobs = [
      status('image-job', {preview: preview('image-job', 'image')}),
      status('video-job', {preview: preview('video-job', 'video')}),
      status('empty-job', {preview_notice: 'Waiting for the next preview frame.'}),
      status('broken-job', {preview: {...preview('broken-job', 'image'), url: '/api/v1/previews/broken.svg'}}),
      status('terminal-job', {
        preview: preview('terminal-job', 'image'),
        enhancement: {version: 1, state: 'enhancing'},
      }),
    ];
    const polled = {
      'image-job': jobs[0],
      'video-job': jobs[1],
      'empty-job': jobs[2],
      'broken-job': jobs[3],
      'terminal-job': status('terminal-job', {
        status: 'failed',
        progress: 45,
        step: 4,
        total_steps: 10,
        phase: 'Preview complete',
        message: 'Generation failed',
        error: 'Fixture failure',
        preview: null,
        preview_notice: null,
        enhancement: {version: 1, state: 'failed'},
      }),
    };

    const openPage = async context => {
      const page = await context.newPage();
      const errors = [];
      page.on('pageerror', error => errors.push(error.message));
      await page.route('**/*', async route => {
        const url = new URL(route.request().url());
        const endpoint = url.pathname;
        if (endpoint === '/') return route.fulfill({contentType: 'text/html', body: '<div id="root"></div>'});
        if (endpoint === '/api/v1/system-config') {
          if (route.request().method() === 'PUT') savedConfig = {...savedConfig, ...route.request().postDataJSON()};
          return jsonResponse(route, savedConfig);
        }
        if (endpoint === '/api/v1/services-config') return jsonResponse(route, {auto_performance: true});
        if (endpoint === '/api/v1/jobs') return jsonResponse(route, {jobs});
        if (endpoint.startsWith('/api/v1/status/')) {
          const id = decodeURIComponent(endpoint.slice('/api/v1/status/'.length));
          return jsonResponse(route, polled[id] || status(id));
        }
        if (endpoint === '/api/v1/previews/broken.svg') {
          return route.fulfill({status: 404, body: 'missing preview'});
        }
        if (endpoint.startsWith('/api/v1/previews/')) {
          return route.fulfill({
            contentType: 'image/svg+xml',
            body: '<svg xmlns="http://www.w3.org/2000/svg" width="8" height="8"><rect width="8" height="8" fill="blue"/></svg>',
          });
        }
        return jsonResponse(route, {});
      });
      await page.goto('http://preview.test/');
      await page.addScriptTag({content: bundle.outputFiles[0].text});
      return {page, errors};
    };

    // Missing config defaults to Live Video. Explicit selections, including
    // Off, persist to server config and survive a fresh page.
    const settingsContext = await browser.newContext();
    let {page: settingsPage} = await openPage(settingsContext);
    await settingsPage.evaluate(async ({models}) => {
      await window.store.getState().loadSystemConfig();
      window.store.setState(state => ({
        servicesConfig: {auto_performance: true},
        models,
        generationMode: 'video',
        params: {...state.params, model_type: 'minimax_h3_fused_turbo'},
      }));
      window.mountSettings();
    }, {models: previewModels});
    const selector = settingsPage.getByLabel('Generation Preview', {exact: true});
    await selector.waitFor();
    assert.equal(await selector.inputValue(), 'tiny_vae_video', 'missing backend setting defaults to Live Video');
    const capabilityStatus = key => settingsPage.locator(`[data-preview-capability="${key}"] [data-preview-capability-status]`);
    assert.equal(await capabilityStatus('rgb').innerText(), 'Unavailable', 'H3 does not advertise RGB support');
    assert.equal(await capabilityStatus('tiny_vae_frames').innerText(), 'Supported', 'H3 advertises Tiny VAE frame support');
    assert.equal(await capabilityStatus('tiny_vae_video').innerText(), 'Supported', 'H3 advertises Tiny VAE video support');
    assert.equal(await selector.locator('option').count(), 4, 'all preview options remain selectable');
    const longModelName = settingsPage.getByText('MiniMax H3 Fused Turbo — a deliberately long model display name for narrow screens', {exact: true});
    await longModelName.waitFor();
    await settingsPage.setViewportSize({width: 320, height: 800});
    const wrapping = await longModelName.evaluate(element => ({
      overflowWrap: getComputedStyle(element.parentElement).overflowWrap,
      fits: element.parentElement.scrollWidth <= element.parentElement.clientWidth,
    }));
    assert.equal(wrapping.overflowWrap, 'anywhere', 'long model names can wrap at narrow widths');
    assert.equal(wrapping.fits, true, 'long model names do not overflow their available width');
    await settingsPage.setViewportSize({width: 1280, height: 800});
    await selector.selectOption('tiny_vae_frames');
    await settingsPage.waitForFunction(() => window.store.getState().systemConfig?.generation_preview === 'tiny_vae_frames');
    assert.equal(savedConfig.generation_preview, 'tiny_vae_frames', 'selection is persisted through system config');
    assert.equal(await settingsPage.evaluate(() => window.store.getState().servicesConfig.auto_performance), true, 'choosing a preview leaves auto-performance enabled');

    await settingsPage.evaluate(() => window.store.setState(state => ({
      generationMode: 'image',
      params: {...state.params, model_type: 'image-preview-model'},
    })));
    await settingsPage.getByText('Supported Image Model', {exact: true}).waitFor();
    assert.equal(await capabilityStatus('rgb').innerText(), 'Supported', 'supported image models expose RGB frames');
    assert.equal(await capabilityStatus('tiny_vae_frames').innerText(), 'Supported', 'supported image models expose clearer frames');
    assert.equal(await capabilityStatus('tiny_vae_video').innerText(), 'Supported · still previews', 'Live Video support is described as still previews in Image mode');
    await settingsPage.getByText('Image generations show still previews, including when Live Video is selected.', {exact: true}).waitFor();
    const imageDescriptions = await selector.getAttribute('aria-describedby');
    assert.ok(imageDescriptions.includes('generation-preview-image-description'), 'Image still-preview guidance is associated with the selector');
    assert.ok(await settingsPage.evaluate(ids => ids.split(/\s+/).every(id => document.getElementById(id)), imageDescriptions), 'selector description IDs resolve to visible elements');
    assert.equal(await selector.inputValue(), 'tiny_vae_frames', 'changing model and generation mode preserves the selected preview preference');

    await settingsPage.evaluate(() => window.store.setState(state => ({
      generationMode: 'video',
      params: {...state.params, model_type: 'rgb-only-model'},
    })));
    await settingsPage.getByText('RGB Only Model', {exact: true}).waitFor();
    assert.equal(await capabilityStatus('rgb').innerText(), 'Supported', 'RGB-only variants keep RGB support');
    assert.equal(await capabilityStatus('tiny_vae_frames').innerText(), 'Unavailable', 'RGB-only variants do not imply Tiny VAE frame support');
    assert.equal(await capabilityStatus('tiny_vae_video').innerText(), 'Unavailable', 'RGB-only variants do not imply Tiny VAE video support');

    await settingsPage.evaluate(() => window.store.setState(state => ({
      params: {...state.params, model_type: 'legacy-model'},
    })));
    await settingsPage.getByText('Model Without Preview Metadata', {exact: true}).waitFor();
    assert.equal(await capabilityStatus('rgb').innerText(), 'Unknown', 'missing model metadata stays unknown');
    assert.equal(await capabilityStatus('tiny_vae_frames').innerText(), 'Unknown', 'missing Tiny VAE frame metadata stays unknown');
    assert.equal(await capabilityStatus('tiny_vae_video').innerText(), 'Unknown', 'missing Tiny VAE video metadata stays unknown');
    await settingsPage.getByText("Preview support hasn't been reported for this model.", {exact: true}).waitFor();

    await settingsPage.evaluate(() => window.store.setState(state => ({
      params: {...state.params, model_type: 'missing-model'},
    })));
    await settingsPage.getByText('Selected Studio model is unavailable (missing-model).', {exact: true}).waitFor();
    await settingsPage.getByText('Choose a Studio model to check preview support.', {exact: true}).waitFor();
    assert.equal(await capabilityStatus('rgb').innerText(), 'Unknown', 'a missing selected model stays unknown');

    await settingsPage.evaluate(() => window.store.setState(state => ({
      generationMode: 'audio',
      params: {...state.params, model_type: 'minimax_h3_fused_turbo'},
    })));
    await settingsPage.getByText('Audio generation has no visual previews.', {exact: false}).waitFor();
    assert.equal(await settingsPage.locator('[data-preview-capability]').count(), 0, 'Audio mode does not advertise visual preview capabilities');
    assert.equal(await selector.inputValue(), 'tiny_vae_frames', 'Audio mode leaves the global preview preference unchanged');

    await settingsPage.evaluate(() => window.store.setState({generationMode: 'tools'}));
    await settingsPage.waitForFunction(() => document.getElementById('generation-preview-description')?.textContent?.startsWith('Tools mode does not use visual previews.'));
    assert.equal(await settingsPage.locator('[data-preview-capability]').count(), 0, 'Tools mode does not advertise visual preview capabilities');
    assert.equal(await selector.inputValue(), 'tiny_vae_frames', 'Tools mode leaves the global preview preference unchanged');
    await selector.selectOption('off');
    await settingsPage.waitForFunction(() => window.store.getState().systemConfig?.generation_preview === 'off');
    assert.equal(savedConfig.generation_preview, 'off', 'explicit Off is persisted');
    await settingsPage.close();
    ({page: settingsPage} = await openPage(settingsContext));
    await settingsPage.evaluate(async () => {
      await window.store.getState().loadSystemConfig();
      window.mountSettings();
    });
    assert.equal(await settingsPage.getByLabel('Generation Preview', {exact: true}).inputValue(), 'off', 'explicit Off survives a fresh page');
    await settingsPage.close();
    await settingsContext.close();

    // Make video playback deterministic in headless browser runs while keeping
    // the real video element, accessible control, URL updates, and React state.
    const jobsContext = await browser.newContext();
    const {page: jobsPage, errors} = await openPage(jobsContext);
    await jobsPage.addInitScript(() => {
      window.__holdPreviewPlay = false;
      window.__nextPreviewPlayError = null;
      window.__previewPending = [];
      window.__previewPlayCalls = [];
      window.__previewReadyState = 0;
      const mediaSrc = Object.getOwnPropertyDescriptor(HTMLMediaElement.prototype, 'src');
      Object.defineProperty(HTMLMediaElement.prototype, 'src', {
        configurable: true,
        get() { return this.dataset.previewSrc || (mediaSrc ? mediaSrc.get.call(this) : ''); },
        set(value) { this.dataset.previewSrc = String(value); },
      });
      const nativeSetAttribute = Element.prototype.setAttribute;
      Element.prototype.setAttribute = function(name, value) {
        if (this instanceof HTMLVideoElement && name.toLowerCase() === 'src') {
          this.dataset.previewSrc = String(value);
          return;
        }
        nativeSetAttribute.call(this, name, value);
      };
      Object.defineProperty(HTMLMediaElement.prototype, 'paused', {
        configurable: true,
        get() { return this.dataset.previewPlayback !== 'playing'; },
      });
      Object.defineProperty(HTMLMediaElement.prototype, 'readyState', {
        configurable: true,
        get() { return window.__previewReadyState; },
      });
      HTMLMediaElement.prototype.play = function() {
        this.dataset.previewPlayback = 'playing';
        window.__previewPlayCalls.push({
          src: this.dataset.previewSrc,
          readyState: this.readyState,
          autoplay: this.autoplay,
          muted: this.muted,
          defaultMuted: this.defaultMuted,
          playsInline: this.playsInline,
        });
        if (window.__nextPreviewPlayError) {
          const name = window.__nextPreviewPlayError;
          window.__nextPreviewPlayError = null;
          const error = new Error(`preview play rejected: ${name}`);
          error.name = name;
          return Promise.reject(error);
        }
        if (window.__holdPreviewPlay) {
          return new Promise((resolve, reject) => window.__previewPending.push({
            src: this.dataset.previewSrc,
            resolve,
            reject,
          }));
        }
        return Promise.resolve();
      };
      HTMLMediaElement.prototype.pause = function() {
        this.dataset.previewPlayback = 'paused';
      };
    });
    // The init script is registered before any navigation script runs; reload
    // so the media shims are installed before the app bundle mounts.
    await jobsPage.reload();
    await jobsPage.addScriptTag({content: bundle.outputFiles[0].text});
    await jobsPage.evaluate(() => window.mountJobs());
    // Safari may reject automatic playback even for a muted inline video.
    // Keep the fixture at readyState 0 and reject the first play() attempt to
    // exercise immediate initiation and the tap-to-play recovery path.
    await jobsPage.evaluate(() => { window.__nextPreviewPlayError = 'NotAllowedError'; });
    await jobsPage.evaluate(() => window.reconnectJobs());
    await jobsPage.waitForFunction(() => window.store.getState().jobs.length === 5);

    const imageCard = jobsPage.locator('[data-generation-job-id="image-job"]');
    const videoCard = jobsPage.locator('[data-generation-job-id="video-job"]');
    const emptyCard = jobsPage.locator('[data-generation-job-id="empty-job"]');
    const brokenCard = jobsPage.locator('[data-generation-job-id="broken-job"]');
    const terminalCard = jobsPage.locator('[data-generation-job-id="terminal-job"]');
    await imageCard.getByText('Clip 1/2 · Window 2/5').first().waitFor();
    await imageCard.getByText('Full Studio render ~50s').waitFor();
    await imageCard.locator('img').waitFor();
    assert.equal(await emptyCard.locator('[data-generation-preview]').count(), 0, 'older job status without preview renders its normal placeholder');
    await emptyCard.getByText('Waiting for the next preview frame.').waitFor();
    await brokenCard.getByText('Preview unavailable').waitFor();
    await brokenCard.getByText('Step 4/10').waitFor();
    assert.equal(await jobsPage.evaluate(() => window.store.getState().jobs.find(job => job.id === 'broken-job').status), 'running', 'preview load errors do not change job state');
    assert.equal(await jobsPage.evaluate(() => window.store.getState().jobs.find(job => job.id === 'broken-job').progress), 0.45, 'preview errors do not alter progress');

    const video = videoCard.locator('video');
    assert.equal(await video.evaluate(element => element.muted), true, 'live preview is muted');
    assert.equal(await video.evaluate(element => element.defaultMuted), true, 'live preview has a muted HTML attribute for Safari autoplay policy');
    assert.equal(await video.evaluate(element => element.loop), true, 'live preview loops');
    assert.equal(await video.evaluate(element => element.playsInline), true, 'live preview plays inline');
    assert.equal(await videoCard.locator('video').count(), 1, 'a blocked autoplay keeps the video element mounted');
    await videoCard.getByRole('button', {name: 'Play live preview'}).waitFor();
    assert.ok(await jobsPage.evaluate(() => window.__previewPlayCalls.some(call => (
      call.readyState === 0 && call.autoplay && call.muted && call.defaultMuted && call.playsInline
    ))), 'play() starts before loaded data with explicit muted inline autoplay attributes');
    await videoCard.getByRole('button', {name: 'Play live preview'}).click();
    await videoCard.getByRole('button', {name: 'Pause live preview'}).waitFor();
    assert.equal(await video.getAttribute('data-preview-playback'), 'playing', 'the accessible Play control recovers from a policy-blocked autoplay');
    await videoCard.getByRole('button', {name: 'Hide generation information'}).click();
    await videoCard.getByRole('button', {name: 'Show generation information'}).waitFor();
    assert.equal(await videoCard.getByText('Full Studio render ~50s').isVisible(), false, 'clicking the preview hides its information');
    assert.equal(await video.getAttribute('data-preview-playback'), 'playing', 'hiding information does not pause playback');
    await videoCard.getByRole('button', {name: 'Show generation information'}).press('Enter');
    await videoCard.getByText('Full Studio render ~50s').waitFor();
    await videoCard.getByRole('button', {name: 'Hide generation information'}).press('Space');
    await videoCard.getByRole('button', {name: 'Pause live preview'}).click();
    await videoCard.getByRole('button', {name: 'Play live preview'}).waitFor();
    assert.equal(await video.getAttribute('data-preview-playback'), 'paused', 'the separate Pause control pauses playback');
    assert.equal(await videoCard.getByRole('button', {name: 'Show generation information'}).count(), 1, 'Pause does not change overlay visibility');

    // Only video-job advances. Reconnect polling must update that job's media
    // without replacing another job's preview or undoing a user pause.
    polled['video-job'] = status('video-job', {preview: preview('video-job', 'video', 2)});
    await jobsPage.waitForFunction(() => window.store.getState().jobs.find(job => job.id === 'video-job')?.preview?.revision === 2, null, {timeout: 5000});
    assert.equal(await jobsPage.evaluate(() => window.store.getState().jobs.find(job => job.id === 'image-job').preview.revision), 1, 'unrelated job keeps its own preview revision');
    assert.match(await video.evaluate(element => element.dataset.previewSrc), /revision=2$/, 'video source advances with the new preview revision');
    await videoCard.getByRole('button', {name: 'Play live preview'}).waitFor();
    assert.equal(await video.getAttribute('data-preview-playback'), 'paused', 'revision updates preserve the paused state');
    assert.equal(await videoCard.getByRole('button', {name: 'Show generation information'}).count(), 1, 'revision updates preserve hidden information');

    // A play() promise that resolves after a pause must not clear
    // the paused state. The next revision also supersedes any older
    // promise still in flight.
    await jobsPage.evaluate(() => { window.__holdPreviewPlay = true; });
    await videoCard.getByRole('button', {name: 'Play live preview'}).click();
    await videoCard.getByRole('button', {name: 'Pause live preview'}).waitFor();
    assert.equal(await video.getAttribute('data-preview-playback'), 'playing', 'the accessible control resumes playback');
    await videoCard.getByRole('button', {name: 'Pause live preview'}).click();
    await videoCard.getByRole('button', {name: 'Play live preview'}).waitFor();
    await jobsPage.evaluate(() => window.__previewPending.splice(0).forEach(pending => pending.resolve()));
    await jobsPage.waitForTimeout(25);
    assert.equal(await videoCard.getByRole('button', {name: 'Play live preview'}).count(), 1, 'a late play resolution cannot undo a pause');
    assert.equal(await video.getAttribute('data-preview-playback'), 'paused');

    // Start another pending request, supersede it with a newer preview, then
    // pause the current media before the old request resolves.
    await videoCard.getByRole('button', {name: 'Play live preview'}).click();
    await jobsPage.evaluate(() => {
      window.__holdPreviewPlay = false;
      window.store.setState(state => ({jobs: state.jobs.map(job => job.id === 'video-job'
        ? {...job, preview: {...job.preview, revision: 3, url: '/api/v1/previews/video-job?revision=3'}}
        : job)}));
    });
    await jobsPage.waitForFunction(() => window.store.getState().jobs.find(job => job.id === 'video-job')?.preview?.revision === 3);
    await videoCard.getByRole('button', {name: 'Pause live preview'}).waitFor();
    await videoCard.getByRole('button', {name: 'Pause live preview'}).click();
    await videoCard.getByRole('button', {name: 'Play live preview'}).waitFor();
    await jobsPage.evaluate(() => window.__previewPending.splice(0).forEach(pending => pending.resolve()));
    await jobsPage.waitForTimeout(25);
    assert.equal(await videoCard.getByRole('button', {name: 'Play live preview'}).count(), 1, 'a superseded revision promise cannot reset the paused state');
    assert.equal(await video.getAttribute('data-preview-playback'), 'paused');

    // Backend window transitions intentionally clear preview. The JobPlaceholder
    // survives that gap, so its preference must carry into the next media node.
    polled['video-job'] = status('video-job', {preview: null, preview_notice: 'Preparing next window.'});
    await jobsPage.evaluate(() => window.store.setState(state => ({jobs: state.jobs.map(job => job.id === 'video-job'
      ? {...job, preview: null, previewNotice: null}
      : job)})));
    await videoCard.locator('video').waitFor({state: 'detached'});
    await videoCard.getByText('Step 4/10').waitFor();
    const nextWindowPreview = preview('video-job', 'video', 4);
    nextWindowPreview.window = 3;
    polled['video-job'] = status('video-job', {preview: nextWindowPreview});
    await jobsPage.evaluate(() => window.store.setState(state => ({jobs: state.jobs.map(job => job.id === 'video-job'
      ? {...job, preview: {url: '/api/v1/previews/video-job?revision=4', kind: 'video', revision: 4, mode: 'tiny_vae_video', window: 3, total_windows: 5, clip: 1, total_clips: 2}}
      : job)})));
    await videoCard.locator('video').waitFor();
    await videoCard.getByRole('button', {name: 'Play live preview'}).waitFor();
    const resumedWindowVideo = videoCard.locator('video');
    assert.equal(await resumedWindowVideo.getAttribute('data-preview-playback'), 'paused', 'pause preference survives a null preview gap and next-window remount');
    assert.equal(await videoCard.getByRole('button', {name: 'Show generation information'}).count(), 1, 'hidden information survives a null preview gap and next-window remount');

    // A separate job owns a fresh preference and starts playing normally.
    await jobsPage.evaluate(() => {
      const source = window.store.getState().jobs.find(job => job.id === 'video-job');
      window.store.setState(state => ({jobs: [...state.jobs, {
        ...source,
        id: 'fresh-video-job',
        preview: {...source.preview, url: '/api/v1/previews/fresh-video-job?revision=1', revision: 1},
      }]}));
    });
    const freshVideoCard = jobsPage.locator('[data-generation-job-id="fresh-video-job"]');
    await freshVideoCard.getByRole('button', {name: 'Pause live preview'}).waitFor();
    assert.equal(await freshVideoCard.locator('video').getAttribute('data-preview-playback'), 'playing', 'a different job starts with a fresh playback preference');
    assert.equal(await freshVideoCard.getByRole('button', {name: 'Hide generation information'}).count(), 1, 'a different job starts with visible information');

    // A policy-blocked autoplay is tied to its revision, not stored as a user
    // pause. Advancing the preview retries playback without a tap.
    await jobsPage.evaluate(() => {
      window.__nextPreviewPlayError = 'NotAllowedError';
      const source = window.store.getState().jobs.find(job => job.id === 'video-job');
      window.store.setState(state => ({jobs: [...state.jobs, {
        ...source,
        id: 'blocked-retry-job',
        preview: {...source.preview, url: '/api/v1/previews/blocked-retry-job?revision=1', revision: 1},
      }]}));
    });
    const blockedRetryCard = jobsPage.locator('[data-generation-job-id="blocked-retry-job"]');
    await blockedRetryCard.getByRole('button', {name: 'Play live preview'}).waitFor();
    await jobsPage.evaluate(() => window.store.setState(state => ({jobs: state.jobs.map(job => job.id === 'blocked-retry-job'
      ? {...job, preview: {...job.preview, url: '/api/v1/previews/blocked-retry-job?revision=2', revision: 2}}
      : job)})));
    await blockedRetryCard.getByRole('button', {name: 'Pause live preview'}).waitFor();
    assert.equal(await blockedRetryCard.locator('video').getAttribute('data-preview-playback'), 'playing', 'a new revision retries after policy-blocked autoplay without a tap');

    // AbortError is transient during source/media transitions. Keep the video
    // and Play control available so the user can recover with a tap.
    await jobsPage.evaluate(() => {
      window.__nextPreviewPlayError = 'AbortError';
      const source = window.store.getState().jobs.find(job => job.id === 'video-job');
      window.store.setState(state => ({jobs: [...state.jobs, {
        ...source,
        id: 'aborted-video-job',
        preview: {...source.preview, url: '/api/v1/previews/aborted-video-job?revision=1'},
      }]}));
    });
    const abortedVideoCard = jobsPage.locator('[data-generation-job-id="aborted-video-job"]');
    await abortedVideoCard.getByRole('button', {name: 'Play live preview'}).waitFor();
    assert.equal(await abortedVideoCard.locator('video').count(), 1, 'AbortError does not remove the live preview');
    assert.equal(await abortedVideoCard.getByText('Preview unavailable').count(), 0, 'AbortError is not shown as a media failure');
    await abortedVideoCard.getByRole('button', {name: 'Play live preview'}).click();
    await abortedVideoCard.getByRole('button', {name: 'Pause live preview'}).waitFor();

    // A stale policy rejection from an older revision cannot pause or fail the
    // replacement video after its own play() request succeeds.
    await jobsPage.evaluate(() => {
      window.__holdPreviewPlay = true;
      const source = window.store.getState().jobs.find(job => job.id === 'video-job');
      window.store.setState(state => ({jobs: [...state.jobs, {
        ...source,
        id: 'stale-rejection-job',
        preview: {...source.preview, url: '/api/v1/previews/stale-rejection-job?revision=1', revision: 1},
      }]}));
    });
    const staleRejectionCard = jobsPage.locator('[data-generation-job-id="stale-rejection-job"]');
    await jobsPage.waitForFunction(() => window.__previewPending.some(pending => pending.src.endsWith('stale-rejection-job?revision=1')));
    await jobsPage.evaluate(() => {
      window.__holdPreviewPlay = false;
      window.store.setState(state => ({jobs: state.jobs.map(job => job.id === 'stale-rejection-job'
        ? {...job, preview: {...job.preview, url: '/api/v1/previews/stale-rejection-job?revision=2', revision: 2}}
        : job)}));
    });
    await jobsPage.waitForFunction(() => window.store.getState().jobs.find(job => job.id === 'stale-rejection-job')?.preview?.revision === 2);
    await staleRejectionCard.getByRole('button', {name: 'Pause live preview'}).waitFor();
    await jobsPage.evaluate(() => {
      const pending = window.__previewPending.find(item => item.src.endsWith('stale-rejection-job?revision=1'));
      pending.reject(Object.assign(new Error('stale autoplay rejection'), {name: 'NotAllowedError'}));
    });
    await jobsPage.waitForTimeout(25);
    assert.equal(await staleRejectionCard.getByRole('button', {name: 'Pause live preview'}).count(), 1, 'a stale rejection cannot pause a newer revision');
    assert.equal(await staleRejectionCard.getByText('Preview unavailable').count(), 0, 'a stale rejection cannot fail a newer revision');
    assert.equal(await staleRejectionCard.locator('video').getAttribute('data-preview-playback'), 'playing');

    await jobsPage.waitForFunction(() => {
      const job = window.store.getState().jobs.find(item => item.id === 'terminal-job');
      return job?.status === 'failed' && job.preview === null && job.previewNotice === null;
    }, null, {timeout: 5000});
    assert.equal(await terminalCard.locator('[data-generation-preview]').count(), 0, 'terminal null clears the preview');
    assert.deepEqual(errors, [], 'preview errors stay local to the media tile');
    await jobsContext.close();
    console.log('Generation previews passed: settings persistence, image/video rendering, Safari-safe play initiation and recovery, pause controls and async races, null-gap preference, per-job isolation, older backend, errors, progress, and terminal clearing.');
  } finally {
    await browser.close();
  }
})().catch(error => {console.error(error); process.exitCode = 1;});
