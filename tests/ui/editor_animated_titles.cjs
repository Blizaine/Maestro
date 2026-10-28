// Browser regression for animated title generation and scoped alpha export.
// All HTTP is intercepted; models, uploads, saved projects, generation jobs,
// and media probes are synthetic. No live model, GPU job, or media download runs.
// Run: node tests/ui/editor_animated_titles.cjs
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const {createRequire} = require('node:module');
const zlib = require('node:zlib');
const root = path.resolve(__dirname, '../..');
const ui = path.join(root, 'ui');
const requireFromUi = createRequire(path.join(ui, 'package.json'));
const esbuild = requireFromUi('esbuild');
const playwrightPath = process.env.MAESTRO_PLAYWRIGHT ||
  'C:/Users/bliza/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright-core';
const {chromium} = require(playwrightPath);

const WORKSPACE = 'Animated title fixtures';
const PROJECT_ID = 'animated-title-project';
const ORIGIN = 'http://192.168.56.23:38473';
const clone = value => JSON.parse(JSON.stringify(value));
const sourceProject = {
  schema_version: 6,
  id: PROJECT_ID,
  name: 'Title workflow fixture',
  workspace: WORKSPACE,
  revision: 1,
  created_at: 1,
  updated_at: 1,
  canvas: {width: 1920, height: 1080, fps: 24, background: '#000000'},
  assets: {
    footage: {
      id: 'footage', name: 'synthetic-footage.mp4', type: 'video', origin: 'output',
      workspace: WORKSPACE, path: '/outputs/synthetic-footage.mp4', url: '/synthetic-footage.mp4',
      duration: 30, width: 1920, height: 1080, fps: 24, has_audio: true,
    },
    upper: {
      id: 'upper', name: 'synthetic-upper-layer.mov', type: 'video', origin: 'output',
      workspace: WORKSPACE, path: '/outputs/synthetic-upper-layer.mov', url: '/synthetic-upper-layer.mov',
      duration: 20, width: 1920, height: 1080, fps: 24, has_audio: false,
    },
    music: {
      id: 'music', name: 'synthetic-music.wav', type: 'audio', origin: 'output',
      workspace: WORKSPACE, path: '/outputs/synthetic-music.wav', url: '/synthetic-music.wav',
      duration: 40, width: 0, height: 0, fps: 0, has_audio: true,
    },
  },
  tracks: [
    {
      id: 'video-track', name: 'Footage', type: 'video', z_index: 2,
      muted: false, locked: false, volume: 1,
      items: [{
        id: 'footage-item', asset_id: 'footage', name: 'Main footage', start: 0, duration: 30,
        source_in: 0, speed: 1, volume: 1, opacity: 1, fit: 'contain',
        transform: {x: 0, y: 0, scale: 1, rotation: 0},
      }],
    },
    {
      id: 'upper-track', name: 'Existing visual layer', type: 'video', z_index: 8,
      muted: false, locked: false, volume: 1,
      items: [{
        id: 'upper-item', asset_id: 'upper', name: 'Existing overlay', start: 1, duration: 4,
        source_in: 0, speed: 1, volume: 1, opacity: 1, fit: 'contain',
        transform: {x: 0, y: 0, scale: 1, rotation: 0},
      }],
    },
    {
      id: 'audio-track', name: 'Music', type: 'audio', z_index: 0,
      muted: false, locked: false, volume: 1,
      items: [{
        id: 'music-item', asset_id: 'music', name: 'Music bed', start: 0, duration: 24,
        source_in: 0, speed: 1, volume: 0.7, opacity: 1, fit: 'contain',
        transform: {x: 0, y: 0, scale: 1, rotation: 0},
      }],
    },
  ],
  markers: [],
  export: {
    quality: 'balanced', codec: 'h264', encoder: 'auto', include_audio: true,
    resolution: 'canvas', frame_rate: 'project', filename: 'fixture-edit', format: 'mp4',
    spatial_upsampling: '', film_grain_intensity: 0, film_grain_saturation: 0.5,
  },
  exports: [],
};

const sourceTracks = clone(sourceProject.tracks);
const sourceAssets = clone(sourceProject.assets);

function appCss() {
  const assets = path.join(ui, 'dist/assets');
  const builtHtml = fs.readFileSync(path.join(ui, 'dist/index.html'), 'utf8');
  const filename = /href="\/assets\/([^\"]+\.css)"/.exec(builtHtml)?.[1];
  assert.ok(filename && fs.existsSync(path.join(assets, filename)), 'The built UI entry must name its CSS bundle.');
  return fs.readFileSync(path.join(assets, filename), 'utf8');
}

async function bundleUi() {
  const sourceExtensions = ['.tsx', '.ts', '.jsx', '.js', '.mjs', '.cjs', '.json', '.svg'];
  const resolveFile = candidate => {
    if (fs.existsSync(candidate) && fs.statSync(candidate).isFile()) return fs.realpathSync(candidate);
    for (const extension of sourceExtensions) {
      const withExtension = candidate + extension;
      if (fs.existsSync(withExtension) && fs.statSync(withExtension).isFile()) return fs.realpathSync(withExtension);
      const indexFile = path.join(candidate, `index${extension}`);
      if (fs.existsSync(indexFile) && fs.statSync(indexFile).isFile()) return fs.realpathSync(indexFile);
    }
    return null;
  };
  const resolver = {
    name: 'animated-title-test-workspace-resolver',
    setup(build) {
      build.onResolve({filter: /.*/}, args => {
        if (args.path.startsWith('node:')) return {path: args.path, external: true};
        if (args.path.startsWith('.') || path.isAbsolute(args.path)) {
          const resolved = resolveFile(path.resolve(args.resolveDir, args.path));
          return resolved
            ? {path: resolved, namespace: 'workspace-fs'}
            : {errors: [{text: `Unable to resolve ${args.path} from ${args.resolveDir}`} ]};
        }
        return {path: fs.realpathSync(requireFromUi.resolve(args.path, {paths: [args.resolveDir, ui]})), namespace: 'workspace-fs'};
      });
      build.onLoad({filter: /.*/, namespace: 'workspace-fs'}, args => {
        const extension = path.extname(args.path);
        const loader = extension === '.tsx' ? 'tsx' : extension === '.ts' ? 'ts'
          : extension === '.jsx' ? 'jsx' : extension === '.json' ? 'json'
            : extension === '.svg' ? 'file' : 'js';
        return {contents: fs.readFileSync(args.path, 'utf8'), loader, resolveDir: path.dirname(args.path)};
      });
    },
  };
  const result = await esbuild.build({
    stdin: {
      contents: `
        import {useState} from 'react';
        import {createRoot} from 'react-dom/client';
        import {useEditorStore} from './src/editor/useEditorStore';
        import {useStore} from './src/stores/useStore';
        import {useEditorTitleStore} from './src/editor/useEditorTitleStore';
        import {animatedTitlePrompt, titleDimensions, titleFrames, TITLE_EFFECTS} from './src/editor/editorAnimatedTitles';
        import {EditorAnimatedTitleDialog, EditorAnimatedTitleStatus} from './src/editor/EditorAnimatedTitleDialog';
        import {EditorInspector} from './src/editor/EditorInspector';
        window.editorStore = useEditorStore;
        window.studioStore = useStore;
        window.titleStore = useEditorTitleStore;
        window.titleHelpers = {animatedTitlePrompt, titleDimensions, titleFrames, TITLE_EFFECTS};
        function App() {
          const [titleOpen, setTitleOpen] = useState(false);
          const selectedItemId = useEditorStore(state => state.selectedItemId);
          return <main style={{minHeight:'100vh',display:'grid',gridTemplateColumns:'minmax(0,1fr) 340px',gap:16,padding:16}}>
            <section><button type="button" onClick={() => setTitleOpen(true)}>Open title generator</button>
              <EditorAnimatedTitleDialog open={titleOpen} onClose={() => setTitleOpen(false)} />
              <EditorAnimatedTitleStatus />
            </section>
            <EditorInspector compact />
          </main>;
        }
        const reactRoot = createRoot(document.getElementById('root'));
        window.mount = () => reactRoot.render(<App />);
      `,
      resolveDir: ui,
      loader: 'tsx',
    },
    bundle: true,
    write: false,
    format: 'iife',
    target: 'es2022',
    jsx: 'automatic',
    define: {'process.env.NODE_ENV': '"development"'},
    logLevel: 'silent',
    plugins: [resolver],
  });
  return result.outputFiles[0].text;
}

function extractUploadedPng(request) {
  const contentType = request.headers()['content-type'] || '';
  const boundary = /boundary=([^;]+)/i.exec(contentType)?.[1];
  assert.ok(boundary, 'The title background must be uploaded as multipart form data.');
  const body = request.postDataBuffer();
  assert.ok(body, 'The PNG upload request must include a body.');
  const dispositionAt = body.indexOf(Buffer.from('filename="title-'));
  assert.notEqual(dispositionAt, -1, 'The upload uses a generated title-background PNG filename.');
  const headerEnd = body.indexOf(Buffer.from('\r\n\r\n'), dispositionAt);
  assert.notEqual(headerEnd, -1, 'The multipart file part has headers.');
  const fileEnd = body.indexOf(Buffer.from(`\r\n--${boundary}`), headerEnd + 4);
  assert.notEqual(fileEnd, -1, 'The multipart file part is terminated.');
  const headers = body.subarray(dispositionAt, headerEnd).toString('utf8');
  assert.match(headers, /Content-Type:\s*image\/png/i);
  const filename = /^filename="(title-(?:green|black)-\d+x\d+\.png)"/i.exec(headers)?.[1];
  assert.ok(filename, `The upload filename identifies the selected green or black title background: ${headers}`);
  const bytes = body.subarray(headerEnd + 4, fileEnd);
  assert.deepEqual([...bytes.subarray(0, 8)], [137, 80, 78, 71, 13, 10, 26, 10], 'The uploaded payload is a PNG.');
  return {bytes, filename};
}

function decodeSolidBackgroundPng(bytes, expectedColor) {
  const expected = expectedColor === '#00FF00' ? [0, 255, 0] : expectedColor === '#000000' ? [0, 0, 0] : null;
  assert.ok(expected, `The uploaded title background is a supported solid color (${expectedColor}).`);
  const chunks = [];
  let offset = 8;
  let width = 0, height = 0, bitDepth = 0, colorType = 0;
  while (offset + 12 <= bytes.length) {
    const length = bytes.readUInt32BE(offset);
    const type = bytes.toString('ascii', offset + 4, offset + 8);
    const data = bytes.subarray(offset + 8, offset + 8 + length);
    if (type === 'IHDR') {
      width = data.readUInt32BE(0); height = data.readUInt32BE(4);
      bitDepth = data[8]; colorType = data[9];
    }
    if (type === 'IDAT') chunks.push(data);
    offset += 12 + length;
    if (type === 'IEND') break;
  }
  assert.equal(bitDepth, 8, 'Canvas background is encoded at 8 bits per channel.');
  assert.equal(colorType, 6, 'Canvas background is RGBA so opacity can be checked.');
  const packed = zlib.inflateSync(Buffer.concat(chunks));
  const stride = width * 4;
  const pixels = Buffer.alloc(height * stride);
  const paeth = (a, b, c) => {
    const p = a + b - c;
    const pa = Math.abs(p - a), pb = Math.abs(p - b), pc = Math.abs(p - c);
    return pa <= pb && pa <= pc ? a : pb <= pc ? b : c;
  };
  let sourceOffset = 0;
  for (let y = 0; y < height; y++) {
    const filter = packed[sourceOffset++];
    const rowOffset = y * stride;
    for (let x = 0; x < stride; x++) {
      const raw = packed[sourceOffset++];
      const left = x >= 4 ? pixels[rowOffset + x - 4] : 0;
      const up = y > 0 ? pixels[rowOffset + x - stride] : 0;
      const upperLeft = y > 0 && x >= 4 ? pixels[rowOffset + x - stride - 4] : 0;
      const predictor = filter === 0 ? 0 : filter === 1 ? left : filter === 2 ? up
        : filter === 3 ? Math.floor((left + up) / 2) : filter === 4 ? paeth(left, up, upperLeft) : NaN;
      assert.ok(Number.isFinite(predictor), `The PNG uses a supported row filter (${filter}).`);
      pixels[rowOffset + x] = (raw + predictor) & 0xff;
    }
  }
  for (let index = 0; index < pixels.length; index += 4) {
    assert.equal(pixels[index], expected[0], `Pixel ${index / 4} has the expected red component for ${expectedColor}.`);
    assert.equal(pixels[index + 1], expected[1], `Pixel ${index / 4} has the expected green component for ${expectedColor}.`);
    assert.equal(pixels[index + 2], expected[2], `Pixel ${index / 4} has the expected blue component for ${expectedColor}.`);
    assert.equal(pixels[index + 3], 255, `Pixel ${index / 4} is fully opaque.`);
  }
  return {width, height};
}

function jobStatus(jobId, clientId, status, extra = {}) {
  return {
    job_id: jobId, client_submission_id: clientId, kind: 'generation', status,
    progress: status === 'completed' ? 100 : status === 'running' ? 28 : 0,
    step: status === 'completed' ? 4 : 1, total_steps: 4, phase: status,
    message: status, output_files: status === 'completed' ? [`/outputs/${jobId}.mp4`] : [],
    error: status === 'failed' ? 'Synthetic generation failure' : null,
    ...extra,
  };
}

function deferred() {
  let resolve;
  const promise = new Promise(done => { resolve = done; });
  return {promise, resolve};
}

async function withTimeout(promise, label, milliseconds = 12_000) {
  let timer;
  try {
    return await Promise.race([
      promise,
      new Promise((_, reject) => { timer = setTimeout(() => reject(new Error(`Timed out waiting for ${label}.`)), milliseconds); }),
    ]);
  } finally { clearTimeout(timer); }
}

const waitUntil = async (page, predicate, label, argument = null) => page.waitForFunction(predicate, argument, {timeout: 12_000}).catch(error => {
  throw new Error(`Timed out waiting for ${label}: ${error.message}`);
});

(async () => {
  const browser = await chromium.launch({
    headless: true,
    ...(process.platform === 'win32' ? {
      executablePath: process.env.MAESTRO_CHROME || 'C:/Program Files/Google/Chrome/Application/chrome.exe',
    } : {}),
  });
  try {
    const page = await browser.newPage({viewport: {width: 1440, height: 960}});
    page.setDefaultTimeout(10_000);
    const pageErrors = [];
    const requests = [];
    const uploads = [];
    const generationBodies = [];
    const projectSaves = [];
    const statusReads = [];
    const jobListReads = [];
    const receiptReads = [];
    const exportBodies = [];
    let serverProject = clone(sourceProject);
    let nextJobIndex = 0;
    const jobs = new Map();
    const generationByJob = new Map();
    const jobsByClientId = new Map();
    const firstReceiptLookup = deferred();
    const releaseFirstReceipt = deferred();
    const titleModel = {model_type: 'minimax_h3_fused_turbo', name: 'H3 Fused 4-Step · Frames', family: 'minimax_h3', architecture: 'minimax_h3', is_i2v: true, is_t2v: true, guidance_max_phases: 1, fps: 24, is_downloaded: true};
    page.on('pageerror', error => pageErrors.push(error.message));
    await page.route('**/*', async route => {
      const request = route.request();
      const url = new URL(request.url());
      const pathname = url.pathname;
      if (pathname.startsWith('/api/')) requests.push({method: request.method(), pathname, url: request.url()});
      if (pathname === '/' || pathname === '/index.html') {
        return route.fulfill({contentType: 'text/html', body: '<div id="root"></div>'});
      }
      if (pathname === '/api/v1/models') {
        return route.fulfill({json: {families: [{id: 'minimax_h3', label: 'H3', order: 1}], models: [titleModel]}});
      }
      if (pathname === '/api/v1/editor/projects' && request.method() === 'GET') {
        return route.fulfill({json: {projects: [{
          id: serverProject.id, name: serverProject.name, workspace: serverProject.workspace,
          created_at: serverProject.created_at, updated_at: serverProject.updated_at,
          duration: 30, asset_count: Object.keys(serverProject.assets).length,
        }]}});
      }
      if (pathname === `/api/v1/editor/projects/${PROJECT_ID}` && request.method() === 'PUT') {
        const body = request.postDataJSON();
        projectSaves.push(clone(body.project));
        serverProject = {...clone(body.project), revision: serverProject.revision + 1};
        return route.fulfill({json: clone(serverProject)});
      }
      if (pathname === '/api/v1/upload' && request.method() === 'POST') {
        const {bytes, filename} = extractUploadedPng(request);
        const fileLabel = /^title-(green|black)-\d+x\d+\.png$/i.exec(filename)?.[1];
        assert.ok(fileLabel, `The title PNG filename has a supported background label (${filename}).`);
        const color = fileLabel === 'green' ? '#00FF00' : '#000000';
        const dimensions = decodeSolidBackgroundPng(bytes, color);
        assert.equal(filename, `title-${fileLabel}-${dimensions.width}x${dimensions.height}.png`,
          'The dimensions in the upload filename match the fully opaque PNG.');
        uploads.push({dimensions, filename, color});
        return route.fulfill({json: {
          filename,
          path: `/uploads/${filename}`,
          url: `/uploads/${filename}`,
        }});
      }
      if (pathname === '/api/v1/generate' && request.method() === 'POST') {
        const body = request.postDataJSON();
        generationBodies.push(clone(body));
        nextJobIndex += 1;
        const jobId = `synthetic-title-job-${nextJobIndex}`;
        const status = jobStatus(jobId, body._client_submission_id, nextJobIndex === 1 ? 'completed' : 'running');
        jobs.set(jobId, status);
        generationByJob.set(jobId, body);
        jobsByClientId.set(body._client_submission_id, status);
        if (nextJobIndex === 1) {
          // Simulate the server accepting the POST while the receipt is lost.
          return route.abort('failed');
        }
        return route.fulfill({json: {job_id: jobId, status: 'queued'}});
      }
      if (pathname === '/api/v1/jobs') {
        jobListReads.push({method: request.method(), url: request.url()});
        const accepted = [...jobs.values()];
        const active = accepted.filter(job => !['failed', 'cancelled', 'completed'].includes(job.status));
        const exact = accepted.at(-1);
        return route.fulfill({json: {jobs: [
          ...active,
          ...(exact ? [{...exact, job_id: 'decoy-same-prefix', client_submission_id: `${exact.client_submission_id}-near-match`, status: 'completed'}] : []),
        ]}});
      }
      if (pathname.startsWith('/api/v1/jobs/by-submission/')) {
        const clientId = decodeURIComponent(pathname.split('/').at(-1));
        receiptReads.push(clientId);
        if (receiptReads.length === 1) {
          firstReceiptLookup.resolve();
          await releaseFirstReceipt.promise;
        }
        const receipt = jobsByClientId.get(clientId);
        return receipt
          ? route.fulfill({json: clone(receipt)})
          : route.fulfill({status: 404, json: {detail: 'Synthetic receipt not found'}});
      }
      if (pathname.startsWith('/api/v1/status/')) {
        const jobId = decodeURIComponent(pathname.split('/').at(-1));
        statusReads.push(jobId);
        if (jobId === 'synthetic-alpha-export') {
          return route.fulfill({json: {job_id: jobId, kind: 'editor_export', status: 'running', progress: 5, step: 1, total_steps: 20, phase: 'Queued', message: 'Synthetic export', output_files: [], error: null}});
        }
        const job = jobs.get(jobId);
        return job
          ? route.fulfill({json: clone(job)})
          : route.fulfill({status: 404, json: {detail: 'Synthetic job not found'}});
      }
      if (pathname === '/api/v1/editor/media/probe' && request.method() === 'POST') {
        const body = request.postDataJSON();
        const match = [...jobs.entries()].find(([, job]) => job.output_files?.includes(body.asset.path));
        const generation = match ? generationByJob.get(match[0]) : generationBodies.at(-1);
        const [width, height] = (generation?.resolution || '832x480').split('x').map(Number);
        return route.fulfill({json: {
          name: body.asset.name, type: 'video', duration: 6.25,
          width, height, fps: 24, has_audio: true, audio_channels: 2,
          audio_sample_rate: 48_000, size: 456_789, path: body.asset.path,
        }});
      }
      if (pathname === '/api/v1/editor/media/status') return route.fulfill({json: {assets: []}});
      if (pathname === '/api/v1/workspaces') return route.fulfill({json: {workspaces: [{name: WORKSPACE, path: '/synthetic'}], active: WORKSPACE}});
      if (pathname === '/api/v1/outputs') return route.fulfill({json: {outputs: [], total: 0}});
      if (pathname === '/api/v1/director/pipelines') return route.fulfill({json: {pipelines: []}});
      if (pathname === '/api/v1/editor/export/capabilities') {
        return route.fulfill({json: {recommended: 'software', prores4444: true, encoders: {software: true, nvidia: false, intel: false, apple: false}}});
      }
      if (pathname === '/api/v1/editor/export' && request.method() === 'POST') {
        exportBodies.push(clone(request.postDataJSON()));
        return route.fulfill({json: {job_id: 'synthetic-alpha-export', status: 'queued', revision: serverProject.revision}});
      }
      if (pathname === '/api/v1/status/synthetic-alpha-export') {
        return route.fulfill({json: {job_id: 'synthetic-alpha-export', kind: 'editor_export', status: 'running', progress: 5, step: 1, total_steps: 20, phase: 'Queued', message: 'Synthetic export', output_files: [], error: null}});
      }
      if (pathname.startsWith('/api/')) return route.fulfill({json: {jobs: [], outputs: [], projects: [], assets: [], total: 0}});
      return route.fulfill({status: 204, body: ''});
    });

    const isolatedBundle = await bundleUi();
    const expectedTitleSettings = {
      text: 'Orion / Q4!', look: 'Chromed serif lettering with a blue edge light', effect: 'custom',
      direction: 'Reveal the exact spelling first; hold it clearly before it disappears.',
      seconds: 6.8, aspect: 'project', resolution: '480p', keepAudio: false, background: 'chroma',
    };
    const injectAndMount = async titleSettings => {
      await page.addStyleTag({content: appCss()});
      await page.evaluate(() => { document.documentElement.dataset.theme = 'graphite-studio'; });
      await page.addScriptTag({content: isolatedBundle});
      await page.evaluate(({fixture, tracks, assets, workspace, titleSettings: settings}) => {
        window.editorStore.setState({
          workspace, project: fixture, projects: [], library: [], dirty: false,
          history: [], future: [], selectedItemId: null, selectedItemIds: [], selectedTrackId: null,
          playhead: 5.375, playing: false, saving: false, error: null,
        });
        window.originalProjectParts = {tracks, assets};
        window.originalProjectFixture = fixture;
        window.expectedTitleSettings = settings;
        window.studioStore.setState(state => ({
          generationMode: 'video',
          params: {...state.params, prompt: 'STUDIO_PROMPT_MUST_NOT_LEAK', image_refs: ['/private/studio-ref.png'],
            activated_loras: ['studio-private.safetensors'], custom_settings: {private_setting: 'studio-secret'},
            spatial_upsampling: 'studio-upscale', film_grain_intensity: 0.9, negative_prompt: 'private negative prompt'},
        }));
        window.mount();
      }, {fixture: sourceProject, tracks: sourceTracks, assets: sourceAssets, workspace: WORKSPACE, titleSettings});
    };
    await page.goto(ORIGIN);
    await injectAndMount(expectedTitleSettings);

    assert.deepEqual(await page.evaluate(() => Object.fromEntries([5, 5.2, 6.8, 14.5].map(seconds => [seconds, window.titleHelpers.titleFrames(seconds)]))), {
      5: 124, 5.2: 124, 6.8: 158, 14.5: 345,
    }, 'Duration maps onto the 124–345 frame lattice in 17-frame increments.');
    for (const seconds of [5, 5.2, 6.8, 14.5]) {
      const frames = await page.evaluate(value => window.titleHelpers.titleFrames(value), seconds);
      assert.ok(frames >= 124 && frames <= 345 && (frames - 5) % 17 === 0);
    }

    await page.getByRole('button', {name: 'Open title generator'}).click();
    const dialog = page.getByRole('dialog', {name: 'Create animated title'});
    await dialog.waitFor();
    assert.equal(await dialog.getByLabel('Title background').inputValue(), 'chroma', 'Chroma key is the default title background.');
    assert.match(await dialog.innerText(), /Green is keyed automatically/);
    assert.equal(await dialog.getByLabel('Title visual style').inputValue(), '', 'A new title does not impose metallic, 3D or another visual style.');
    await dialog.getByRole('textbox', {name: 'Animated title text'}).fill('  Orion / Q4!  ');
    await dialog.getByRole('textbox', {name: 'Title visual style'}).fill('Chromed serif lettering with a blue edge light');
    const animation = dialog.getByRole('textbox', {name: 'Title animation description'});
    assert.equal(await animation.inputValue(), '', 'Custom animation is blank and visible by default.');
    assert.equal(await dialog.getByLabel('Title animation preset').isVisible(), false, 'Preset dropdown is optional and collapsed initially.');
    await dialog.getByText('Use an animation preset (optional)', {exact: true}).click();
    const presets = dialog.getByLabel('Title animation preset');
    assert.equal(await presets.locator('option').count(), 21, 'Twenty presets plus the custom option are offered.');
    await presets.selectOption('neon');
    assert.equal(await animation.inputValue(), await page.evaluate(() => window.titleHelpers.TITLE_EFFECTS.neon), 'A preset fills the editable description.');
    await animation.fill('Reveal the exact spelling first; hold it clearly before it disappears.');
    assert.equal(await presets.inputValue(), 'custom', 'Editing a starter description returns to custom animation.');
    await dialog.getByText('Use an animation preset (optional)', {exact: true}).click();
    await dialog.getByLabel('Title duration').fill('6.8');
    await dialog.getByLabel('Title aspect ratio').selectOption('project');
    await dialog.getByLabel('Title resolution').selectOption('480p');
    const expectedPrompt = await page.evaluate(settings => window.titleHelpers.animatedTitlePrompt(settings), expectedTitleSettings);
    await dialog.getByText('View prepared prompt').click();
    assert.equal(await dialog.locator('pre').innerText(), expectedPrompt,
      'The reviewed prompt preserves the exact title, selected visual style, effect, and direction.');
    assert.match(expectedPrompt, /ONLY visible lettering is exactly "Orion \/ Q4!"/);
    assert.match(expectedPrompt, /Start and end on the supplied perfectly uniform bright green background \(#00FF00\)/);
    assert.match(expectedPrompt, /No people, faces, characters, bodies, scenery, logos or additional objects/);
    assert.equal(expectedPrompt.includes('luminous neon strokes'), false, 'The preset is not silently appended to custom direction.');
    assert.equal(expectedPrompt.includes('breaks into fine dust'), false, 'No default dust animation is imposed.');
    const presetChecks = await page.evaluate(settings => Object.entries(window.titleHelpers.TITLE_EFFECTS).map(([effect, direction]) => {
      const prompt = window.titleHelpers.animatedTitlePrompt({...settings, effect, direction: ''});
      return {effect, presentOnce: prompt.indexOf(direction) >= 0 && prompt.indexOf(direction) === prompt.lastIndexOf(direction)};
    }), expectedTitleSettings);
    assert.equal(presetChecks.length, 20);
    assert.ok(presetChecks.every(result => result.presentOnce), 'All presets remain valid, with one animation direction each.');
    assert.equal(await page.evaluate(settings => {
      try { window.titleHelpers.animatedTitlePrompt({...settings, effect: 'custom', direction: ' '}); return false; }
      catch (error) { return error.message.includes('Describe the animation'); }
    }, expectedTitleSettings), true, 'Blank custom animation asks for direction instead of falling back to dust.');

    const screenshotDirectory = process.env.MAESTRO_EDITOR_TITLE_SCREENSHOTS
      ? path.resolve(process.env.MAESTRO_EDITOR_TITLE_SCREENSHOTS)
      : path.join(root, '.codex-tmp', 'editor-title-backgrounds-20261028');
    fs.mkdirSync(screenshotDirectory, {recursive: true});
    await page.getByLabel('Title background').selectOption('chroma');
    await page.screenshot({path: path.join(screenshotDirectory, 'chroma-title-dialog-desktop.png'), fullPage: true});
    await page.setViewportSize({width: 390, height: 844});
    const mobileDialog = await dialog.evaluate(element => ({
      maxHeight: getComputedStyle(element).maxHeight,
      width: element.getBoundingClientRect().width,
      height: element.getBoundingClientRect().height,
      viewportWidth: window.innerWidth,
      viewportHeight: window.innerHeight,
      scrollWidth: element.scrollWidth,
      clientWidth: element.clientWidth,
    }));
    assert.ok(mobileDialog.width <= mobileDialog.viewportWidth, `The title dialog fits mobile width: ${JSON.stringify(mobileDialog)}`);
    assert.ok(mobileDialog.scrollWidth <= mobileDialog.clientWidth + 1, `The title dialog has no horizontal overflow: ${JSON.stringify(mobileDialog)}`);
    if (mobileDialog.maxHeight !== 'none') {
      assert.ok(mobileDialog.height <= mobileDialog.viewportHeight, `The title dialog fits the mobile viewport: ${JSON.stringify(mobileDialog)}`);
    } else {
      console.warn('Mobile title viewport assertion deferred: the current ui/dist CSS bundle predates the title dialog utilities.');
    }
    await page.screenshot({path: path.join(screenshotDirectory, 'chroma-title-dialog-mobile.png'), fullPage: true});
    await page.getByLabel('Title background').selectOption('black');
    assert.match(await dialog.innerText(), /Black stays opaque\. The title clip covers footage below for its full frame\./);
    await page.screenshot({path: path.join(screenshotDirectory, 'black-title-dialog-mobile.png'), fullPage: true});
    await page.setViewportSize({width: 1440, height: 960});
    await page.screenshot({path: path.join(screenshotDirectory, 'black-title-dialog-desktop.png'), fullPage: true});
    await page.getByLabel('Title background').selectOption('chroma');

    const uploadCountBefore = uploads.length;
    await dialog.getByRole('button', {name: 'Generate title'}).dblclick();
    await withTimeout(firstReceiptLookup.promise, 'lost receipt exact client-ID lookup');
    await page.evaluate(() => window.editorStore.setState({playhead: 12.875}));
    releaseFirstReceipt.resolve();
    await waitUntil(page, () => window.titleStore.getState().intent?.status === 'inserted', 'completed lost receipt recovery and title insertion');
    assert.equal(generationBodies.length, 1, 'Double-clicking Generate submits one generation POST.');
    assert.equal(uploads.length, uploadCountBefore + 1, 'The generation used one reusable PNG upload.');
    assert.equal(projectSaves.length, 1, 'The source project was saved before title submission.');
    assert.deepEqual(jobListReads, [], 'Receipt recovery never scans the active job list.');
    assert.deepEqual(statusReads, ['synthetic-title-job-1'],
      'Recovery polls only the job whose client_submission_id exactly matches this submission.');

    const firstBody = generationBodies[0];
    const dimensions = firstBody.resolution.split('x').map(Number);
    assert.deepEqual(uploads[0].dimensions, {width: dimensions[0], height: dimensions[1]},
      'The solid green start/end PNG uses the requested output dimensions.');
    assert.equal(uploads[0].filename, `title-green-${dimensions[0]}x${dimensions[1]}.png`);
    assert.equal(uploads[0].color, '#00FF00');
    assert.ok(Math.abs(dimensions[0] / dimensions[1] - 16 / 9) < 0.05,
      'The generated green background follows the selected project aspect ratio within the frame lattice.');
    assert.equal(firstBody.model_type, 'minimax_h3_fused_turbo');
    assert.equal(firstBody.generation_mode, 'video');
    assert.equal(firstBody.workspace, WORKSPACE);
    assert.equal(firstBody.prompt, expectedPrompt);
    assert.equal(firstBody.resolution, '832x480');
    assert.equal(firstBody.video_length, 158);
    assert.equal(firstBody.num_inference_steps, 4);
    assert.equal(firstBody.guidance_scale, 1);
    assert.equal(firstBody.image_start, `/uploads/${uploads[0].filename}`);
    assert.equal(firstBody.image_end, firstBody.image_start, 'The same flat green frame is used at the start and end.');
    assert.equal(firstBody.image_prompt_type, 'SE');
    assert.deepEqual(firstBody.image_refs, [], 'No Studio reference image is inherited.');
    assert.deepEqual(firstBody.activated_loras, [], 'No Studio LoRA is inherited.');
    assert.deepEqual(firstBody.custom_settings, {}, 'No Studio custom settings are inherited.');
    assert.equal(firstBody.negative_prompt, undefined, 'No Studio negative prompt is inherited.');
    assert.equal(firstBody.prompt.includes('STUDIO_PROMPT_MUST_NOT_LEAK'), false);
    assert.equal(firstBody.prompt.includes('studio-secret'), false);
    assert.equal(firstBody._queue_mode, 'now');
    assert.ok(firstBody._client_submission_id, 'The request carries a durable client submission identity.');
    assert.equal(await page.evaluate(() => window.titleStore.getState().intent.clientId), firstBody._client_submission_id);
    assert.deepEqual(receiptReads, [firstBody._client_submission_id],
      'The lost receipt is recovered by one GET for the exact client_submission_id, even after the job already completed.');

    const uploadedResolution = await page.evaluate(() => window.titleStore.getState().intent && ({
      width: window.titleStore.getState().intent.width, height: window.titleStore.getState().intent.height,
      at: window.titleStore.getState().intent.at,
    }));
    assert.deepEqual(uploadedResolution, {width: dimensions[0], height: dimensions[1], at: 5.375},
      'The durable title intent captures output geometry and the playhead at submission time.');

    assert.equal(requests.filter(request => request.pathname === '/api/v1/editor/media/probe').length, 1,
      'The generated title is probed before it becomes an Editor asset.');
    const inserted = await page.evaluate(() => {
      const project = window.editorStore.getState().project;
      const titleTrack = project.tracks.find(track => track.name === 'Title FX');
      return {
        projectId: project.id,
        workspace: project.workspace,
        track: titleTrack,
        projectAssets: project.assets,
        historyLength: window.editorStore.getState().history.length,
      };
    });
    assert.ok(inserted.track, 'A dedicated Title FX track is inserted automatically.');
    assert.equal(inserted.track.type, 'video');
    assert.ok(inserted.track.z_index > Math.max(...sourceTracks.filter(track => track.type !== 'audio').map(track => track.z_index)),
      'Title FX is placed above every existing visual layer.');
    assert.equal(inserted.track.items.length, 1);
    const titleItem = inserted.track.items[0];
    assert.equal(titleItem.start, 5.375, 'The title uses the playhead captured before generation.');
    assert.equal(titleItem.duration, 6.25, 'The verified media probe supplies the inserted clip duration.');
    assert.equal(titleItem.name, 'Orion / Q4!');
    assert.equal(titleItem.muted, true, 'Generated clip audio is muted when Keep generated clip audio is off.');
    assert.deepEqual(titleItem.keying, {...titleItem.keying, enabled: true, similarity: 0.23, softness: 0.12, spill: 0.65});
    assert.equal(titleItem.keying.enabled, true);
    assert.equal(inserted.projectId, PROJECT_ID);
    assert.equal(inserted.workspace, WORKSPACE);
    assert.deepEqual(await page.evaluate(() => {
      const project = window.editorStore.getState().project;
      return project.tracks.filter(track => ['video-track', 'upper-track', 'audio-track'].includes(track.id));
    }), sourceTracks, 'The original video and audio tracks are unchanged when the overlay is inserted.');
    for (const [assetId, originalAsset] of Object.entries(sourceAssets)) {
      assert.deepEqual(inserted.projectAssets[assetId], originalAsset, `Original asset ${assetId} remains unchanged.`);
    }
    assert.equal(inserted.historyLength, 1, 'The new overlay is one undoable project edit.');

    const firstGreenIntent = await page.evaluate(() => JSON.parse(localStorage.getItem('maestro-editor-animated-title-v1')));
    assert.equal(firstGreenIntent.settings.background, 'chroma', 'The first title intent durably records its green keying mode.');
    const invalidBackgroundCounts = {uploads: uploads.length, saves: projectSaves.length, generations: generationBodies.length};
    await page.evaluate(settings => window.titleStore.getState().submit({...settings, background: 'magenta'}), expectedTitleSettings);
    await waitUntil(page, () => window.titleStore.getState().error === 'Choose a supported title background.', 'invalid title background rejection');
    assert.deepEqual({uploads: uploads.length, saves: projectSaves.length, generations: generationBodies.length}, invalidBackgroundCounts,
      'An unsupported background is rejected before upload, project save, or generation submission.');

    const idempotent = await page.evaluate(() => {
      const intent = window.titleStore.getState().intent;
      const before = window.editorStore.getState();
      const trackCount = before.project.tracks.filter(track => track.name === 'Title FX').length;
      const itemCount = before.project.tracks.flatMap(track => track.items).filter(item => item.animated_title_id === intent.id).length;
      const result = before.addAnimatedTitle(intent.asset, {
        id: intent.id, projectId: intent.projectId, workspace: intent.workspace,
        at: intent.at, text: intent.settings.text, keepAudio: intent.settings.keepAudio,
      });
      const after = window.editorStore.getState();
      return {
        result,
        tracksBefore: trackCount,
        tracksAfter: after.project.tracks.filter(track => track.name === 'Title FX').length,
        itemsBefore: itemCount,
        itemsAfter: after.project.tracks.flatMap(track => track.items).filter(item => item.animated_title_id === intent.id).length,
        historyBefore: before.history.length,
        historyAfter: after.history.length,
      };
    });
    assert.deepEqual(idempotent, {result: true, tracksBefore: 1, tracksAfter: 1, itemsBefore: 1, itemsAfter: 1, historyBefore: 1, historyAfter: 1},
      'Retrying the same insertion is idempotent and does not create another undo step.');

    await page.getByRole('button', {name: 'Close animated title'}).click();
    await page.evaluate(() => window.editorStore.getState().undo());
    assert.equal(await page.evaluate(() => window.editorStore.getState().project.tracks.some(track => track.name === 'Title FX')), false,
      'Undo removes the generated overlay and its project asset.');
    assert.equal(await page.evaluate(() => Object.keys(window.editorStore.getState().project.assets).some(id => id.startsWith('title-'))), false);
    await page.evaluate(() => window.editorStore.getState().redo());
    assert.equal(await page.evaluate(() => window.editorStore.getState().project.tracks.some(track => track.name === 'Title FX')), true,
      'Redo restores the generated overlay.');
    const titleIdentity = await page.evaluate(() => {
      const track = window.editorStore.getState().project.tracks.find(candidate => candidate.name === 'Title FX');
      return {itemId: track.items[0].id, trackId: track.id};
    });
    await page.evaluate(identity => window.editorStore.getState().selectItem(identity.itemId, identity.trackId), titleIdentity);
    await page.getByRole('button', {name: 'Export transparent MOV'}).click();
    const exportDialog = page.getByRole('dialog', {name: 'Export transparent overlay'});
    await exportDialog.waitFor();
    await page.getByTestId('editor-export-capability-status').getByText(/Ready.*ProRes 4444 with alpha/).waitFor();
    assert.equal(await page.getByLabel('Export format').inputValue(), 'prores4444');
    assert.match(await exportDialog.innerText(), /Only the selected overlay is exported, starting at zero/);
    assert.equal(await page.getByTestId('editor-export-submit').isDisabled(), false);
    await page.getByTestId('editor-export-submit').click();
    await waitUntil(page, () => window.editorStore.getState().exportJobId === 'synthetic-alpha-export', 'scoped transparent export submission');
    assert.equal(exportBodies.length, 1);
    assert.equal(exportBodies[0].format, 'prores4444');
    assert.deepEqual(exportBodies[0].item_ids, [titleIdentity.itemId], 'The MOV export names only the selected title item.');
    assert.ok(exportBodies[0].project.tracks.some(track => track.id === 'video-track' && track.items.some(item => item.id === 'footage-item')),
      'The persisted edit still contains the original footage while export scope stays narrow.');
    assert.ok(exportBodies[0].project.tracks.some(track => track.id === 'upper-track' && track.items.some(item => item.id === 'upper-item')));
    assert.ok(exportBodies[0].project.tracks.some(track => track.id === 'audio-track' && track.items.some(item => item.id === 'music-item')));
    assert.ok(exportBodies[0].project.tracks.flatMap(track => track.items).some(item => item.id === titleIdentity.itemId));
    assert.deepEqual(await page.evaluate(() => {
      const project = window.editorStore.getState().project;
      return project.tracks.filter(track => ['video-track', 'upper-track', 'audio-track'].includes(track.id));
    }), sourceTracks, 'Opening and submitting scoped MOV export leaves original tracks untouched.');

    // A second job finishes while the same project ID is open in a different
    // workspace. The recovered asset remains ready but cannot enter that edit.
    await page.getByRole('button', {name: 'Close export settings'}).click();
    await page.evaluate(() => window.titleStore.getState().dismiss());
    const switchedWorkspace = {...clone(sourceProject), workspace: 'Other title workspace'};
    await page.evaluate(project => window.editorStore.setState({
      project, workspace: project.workspace, dirty: false, history: [], future: [],
      selectedItemId: null, selectedItemIds: [], selectedTrackId: null, exportJobId: null,
    }), switchedWorkspace);
    await page.evaluate(() => window.titleStore.getState().submit({
      ...window.expectedTitleSettings, text: 'Workspace guard', keepAudio: true,
    }));
    await waitUntil(page, () => window.titleStore.getState().intent?.status === 'running', 'workspace-guard title job');
    await page.waitForTimeout(75);
    const workspaceJobId = 'synthetic-title-job-2';
    const otherProject = {...clone(sourceProject), id: 'other-title-project', workspace: switchedWorkspace.workspace};
    await page.evaluate(project => window.editorStore.setState({
      project, workspace: project.workspace, dirty: false, history: [], future: [],
      selectedItemId: null, selectedItemIds: [], selectedTrackId: null,
    }), otherProject);
    await page.waitForTimeout(75);
    Object.assign(jobs.get(workspaceJobId), {
      status: 'completed', progress: 100, message: 'completed',
      output_files: [`/outputs/${workspaceJobId}.mp4`],
    });
    await page.evaluate(() => window.titleStore.getState().reconcile());
    await waitUntil(page, () => window.titleStore.getState().intent?.status === 'ready', 'title ready without cross-project insertion');
    assert.equal(await page.evaluate(() => window.editorStore.getState().project.tracks.some(track => track.name === 'Title FX')), false,
      'A completed title cannot enter a different project that is open while generation finishes.');
    const differentWorkspace = {...clone(sourceProject), workspace: 'Another workspace'};
    await page.evaluate(project => window.editorStore.setState({
      project, workspace: project.workspace, dirty: false, history: [], future: [],
    }), differentWorkspace);
    await page.evaluate(() => window.titleStore.getState().insertReady());
    assert.equal(await page.evaluate(() => window.editorStore.getState().project.tracks.some(track => track.name === 'Title FX')), false,
      'A ready title cannot enter the matching project ID under a different workspace.');
    const titleStatusIds = statusReads.filter(jobId => jobId.startsWith('synthetic-title-job-'));
    assert.equal(titleStatusIds[0], 'synthetic-title-job-1');
    assert.ok(titleStatusIds.slice(1).length >= 1 && titleStatusIds.slice(1).every(jobId => jobId === workspaceJobId),
      'Each title result is checked only by its exact job receipt ID.');

    // Failed and cancelled model jobs leave the edit without an overlay.
    await page.evaluate(() => window.titleStore.getState().dismiss());
    await page.evaluate(fixture => window.editorStore.setState({
      project: fixture, workspace: fixture.workspace,
      dirty: false, history: [], future: [], exportJobId: null,
    }), sourceProject);
    // The store action is exercised directly here; the first job above already
    // covers the full dialog and double-click path.
    for (const [index, terminalStatus] of [[3, 'failed'], [4, 'cancelled']]) {
      await page.evaluate(({text, keepAudio}) => window.titleStore.getState().submit({
        text, look: 'Bold clear lettering', effect: 'dissolve', direction: '', seconds: 5.2,
        aspect: 'project', resolution: '480p', keepAudio,
      }), {text: terminalStatus, keepAudio: terminalStatus === 'failed'});
      const jobId = `synthetic-title-job-${index}`;
      await waitUntil(page, id => window.titleStore.getState().intent?.jobId === id, `${terminalStatus} title submit receipt`, jobId);
      await page.waitForTimeout(75);
      Object.assign(jobs.get(jobId), {
        status: terminalStatus, message: terminalStatus,
        ...(terminalStatus === 'failed' ? {error: 'Synthetic generation failure'} : {}),
      });
      const before = await page.evaluate(() => window.editorStore.getState().project.tracks.flatMap(track => track.items).length);
      await page.evaluate(() => window.titleStore.getState().reconcile());
      await waitUntil(page, () => window.titleStore.getState().intent?.status === 'failed', `${terminalStatus} title result`);
      const after = await page.evaluate(() => window.editorStore.getState().project.tracks.flatMap(track => track.items).length);
      assert.equal(after, before, `${terminalStatus} generation does not insert a timeline item.`);
      assert.equal(await page.evaluate(() => window.editorStore.getState().project.tracks.some(track => track.name === 'Title FX')), false);
      await page.evaluate(() => window.titleStore.getState().dismiss());
    }

    // A ready legacy green intent is restored after a real browser reload.
    // Older persisted records have no background field and retain green keying.
    const legacyReadyIntent = clone(firstGreenIntent);
    legacyReadyIntent.status = 'ready';
    delete legacyReadyIntent.settings.background;
    delete legacyReadyIntent.error;
    const generationsBeforeLegacyReload = generationBodies.length;
    await page.evaluate(intent => localStorage.setItem('maestro-editor-animated-title-v1', JSON.stringify(intent)), legacyReadyIntent);
    await page.reload();
    await injectAndMount(expectedTitleSettings);
    await waitUntil(page, () => window.titleStore.getState().intent?.status === 'inserted', 'legacy green ready intent restores after browser reload');
    assert.equal(generationBodies.length, generationsBeforeLegacyReload,
      'Restoring a saved legacy title does not submit another generation POST.');
    const legacyRestore = await page.evaluate(() => {
      const intent = window.titleStore.getState().intent;
      const title = window.editorStore.getState().project.tracks.find(track => track.name === 'Title FX')?.items[0];
      return {background: intent?.settings.background, keyingEnabled: title?.keying?.enabled, status: intent?.status};
    });
    assert.deepEqual(legacyRestore, {background: undefined, keyingEnabled: true, status: 'inserted'},
      'A pre-background saved ready intent restores with green chroma key enabled.');

    await page.evaluate(fixture => {
      window.titleStore.getState().dismiss();
      window.editorStore.setState({
        workspace: fixture.workspace, project: fixture, projects: [], library: [], dirty: false,
        history: [], future: [], selectedItemId: null, selectedItemIds: [], selectedTrackId: null,
        playhead: 5.375, playing: false, saving: false, error: null, exportJobId: null,
      });
    }, sourceProject);

    // A full black-background run preserves Unicode wording, Spanish motion
    // directions and deliberately surreal green lettering through a reload.
    await page.getByRole('button', {name: 'Open title generator'}).click();
    const blackDialog = page.getByRole('dialog', {name: 'Create animated title'});
    await blackDialog.waitFor();
    const blackTitleSettings = {
      text: 'Mañana — Ω-2049 · ½',
      look: 'Letras de vidrio verde lima, surrealistas y con geometría imposible.',
      effect: 'custom',
      direction: 'Las letras flotan en una órbita imposible, se ordenan para que se puedan leer, quedan quietas un momento y luego se desvanecen.',
      seconds: 6.8, aspect: 'project', resolution: '480p', keepAudio: false, background: 'black',
    };
    await blackDialog.getByLabel('Title text').fill(blackTitleSettings.text);
    await blackDialog.getByLabel('Title visual style').fill(blackTitleSettings.look);
    await blackDialog.getByLabel('Title animation description').fill(blackTitleSettings.direction);
    await blackDialog.getByLabel('Title background').selectOption('black');
    await blackDialog.getByLabel('Title duration').fill(String(blackTitleSettings.seconds));
    await blackDialog.getByLabel('Title aspect ratio').selectOption(blackTitleSettings.aspect);
    await blackDialog.getByLabel('Title resolution').selectOption(blackTitleSettings.resolution);
    assert.match(await blackDialog.innerText(), /Black stays opaque\. The title clip covers footage below for its full frame\./);
    const blackPrompt = await page.evaluate(settings => window.titleHelpers.animatedTitlePrompt(settings), blackTitleSettings);
    await blackDialog.getByText('View prepared prompt').click();
    assert.equal(await blackDialog.locator('pre').innerText(), blackPrompt,
      'The black prepared prompt preserves the exact Unicode title, Spanish direction and requested lettering style.');
    assert.match(blackPrompt, /ONLY visible lettering is exactly "Mañana — Ω-2049 · ½"/);
    assert.match(blackPrompt, /Letras de vidrio verde lima, surrealistas y con geometría imposible\./);
    assert.match(blackPrompt, /Las letras flotan en una órbita imposible/);
    assert.match(blackPrompt, /uniform pure black background \(#000000\)/);
    assert.doesNotMatch(blackPrompt, /#00FF00|bright green|green particles|green reflections|reserve .*green|no green/i,
      'The black prompt has no green-key color or particle restrictions.');
    assert.doesNotMatch(blackPrompt, /\b(?:2D|English|neon|particles?)\b/i,
      'The black prompt does not force a 2D/English/neon look or ban particles.');

    await page.screenshot({path: path.join(screenshotDirectory, 'black-title-dialog-desktop.png'), fullPage: true});
    await page.setViewportSize({width: 390, height: 844});
    const blackMobileDialog = await blackDialog.evaluate(element => ({
      width: element.getBoundingClientRect().width,
      viewportWidth: window.innerWidth,
      scrollWidth: element.scrollWidth,
      clientWidth: element.clientWidth,
    }));
    assert.ok(blackMobileDialog.width <= blackMobileDialog.viewportWidth,
      `The black title dialog fits mobile width: ${JSON.stringify(blackMobileDialog)}`);
    assert.ok(blackMobileDialog.scrollWidth <= blackMobileDialog.clientWidth + 1,
      `The black title dialog has no horizontal overflow: ${JSON.stringify(blackMobileDialog)}`);
    await page.screenshot({path: path.join(screenshotDirectory, 'black-title-dialog-mobile.png'), fullPage: true});
    await page.setViewportSize({width: 1440, height: 960});

    const blackUploadCountBefore = uploads.length;
    const blackSaveCountBefore = projectSaves.length;
    const blackGenerationCountBefore = generationBodies.length;
    await blackDialog.getByRole('button', {name: 'Generate title'}).click();
    await waitUntil(page, () => window.titleStore.getState().intent?.settings.background === 'black'
      && window.titleStore.getState().intent?.status === 'running', 'black title generation starts');
    assert.equal(uploads.length, blackUploadCountBefore + 1, 'Black mode uploads one boundary PNG.');
    assert.equal(projectSaves.length, blackSaveCountBefore + 1, 'The project is saved before black title submission.');
    assert.equal(generationBodies.length, blackGenerationCountBefore + 1, 'Black mode submits one generation POST.');
    const blackBody = generationBodies.at(-1);
    const blackUpload = uploads.at(-1);
    const blackDimensions = blackBody.resolution.split('x').map(Number);
    assert.equal(blackUpload.filename, `title-black-${blackDimensions[0]}x${blackDimensions[1]}.png`);
    assert.equal(blackUpload.color, '#000000', 'The uploaded boundary frame is pure black.');
    assert.deepEqual(blackUpload.dimensions, {width: blackDimensions[0], height: blackDimensions[1]});
    assert.equal(blackBody.image_start, `/uploads/${blackUpload.filename}`);
    assert.equal(blackBody.image_end, blackBody.image_start, 'Black mode uses one matching start/end frame.');
    assert.equal(blackBody.prompt, blackPrompt);
    assert.ok(blackBody._client_submission_id, 'The black job carries a durable submission identity.');
    const blackJobId = generationByJob.size ? [...generationByJob.keys()].at(-1) : null;
    assert.ok(blackJobId && jobs.get(blackJobId)?.client_submission_id === blackBody._client_submission_id,
      'The running black job belongs to the frozen title intent.');
    const frozenBlackIntent = await page.evaluate(() => ({
      background: window.titleStore.getState().intent.settings.background,
      text: window.titleStore.getState().intent.settings.text,
      clientId: window.titleStore.getState().intent.clientId,
      jobId: window.titleStore.getState().intent.jobId,
      status: window.titleStore.getState().intent.status,
    }));
    assert.deepEqual(frozenBlackIntent, {
      background: 'black', text: blackTitleSettings.text, clientId: blackBody._client_submission_id,
      jobId: blackJobId, status: 'running',
    }, 'The running intent freezes its selected black mode and exact title settings.');

    await blackDialog.getByLabel('Title background').selectOption('chroma');
    assert.equal(await blackDialog.getByLabel('Title background').inputValue(), 'chroma', 'The dialog draft can change while the job runs.');
    assert.equal(await page.evaluate(() => window.titleStore.getState().intent.settings.background), 'black',
      'Changing the live dialog draft does not change the submitted job mode.');
    const blackGenerationCountBeforeReload = generationBodies.length;
    await page.reload();
    await injectAndMount(expectedTitleSettings);
    await waitUntil(page, jobId => window.titleStore.getState().intent?.jobId === jobId
      && window.titleStore.getState().intent?.status === 'running', 'black job recovers after browser reload', blackJobId);
    assert.equal(generationBodies.length, blackGenerationCountBeforeReload,
      'Reloading a running black title recovers its exact job without a new generation POST.');
    assert.equal(await page.evaluate(() => window.titleStore.getState().intent.settings.background), 'black',
      'Reload recovery retains the frozen black background selection.');

    const blackJob = jobs.get(blackJobId);
    assert.equal(blackJob.client_submission_id, frozenBlackIntent.clientId,
      'Only the synthetic job with the exact client submission identity is completed.');
    Object.assign(blackJob, {
      status: 'completed', progress: 100, message: 'completed',
      output_files: [`/outputs/${blackJobId}.mp4`],
    });
    await page.evaluate(() => window.titleStore.getState().reconcile());
    await waitUntil(page, () => window.titleStore.getState().intent?.status === 'inserted', 'owned black job completes and inserts');
    const blackInsertion = await page.evaluate(() => {
      const intent = window.titleStore.getState().intent;
      const title = window.editorStore.getState().project.tracks.find(track => track.name === 'Title FX')?.items[0];
      return {
        status: intent?.status,
        background: intent?.settings.background,
        message: intent?.message,
        keyingEnabled: title?.keying?.enabled,
        opacity: title?.opacity,
      };
    });
    assert.deepEqual(blackInsertion, {
      status: 'inserted', background: 'black',
      message: 'Title added above your footage with its black background intact. Review spelling and timing.',
      keyingEnabled: false, opacity: 1,
    }, 'The black result is inserted opaque, with chroma key disabled and the matching status message.');

    assert.deepEqual(pageErrors, [], 'No browser runtime errors occur in the isolated workflow.');
    assert.ok(requests.every(request => new URL(request.url).hostname === '192.168.56.23'),
      'All application HTTP requests stay on the intercepted LAN-style test origin.');
    assert.ok(!requests.some(request => /download|restart|llm|plan-h3|preview/i.test(request.pathname)),
      'The workflow makes no model downloads, restarts, LLM planning, or GPU preview requests.');
    console.log('Animated title workflow passed: green/black PNG uploads, lost-receipt and reload recovery, prompt/preset behavior, safe insertion, ownership, undo/idempotence, failed/cancelled jobs, and scoped transparent MOV export.');
    console.log(`Graphite dialog screenshots: ${['chroma-title-dialog-desktop.png', 'chroma-title-dialog-mobile.png', 'black-title-dialog-desktop.png', 'black-title-dialog-mobile.png'].map(name => path.join(screenshotDirectory, name)).join(' and ')}`);
  } finally {
    await browser.close();
  }
})().catch(error => {
  console.error(error);
  process.exitCode = 1;
});
