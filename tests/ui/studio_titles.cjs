// Isolated browser regression for Video → Create → Titles in the real Studio
// sidebar. Every HTTP request is intercepted; no model or GPU work can run.
// Run: node tests/ui/studio_titles.cjs
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const {createRequire} = require('node:module');

const root = path.resolve(__dirname, '../..');
const ui = path.join(root, 'ui');
const requireFromUi = createRequire(path.join(ui, 'package.json'));
const esbuild = requireFromUi('esbuild');
const playwrightPath = process.env.MAESTRO_PLAYWRIGHT ||
  'C:/Users/bliza/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright-core';
const {chromium} = require(playwrightPath);

const ORIGIN = 'http://studio-titles.test';
const WORKSPACE = 'Studio Titles Workspace';
const TITLE_MODEL = 'minimax_h3_fused_turbo';
const SCREENSHOTS = process.env.MAESTRO_STUDIO_TITLES_SCREENSHOTS ||
  path.join(root, '.codex-tmp/studio-titles-ui');
const families = [
  {id: 'minimax_h3', label: 'MiniMax H3', order: 1},
  {id: 'viggle', label: 'Viggle', order: 2},
  {id: 'longcat', label: 'LongCat', order: 3},
  {id: 'synthetic_video', label: 'Other Frames models', order: 4},
];
const models = [
  {model_type: 'test_frames_text', name: 'Test Frames T2V', family: 'synthetic_video', architecture: 'wan',
    is_t2v: true, is_i2v: false, supports_end_frame: false, guidance_max_phases: 1, fps: 16, is_downloaded: true},
  {model_type: 'test_frames_image', name: 'Test Frames I2V', family: 'synthetic_video', architecture: 'wan',
    is_t2v: false, is_i2v: true, supports_end_frame: false, guidance_max_phases: 1, fps: 24, is_downloaded: true},
  {
    model_type: TITLE_MODEL, name: 'H3 Fused 4-Step · Frames', family: 'minimax_h3',
    architecture: 'minimax_h3', is_i2v: true, is_t2v: true, supports_end_frame: true,
    supports_audio_input: true, guidance_max_phases: 1, fps: 24, is_downloaded: true,
  },
  {
    model_type: 'viggle_animate', name: 'Viggle Animate', family: 'viggle',
    architecture: 'viggle_animate', is_i2v: true, is_t2v: false,
    guidance_max_phases: 1, fps: 30, is_downloaded: true,
  },
  {
    model_type: 'longcat_avatar', name: 'LongCat Avatar', family: 'longcat',
    architecture: 'longcat_avatar', is_i2v: true, is_t2v: false,
    supports_audio_input: true, guidance_max_phases: 1, fps: 24, is_downloaded: true,
  },
  {
    model_type: 'longcat_avatar_multi', name: 'LongCat Avatar Multi', family: 'longcat',
    architecture: 'longcat_avatar', is_i2v: true, is_t2v: false,
    supports_audio_input: true, guidance_max_phases: 1, fps: 24, is_downloaded: true,
  },
];
const catalogue = {families, models};
const modelOptions = {
  t2v_class: 'text', i2v_class: 'image', image_ref_choices: ['I'],
  video_prompt_type_sources: [], image_prompt_type_sources: [],
  resolution_presets: {}, default_num_inference_steps: 4,
};
const stats = {
  gpu: {available: true, name: 'Synthetic GPU', percent: 0, vram_used_gb: 1, vram_total_gb: 24},
  ram: {used_gb: 2, total_gb: 32, percent: 6}, cpu: {percent: 1}, model: {loaded: false},
};

function appCss() {
  const html = fs.readFileSync(path.join(ui, 'dist/index.html'), 'utf8');
  const filename = /href="\/assets\/([^"]+\.css)"/.exec(html)?.[1];
  assert.ok(filename, 'Build the UI before capturing Studio Titles screenshots.');
  const file = path.join(ui, 'dist/assets', filename);
  assert.ok(fs.existsSync(file), `Built stylesheet exists: ${file}`);
  return fs.readFileSync(file, 'utf8');
}

async function bundleUi() {
  const extensions = ['.tsx', '.ts', '.jsx', '.js', '.mjs', '.cjs', '.json', '.svg'];
  const resolveFile = candidate => {
    if (fs.existsSync(candidate) && fs.statSync(candidate).isFile()) return fs.realpathSync(candidate);
    for (const extension of extensions) {
      const withExtension = candidate + extension;
      if (fs.existsSync(withExtension) && fs.statSync(withExtension).isFile()) return fs.realpathSync(withExtension);
      const indexFile = path.join(candidate, `index${extension}`);
      if (fs.existsSync(indexFile) && fs.statSync(indexFile).isFile()) return fs.realpathSync(indexFile);
    }
    return null;
  };
  const resolver = {
    name: 'studio-titles-test-workspace-resolver',
    setup(build) {
      build.onResolve({filter: /.*/}, args => {
        if (args.path.startsWith('node:')) return {path: args.path, external: true};
        if (args.path.startsWith('.') || path.isAbsolute(args.path)) {
          const resolved = resolveFile(path.resolve(args.resolveDir, args.path));
          return resolved ? {path: resolved, namespace: 'workspace-fs'}
            : {errors: [{text: `Unable to resolve ${args.path} from ${args.resolveDir}`}]};
        }
        return {
          path: fs.realpathSync(requireFromUi.resolve(args.path, {paths: [args.resolveDir, ui]})),
          namespace: 'workspace-fs',
        };
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
        import {createRoot} from 'react-dom/client';
        import {useStore} from './src/stores/useStore';
        import {useEditorStore} from './src/editor/useEditorStore';
        import {useEditorTitleStore} from './src/editor/useEditorTitleStore';
        import {Sidebar} from './src/components/Sidebar/Sidebar';
        import {TITLE_EFFECT_LABELS, TITLE_SEQUENCE_DURATIONS, alignTitleSequence, setTitlePhraseSeconds, titleDimensions, titleFrames, titleSequenceTimeline, newTitlePhrases, animatedTitlePrompt, animatedTitleParams, insertAnimatedTitle} from './src/editor/editorAnimatedTitles';
        window.store = useStore;
        window.editorStore = useEditorStore;
        window.titleStore = useEditorTitleStore;
        window.titleHelpers = {TITLE_EFFECT_LABELS, TITLE_SEQUENCE_DURATIONS, alignTitleSequence, setTitlePhraseSeconds, titleDimensions, titleFrames, titleSequenceTimeline, newTitlePhrases, animatedTitlePrompt, animatedTitleParams, insertAnimatedTitle};
        window.baseParams = {...useStore.getState().params};
        window.seedScenario = (scenario = {}) => {
          const model = scenario.model || '${TITLE_MODEL}';
          const workflow = scenario.workflow || 'frames';
          const imageMode = scenario.imageMode ?? (workflow === 'blend' ? 4 : 0);
          const prompt = scenario.prompt || 'Frames prompt stays in Studio.';
          const refName = scenario.refName || 'frames-reference.png';
          const audioGuide = scenario.audioGuide;
          const audioGuide2 = scenario.audioGuide2;
          const refPath = '/uploads/' + refName;
          const reference = new File(['synthetic image bytes'], refName, {type: 'image/png'});
          useStore.setState(state => ({
            modelsLoaded: true, modelsLoading: false, modelsLoadError: null,
            models: ${JSON.stringify(models)}, families: ${JSON.stringify(families)},
            enabledModels: new Set(${JSON.stringify(models.map(model => model.model_type))}),
            modelVisibilityOverrides: {}, modelOptions: ${JSON.stringify(modelOptions)},
            selectedModelPerMode: {...state.selectedModelPerMode, video: model},
            generationMode: 'video', sidebarMode: 'studio', sidebarOpen: true,
            servicesConfig: {worlds_enabled: true, nsfw_mode: false},
            studioVideoWorkflow: workflow, studioImageWorkflow: 'generate',
            studioVideoCreateRoute: 'auto', studioVideoEffectiveCreateRoute: 'generate',
            videoSubModeStash: {}, activeWorkspace: scenario.workspace || '${WORKSPACE}', browsingAllFolders: false,
            browsingUploads: false, imageRefs: [reference], imageRefType: 'image',
            audioGuideFilename: scenario.audioGuideFilename || null,
            audioGuide2Filename: scenario.audioGuide2Filename || null,
            startImage: null, endImage: null, continueVideo: null, continueVideoPath: '',
            continueVideoUrl: '', continueVideoDuration: 0, jobs: [], outputs: [],
            outputsTotal: 0, outputsLoading: false, galleryInitializing: false,
            isGenerating: false, isEnhancing: false,
            params: {...window.baseParams, ...state.params, model_type: model, prompt,
              image_mode: imageMode, image_refs: [refPath], video_prompt_type: '',
              audio_prompt_type: scenario.audioPromptType || '',
              audio_guide: audioGuide, audio_guide2: audioGuide2,
              speakers_locations: scenario.speakersLocations,
              image_prompt_type: 'I', _studio_video_workflow: workflow,
              activated_loras: [], loras_multipliers: ''},
          }));
          useEditorStore.setState({project: null, playhead: 0});
        };
        window.setScenarioInputs = (prompt, refName) => {
          const state = useStore.getState();
          const reference = new File(['synthetic image bytes'], refName, {type: 'image/png'});
          useStore.setState({
            imageRefs: [reference], imageRefType: 'image',
            params: {...state.params, prompt, image_refs: ['/uploads/' + refName]},
          });
        };
        window.mount = () => {
          document.documentElement.dataset.theme = 'graphite-studio';
          window.reactRoot = createRoot(document.getElementById('root'));
          window.reactRoot.render(<Sidebar onOpenWorlds={() => {window.worldsOpened = (window.worldsOpened || 0) + 1;}}/>);
        };
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

function jobStatus(jobId, submissionId, status, outputFiles = []) {
  return {
    job_id: jobId, client_submission_id: submissionId, status,
    progress: status === 'completed' ? 100 : status === 'running' ? 25 : 0,
    step: status === 'completed' ? 4 : 1, total_steps: 4,
    phase: status, message: `Synthetic ${status}`, output_files: outputFiles,
    error: status === 'failed' ? 'Synthetic generation failure'
      : status === 'cancelled' ? 'Synthetic cancellation' : null,
    show_in_gallery: true, kind: 'generation',
  };
}

async function waitFor(page, expression, description, timeout = 10000, argument = undefined) {
  try {
    await page.waitForFunction(expression, argument, {timeout});
  } catch (error) {
    const diagnostic = await page.evaluate(() => ({
      status: window.titleStore?.getState().intent?.status,
      error: window.titleStore?.getState().error || window.titleStore?.getState().intent?.error,
      alerts: [...document.querySelectorAll('[role="alert"]')].map(element => element.textContent),
      enhance: window.store?.getState().enhanceOnGeneration,
      enhanceDefault: window.store?.getState().enhanceOnGenerationDefault,
      draft: JSON.parse(localStorage.getItem('maestro-studio-title-draft-v1') || 'null'),
    }));
    throw new Error(`Timed out waiting for ${description}: ${error.message}\n${JSON.stringify(diagnostic)}`);
  }
}

async function waitForNode(predicate, description, timeout = 10000) {
  const started = Date.now();
  while (!predicate()) {
    if (Date.now() - started > timeout) throw new Error(`Timed out waiting for ${description}.`);
    await new Promise(resolve => setTimeout(resolve, 20));
  }
}

function deferred() {
  let resolve;
  const promise = new Promise(done => { resolve = done; });
  return {promise, resolve};
}

async function mount(page, bundle, css, scenario = {}) {
  await page.addStyleTag({content: css});
  await page.addScriptTag({content: bundle});
  await page.evaluate(scenario => { window.seedScenario(scenario); window.mount(); }, scenario);
  await page.locator('.maestro-sidebar').waitFor();
}

async function chooseWorkflow(page, name) {
  const sidebar = page.locator('.maestro-sidebar');
  await sidebar.locator('.studio-workflow-header button[title]').click();
  const dialog = page.getByRole('dialog', {name: 'Choose a workflow'});
  await dialog.waitFor();
  await dialog.getByRole('button', {name: new RegExp(`^${name}(?:\\s|$)`)}).click();
  const value = name === 'Animate' ? 'animate' : name.toLowerCase();
  await waitFor(page, value => window.store.getState().studioVideoWorkflow === value,
    `workflow ${name}`, 10000, value);
}

async function stateInputs(page) {
  return page.evaluate(() => {
    const state = window.store.getState();
    return {
      model: state.params.model_type,
      prompt: state.params.prompt,
      imageRefs: state.params.image_refs,
      referenceFiles: state.imageRefs.map(file => file.name),
    };
  });
}

async function assertTitleModeHasOnlyItsOwnComposer(page) {
  const sidebar = page.locator('.maestro-sidebar');
  await page.getByRole('region', {name: 'Studio Titles'}).waitFor();
  assert.equal(await sidebar.locator('textarea[aria-label="Generation prompt"]').count(), 0,
    'Titles replaces the ordinary generation prompt.');
  assert.equal(await sidebar.locator('[data-studio-footer]').count(), 0,
    'Titles hides the ordinary settings/footer controls.');
  assert.equal(await sidebar.getByRole('button', {name: 'Open model browser'}).count(), 0,
    'Titles does not expose the ordinary model/LoRA browser.');
  assert.equal(await sidebar.getByText('Control video', {exact: true}).count(), 0,
    'Titles hides the ordinary control-video input.');
  assert.equal(await sidebar.getByText(/LoRA/i).count(), 0,
    'Titles does not expose ordinary LoRA controls.');
  const footerLayout = await page.evaluate(() => {
    const body = document.querySelector('[data-testid="studio-body-scroll"]');
    const host = document.querySelector('[data-testid="titles-footer-host"]');
    const footer = document.querySelector('[data-testid="titles-generate-footer"]');
    const hardware = document.querySelector('.studio-hardware');
    return {
      hostOutsideBody: !!body && !!host && !body.contains(host),
      footerInHost: !!host && !!footer && footer.parentElement === host,
      hostVisible: !!host && !host.hidden,
      modelType: document.querySelector('[aria-label="Choose model"]')?.getAttribute('data-model-type'),
      hardwareVisible: !!hardware && getComputedStyle(hardware).display !== 'none' && hardware.getBoundingClientRect().height > 0,
    };
  });
  assert.equal(footerLayout.hostOutsideBody, true, 'The Titles footer host is a sibling outside the scrolling title form.');
  assert.equal(footerLayout.footerInHost, true, 'The title generate row is portaled into its fixed footer host.');
  assert.equal(footerLayout.hostVisible, true, 'The fixed Titles footer host is shown in Titles mode.');
  assert.equal(footerLayout.modelType, TITLE_MODEL, 'The title footer identifies its fixed H3 model.');
  assert.equal(footerLayout.hardwareVisible, true, 'The global hardware monitor stays visible below the title footer.');
  const formatControls = page.getByTestId('titles-output-format-controls');
  assert.equal(await formatControls.locator('button.studio-setting-chip').count(), 4,
    'Resolution, aspect, duration and Advanced use the same compact chips as the other Studio modes.');
  assert.equal(await formatControls.evaluate(element => !!element.closest('[data-testid="titles-generate-footer"]')), true,
    'The everyday format controls remain in the fixed footer rather than the scrolling title form.');
  const promptToolbar = await page.getByTestId('title-prompt-field').evaluate(element => {
    const input = element.querySelector('textarea');
    const enhance = element.querySelector('[aria-label="Enhance title now"]');
    const options = element.querySelector('[aria-label="Title enhancement options"]');
    const field = element.getBoundingClientRect();
    const button = enhance?.getBoundingClientRect();
    return {input: !!input, enhance: !!enhance, options: !!options,
      buttonInside: !!button && button.left >= field.left && button.right <= field.right
        && button.top >= field.top && button.bottom <= field.bottom};
  });
  assert.deepEqual(promptToolbar, {input: true, enhance: true, options: true, buttonInside: true},
    'The title enhancement split button belongs to the prompt field and fits inside its bounds.');
  const hardware = sidebar.locator('.studio-hardware');
  const collapse = hardware.locator('button[title="Collapse"]');
  const expand = hardware.locator('button[title="Show hardware status"]');
  if (await collapse.isVisible()) {
    await collapse.click();
    await expand.waitFor({state: 'visible'});
    await expand.click();
  } else {
    await expand.waitFor({state: 'visible'});
    await expand.click();
    await collapse.waitFor({state: 'visible'});
    await collapse.click();
  }
  await collapse.waitFor({state: 'visible'});
  assert.equal(await page.evaluate(() => window.editorStore.getState().project), null,
    'The Studio panel renders without an Editor project.');
}

async function assertHardwareExpandable(page, label) {
  const hardware = page.locator('.studio-hardware');
  assert.equal(await hardware.isVisible(), true, `${label}: the global hardware monitor remains visible.`);
  const collapse = hardware.locator('button[title="Collapse"]');
  const expand = hardware.locator('button[title="Show hardware status"]');
  if (await collapse.isVisible()) {
    await collapse.click();
    await expand.waitFor({state: 'visible'});
    await expand.click();
  } else {
    await expand.waitFor({state: 'visible'});
    await expand.click();
    await collapse.waitFor({state: 'visible'});
    await collapse.click();
  }
  await collapse.waitFor({state: 'visible'});
}

async function acceptManualTitlePlan(panel) {
  const advanced = await openTitleAdvanced(panel.page());
  await advanced.getByRole('button', {name: 'Use manual title plan'}).click();
}

async function openTitleAdvanced(page) {
  const advanced = page.getByRole('dialog', {name: 'Advanced title settings', exact: true});
  if (!await advanced.isVisible()) await page.getByRole('button', {name: 'Advanced title settings', exact: true}).click();
  await advanced.waitFor();
  return advanced;
}

async function uploadedPixel(page, upload) {
  const body = upload.buffer;
  const start = body.indexOf(Buffer.from([137, 80, 78, 71, 13, 10, 26, 10]));
  assert.ok(start >= 0, 'The selected background was uploaded as PNG.');
  let end = start + 8;
  while (end < body.length) {
    const size = body.readUInt32BE(end);
    const type = body.toString('ascii', end + 4, end + 8);
    end += size + 12;
    if (type === 'IEND') break;
  }
  return page.evaluate(async base64 => {
    const bytes = Uint8Array.from(atob(base64), character => character.charCodeAt(0));
    const bitmap = await createImageBitmap(new Blob([bytes], {type: 'image/png'}));
    const canvas = document.createElement('canvas'); canvas.width = canvas.height = 1;
    const context = canvas.getContext('2d'); context.drawImage(bitmap, 0, 0);
    return Array.from(context.getImageData(0, 0, 1, 1).data);
  }, body.subarray(start, end).toString('base64'));
}

(async () => {
  const css = appCss();
  const bundle = await bundleUi();
  const requestLog = [];
  const generationRequests = [];
  const titlePlanRequests = [];
  const uploadRequests = [];
  const uploadFailures = [];
  const receiptLookups = [];
  const statusLookups = [];
  const probes = [];
  const preferenceUpdates = [];
  let savedEnhancementDefault = false;
  const outputRefreshes = [];
  const projectApiRequests = [];
  const receipts = new Map();
  const statuses = new Map();
  const submissionModes = [];
  let generationGate = null;
  let failNextTitleUpload = false;
  let nextSubmissionMode = {status: 'queued', loseResponse: true};
  let currentOutput = null;
  const pageErrors = [];

  fs.mkdirSync(SCREENSHOTS, {recursive: true});
  const browser = await chromium.launch({
    headless: true,
    ignoreDefaultArgs: ['--hide-scrollbars'],
    ...(process.platform === 'win32' ? {
      executablePath: process.env.MAESTRO_CHROME || 'C:/Program Files/Google/Chrome/Application/chrome.exe',
    } : {}),
  });
  try {
    const page = await browser.newPage({viewport: {width: 1360, height: 900}});
    page.on('pageerror', error => { pageErrors.push(error.message); console.error(error.stack); });
    await page.route('**/*', async route => {
      const request = route.request();
      const url = new URL(request.url());
      const endpoint = url.pathname;
      const method = request.method();
      requestLog.push({method, endpoint, search: url.search});
      const json = body => route.fulfill({contentType: 'application/json', body: JSON.stringify(body)});

      if (endpoint === '/') {
        return route.fulfill({contentType: 'text/html', body: '<meta name="viewport" content="width=device-width,initial-scale=1"><button id="outside">Outside sidebar</button><div id="root" style="height:100dvh;display:flex"></div>'});
      }
      if (endpoint === '/api/v1/models') return json(catalogue);
      if (endpoint.startsWith('/api/v1/model-options/test_frames_')) return json({...modelOptions,
        architecture: 'wan', fps: endpoint.endsWith('_text') ? 16 : 24,
        frames_minimum: 1, frames_steps: endpoint.endsWith('_text') ? 4 : 8, frames_maximum: endpoint.endsWith('_text') ? 161 : 241,
        default_num_inference_steps: endpoint.endsWith('_text') ? 12 : 8, default_guidance_scale: 2.5});
      if (endpoint.startsWith('/api/v1/model-options/')) return json(modelOptions);
      if (endpoint.startsWith('/api/v1/defaults/test_frames_')) return json({
        num_inference_steps: endpoint.endsWith('_text') ? 12 : 8, guidance_scale: 2.5, sample_solver: 'euler',
      });
      if (endpoint === '/api/v1/studio-preferences' && method === 'PUT') {
        const update = request.postDataJSON();
        preferenceUpdates.push(update);
        if (typeof update.enhance_on_generation_default === 'boolean') {
          savedEnhancementDefault = update.enhance_on_generation_default;
        }
        return json({generation_mode: 'video', studio_video_workflow: 'titles'});
      }
      if (endpoint === '/api/v1/studio-preferences') {
        return json({generation_mode: 'video', studio_video_workflow: 'frames',
          selected_model_per_mode: {video: TITLE_MODEL}, enhance_on_generation_default: savedEnhancementDefault});
      }
      if (endpoint === '/api/v1/workspaces') {
        return json({workspaces: [{name: WORKSPACE, path: '/workspaces/titles', file_count: 0}], active: WORKSPACE});
      }
      if (endpoint === '/api/v1/outputs') {
        outputRefreshes.push(url.searchParams.get('workspace'));
        return json({outputs: currentOutput ? [currentOutput] : [], total: currentOutput ? 1 : 0, next_cursor: null});
      }
      if (endpoint === '/api/v1/jobs' || endpoint === '/api/v1/jobs/active') return json({jobs: []});
      if (endpoint.startsWith('/api/v1/status/')) {
        const jobId = decodeURIComponent(endpoint.split('/').at(-1));
        statusLookups.push(jobId);
        return json(statuses.get(jobId) || jobStatus(jobId, '', 'queued'));
      }
      if (endpoint.startsWith('/api/v1/jobs/by-submission/')) {
        const clientId = decodeURIComponent(endpoint.split('/').at(-1));
        receiptLookups.push(clientId);
        const receipt = receipts.get(clientId);
        return receipt ? json(receipt) : route.fulfill({status: 404, contentType: 'application/json', body: '{"detail":"not found"}'});
      }
      if (endpoint === '/api/v1/upload') {
        const body = request.postData() || '';
        if (failNextTitleUpload) {
          failNextTitleUpload = false;
          uploadFailures.push({search: url.search, contentType: request.headers()['content-type'] || '', body});
          return route.fulfill({status: 500, contentType: 'application/json', body: '{"detail":"Synthetic title upload failure"}'});
        }
        uploadRequests.push({search: url.search, contentType: request.headers()['content-type'] || '', body, buffer: request.postDataBuffer()});
        const filename = /filename="([^"]+)"/.exec(body)?.[1] || 'title-background.png';
        return json({filename, path: `/uploads/${filename}`, url: `/uploads/${filename}`});
      }
      if (endpoint === '/api/v1/generate' && method === 'POST') {
        const body = request.postDataJSON();
        generationRequests.push(body);
        const jobId = `studio-title-job-${generationRequests.length}`;
        const clientId = body._client_submission_id;
        const mode = nextSubmissionMode;
        submissionModes.push(mode);
        nextSubmissionMode = {status: 'queued', loseResponse: false};
        const outputPath = `/outputs/${WORKSPACE}/studio-title-${generationRequests.length}.mp4`;
        const status = jobStatus(jobId, clientId, mode.status,
          mode.status === 'completed' ? [outputPath] : []);
        statuses.set(jobId, status);
        receipts.set(clientId, {job_id: jobId, client_submission_id: clientId, status: mode.status});
        if (generationGate) {
          const gate = generationGate;
          generationGate = null;
          gate.started.resolve();
          await gate.release.promise;
        }
        if (mode.loseResponse) return route.abort('failed');
        return json({job_id: jobId, status: 'queued'});
      }
      if (endpoint === '/api/v1/editor/media/probe') {
        const body = request.postDataJSON();
        probes.push(body);
        const asset = body.asset || {};
        const dimensions = /(\d+)x(\d+)/.exec(asset.path || asset.name || '')
          || /(\d+)x(\d+)/.exec(generationRequests.at(-1)?.resolution || '');
        return json({type: 'video', path: asset.path, duration: 5.2,
          width: Number(dimensions?.[1] || 480), height: Number(dimensions?.[2] || 832),
          fps: 24, has_audio: true, size: 1234});
      }
      if (endpoint.startsWith('/api/v1/editor/projects') || endpoint.startsWith('/api/v1/editor/operations')) {
        projectApiRequests.push({method, endpoint});
        return json({});
      }
      if (endpoint === '/api/v1/system-stats') return json(stats);
      if (endpoint === '/api/v1/system-config') return json({configured: true});
      if (endpoint === '/api/v1/llm/plan-title' && method === 'POST') {
        titlePlanRequests.push(request.postDataJSON());
        return json({plan: {sequence: false, phrases: [{text: 'EDITED GALLERY TITLE', animation: 'Reveal, hold clearly, then dissolve away.', weight: 1}],
          look: 'Clean silver lettering', direction: 'Reveal, hold clearly, then dissolve away.'},
          warnings: [], raw_draft: '{"plan":"review this title wording"}', review_required: true,
          draft_history_id: 'studio-title-review'});
      }
      if (endpoint.endsWith('/media-flow/capabilities')) return json({
        neural_rendering: {available: false, reason: 'Isolated browser test', scales: []},
        frame_generation: {available: false, reason: 'Isolated browser test', factors: []},
      });
      if (endpoint === '/api/v1/resolutions') return json({resolutions: []});
      if (endpoint === '/api/v1/model-visibility') return json({configured: true,
        enabled_models: models.map(model => model.model_type), mode_overrides: {}, initialized_mature_models: [], defaults_version: 0});
      if (endpoint.startsWith('/api/')) return json({jobs: [], outputs: [], items: [], presets: [], loras: [], recipes: [], downloads: [], configured: true});
      if (endpoint.endsWith('.png') || endpoint.endsWith('.svg')) {
        return route.fulfill({contentType: endpoint.endsWith('.svg') ? 'image/svg+xml' : 'image/png', body: endpoint.endsWith('.svg') ? '<svg xmlns="http://www.w3.org/2000/svg"/>' : ''});
      }
      return route.fulfill({status: 404, body: ''});
    });

    await page.goto(ORIGIN);
    await mount(page, bundle, css);
    const sidebar = page.locator('.maestro-sidebar');
    const workflowTrigger = sidebar.locator('.studio-workflow-header button[title]');

    // Titles is directly after Blend in Create and before the optional Worlds item.
    await workflowTrigger.click();
    const menu = page.getByRole('dialog', {name: 'Choose a workflow'});
    const labels = await menu.locator('button').evaluateAll(buttons => buttons
      .map(button => button.querySelector('span.min-w-0 > span')?.textContent?.trim())
      .filter(Boolean));
    const blendIndex = labels.indexOf('Blend');
    const titlesIndex = labels.indexOf('Titles');
    const worldsIndex = labels.indexOf('Worlds');
    assert.equal(titlesIndex, blendIndex + 1, `Titles follows Blend: ${labels.join(', ')}`);
    assert.equal(worldsIndex, titlesIndex + 1, `Optional Worlds follows Titles: ${labels.join(', ')}`);
    await page.keyboard.press('Escape');

    // Frame inputs remain intact when Titles is opened and when Frames is reopened.
    let before = await stateInputs(page);
    await page.evaluate(() => window.store.setState(state => ({
      params: {...state.params, _studio_title_settings: {
        text: 'Loaded from gallery', look: 'Brushed silver', effect: 'neon',
        direction: 'Reveal slowly, then hold.', seconds: 6.5, aspect: '9:16',
        resolution: '480p', background: 'black',
      }},
    })));
    await chooseWorkflow(page, 'Titles');
    await assertTitleModeHasOnlyItsOwnComposer(page);
    assert.deepEqual(await stateInputs(page), before, 'Selecting Titles preserves Frames prompt, references, and model.');
    await waitFor(page, text => document.querySelector('[aria-label="Describe your title animation"]')?.value.includes(text),
      'gallery title settings restored into the Studio composer', 10000, 'Loaded from gallery');

    // Titles uses the shared Studio enhancement preference and its own one-shot override.
    const titleEnhancementTrigger = sidebar.getByRole('button', {name: 'Title enhancement options'});
    await titleEnhancementTrigger.click();
    let titleEnhancementMenu = page.getByRole('menu', {name: 'Title enhancement options'});
    assert.equal(await titleEnhancementMenu.getByRole('menuitemcheckbox', {name: 'Enhance on generation'}).getAttribute('aria-checked'), 'false');
    assert.equal(await titleEnhancementMenu.getByRole('menuitemcheckbox', {name: 'Use by default'}).getAttribute('aria-checked'), 'false');
    await titleEnhancementMenu.getByRole('menuitemcheckbox', {name: 'Enhance on generation'}).click();
    assert.equal(await page.evaluate(() => window.store.getState().enhanceOnGeneration), true,
      'The title enhancement checkbox sets a one-shot Studio choice.');
    await titleEnhancementTrigger.click();
    titleEnhancementMenu = page.getByRole('menu', {name: 'Title enhancement options'});
    await titleEnhancementMenu.getByRole('menuitemcheckbox', {name: 'Use by default'}).click();
    await waitFor(page, () => window.store.getState().enhanceOnGenerationDefault === true,
      'shared default title enhancement preference');
    await waitForNode(() => savedEnhancementDefault === true, 'server-persisted title enhancement default');
    assert.equal(await page.evaluate(() => window.store.getState().enhanceOnGeneration), null,
      'Saving the shared default clears the temporary one-shot choice.');
    assert.equal(await page.locator('[data-testid="titles-generate-footer"]').getByRole('button', {name: 'Remove Enhance on generation'}).count(), 1,
      'The footer shows the effective enhancement choice.');

    await page.reload();
    await mount(page, bundle, css, {workflow: 'titles', workspace: WORKSPACE});
    await page.evaluate(async workspace => {
      await window.store.getState().loadModels();
      window.seedScenario({workflow: 'titles', workspace});
    }, WORKSPACE);
    await waitFor(page, () => window.store.getState().enhanceOnGenerationDefault === true,
      'shared title enhancement default after reload');
    await page.locator('.maestro-sidebar').getByRole('button', {name: 'Title enhancement options'}).click();
    titleEnhancementMenu = page.getByRole('menu', {name: 'Title enhancement options'});
    assert.equal(await titleEnhancementMenu.getByRole('menuitemcheckbox', {name: 'Use by default'}).getAttribute('aria-checked'), 'true',
      'The shared enhancement default survives reload.');
    assert.equal(await titleEnhancementMenu.getByRole('menuitemcheckbox', {name: 'Enhance on generation'}).getAttribute('aria-checked'), 'true',
      'The effective title enhancement checkbox reflects the saved default after reload.');
    await titleEnhancementMenu.getByRole('menuitemcheckbox', {name: 'Use by default'}).click();
    await waitFor(page, () => window.store.getState().enhanceOnGenerationDefault === false,
      'turning off the shared title enhancement default');
    await waitForNode(() => savedEnhancementDefault === false, 'server-persisted disabled title enhancement default');
    await titleEnhancementMenu.getByRole('menuitemcheckbox', {name: 'Enhance on generation'}).click();
    const oneShotEnhancementRevision = await page.evaluate(() => window.store.getState().enhanceOnGenerationRevision);
    assert.equal(await page.evaluate(() => window.store.getState().enhanceOnGeneration), true,
      'A one-shot title enhancement can override the saved default.');

    await page.getByLabel('Describe your title animation').fill('Edited loaded gallery title');
    await page.locator('[data-testid="titles-generate-footer"]').getByRole('button', {name: 'Generate title'}).click();
    await waitFor(page, () => window.store.getState().params._studio_title_settings?.titleReviewRequired === true
      || document.querySelector('[aria-label="Title plan review"]')?.open === true,
    'AI title plan review pause');
    assert.equal(titlePlanRequests.length, 1, 'The one-shot title choice requests one title-specific plan.');
    assert.equal(generationRequests.length, 0, 'A review-required title plan pauses before GPU submission.');
    assert.equal(await page.evaluate(() => window.titleStore.getState().intent), null,
      'A review pause creates no owned Studio title intent.');
    assert.equal(await page.evaluate(() => window.store.getState().enhanceOnGeneration), true,
      'A title plan review pause preserves the one-shot enhancement choice.');
    assert.equal(await page.evaluate(() => window.store.getState().enhanceOnGenerationRevision), oneShotEnhancementRevision,
      'A review pause does not revise or consume the one-shot choice.');
    await waitFor(page, () => window.store.getState().params._studio_title_settings === undefined,
      'one-shot gallery title settings cleared after the first edit');
    await chooseWorkflow(page, 'Frames');
    assert.deepEqual(await stateInputs(page), before, 'Returning to Frames restores its normal prompt and references.');
    assert.equal(await page.evaluate(() => window.store.getState().params._studio_title_settings), undefined,
      'Frames does not inherit one-shot title settings from its stored prompt snapshot.');
    await chooseWorkflow(page, 'Titles');
    await waitFor(page, text => document.querySelector('[aria-label="Describe your title animation"]')?.value === text,
      'edited gallery title draft after a Frames round trip', 10000, 'Edited loaded gallery title');
    assert.equal(await page.evaluate(() => window.store.getState().params._studio_title_settings), undefined,
      'Reopening Titles keeps the edited draft without restoring the old one-shot override.');
    await chooseWorkflow(page, 'Frames');

    // Two-speaker Avatar state survives detours through Titles and back to Avatar.
    // Leaving Avatar for Frames must then perform the same B-channel cleanup and
    // choose the same compatible video model as a direct Avatar → Frames switch.
    const avatarScenario = {
      workflow: 'avatar', model: 'longcat_avatar_multi',
      audioPromptType: 'AB', audioGuide: '/uploads/voice-a.wav', audioGuide2: '/uploads/voice-b.wav',
      audioGuideFilename: 'voice-a.wav', audioGuide2Filename: 'voice-b.wav',
      speakersLocations: '0:0:50:100 50:0:100:100',
    };
    await page.evaluate(scenario => window.seedScenario(scenario), avatarScenario);
    let avatarVoices = await page.evaluate(() => {
      const state = window.store.getState();
      return {model: state.params.model_type, mode: state.params.audio_prompt_type,
        voiceA: state.params.audio_guide, voiceB: state.params.audio_guide2,
        filenameA: state.audioGuideFilename, filenameB: state.audioGuide2Filename,
        speakers: state.params.speakers_locations};
    });
    assert.equal(avatarVoices.mode, 'AB');
    assert.equal(avatarVoices.voiceA, avatarScenario.audioGuide);
    assert.equal(avatarVoices.voiceB, avatarScenario.audioGuide2);
    await chooseWorkflow(page, 'Titles');
    assert.deepEqual(await page.evaluate(() => {
      const state = window.store.getState();
      return {model: state.params.model_type, mode: state.params.audio_prompt_type,
        voiceA: state.params.audio_guide, voiceB: state.params.audio_guide2,
        filenameA: state.audioGuideFilename, filenameB: state.audioGuide2Filename,
        speakers: state.params.speakers_locations};
    }), avatarVoices, 'Opening Titles preserves Avatar model and both voice channels.');
    await chooseWorkflow(page, 'Avatar');
    assert.deepEqual(await page.evaluate(() => {
      const state = window.store.getState();
      return {model: state.params.model_type, mode: state.params.audio_prompt_type,
        voiceA: state.params.audio_guide, voiceB: state.params.audio_guide2,
        filenameA: state.audioGuideFilename, filenameB: state.audioGuide2Filename,
        speakers: state.params.speakers_locations};
    }), avatarVoices, 'Returning to Avatar preserves both voice channels.');
    await chooseWorkflow(page, 'Titles');
    await chooseWorkflow(page, 'Frames');
    const titlesDetourFrames = await page.evaluate(() => ({
      model: window.store.getState().params.model_type,
      mode: window.store.getState().params.audio_prompt_type,
    }));
    const titlesDetourStashedAudioMode = await page.evaluate(() =>
      window.store.getState().videoSubModeStash[0]?.params.audio_prompt_type);
    assert.equal(titlesDetourFrames.mode, 'A', 'Leaving the Titles detour removes Avatar’s B audio channel.');
    assert.equal(titlesDetourStashedAudioMode, 'A',
      'Frames stashes the cleaned audio mode so a later submode restore cannot reintroduce B.');

    await page.evaluate(scenario => window.seedScenario(scenario), avatarScenario);
    await chooseWorkflow(page, 'Frames');
    const directAvatarFrames = await page.evaluate(() => ({
      model: window.store.getState().params.model_type,
      mode: window.store.getState().params.audio_prompt_type,
    }));
    assert.equal(directAvatarFrames.mode, 'A', 'Directly leaving two-speaker Avatar removes the B audio channel.');
    assert.equal(directAvatarFrames.model, TITLE_MODEL, 'Frames selects the compatible video model.');
    assert.deepEqual(titlesDetourFrames, directAvatarFrames,
      'Leaving Avatar through Titles applies the same voice cleanup and Frames model selection.');

    // Resume the regular Frames fixture before exercising the other workflow routes.
    await page.evaluate(() => window.seedScenario());

    // Blend has a separate input snapshot; Titles resets only image_mode and restores Blend on return.
    await chooseWorkflow(page, 'Blend');
    await page.evaluate(() => window.setScenarioInputs('Blend prompt remains intact.', 'blend-reference.png'));
    before = await stateInputs(page);
    await chooseWorkflow(page, 'Titles');
    await assertTitleModeHasOnlyItsOwnComposer(page);
    let after = await stateInputs(page);
    assert.deepEqual(after, before, 'Selecting Titles preserves Blend prompt, references, and model.');
    assert.equal(await page.evaluate(() => window.store.getState().params.image_mode), 0,
      'Titles clears image_mode=4 so it cannot route through Blend.');
    await chooseWorkflow(page, 'Blend');
    after = await stateInputs(page);
    assert.deepEqual(after, before, 'Returning to Blend restores the saved Blend inputs.');

    // Selecting Titles from Viggle does not infer or select a different model.
    await chooseWorkflow(page, 'Animate');
    await page.evaluate(() => window.setScenarioInputs('Viggle prompt remains intact.', 'viggle-reference.png'));
    before = await stateInputs(page);
    assert.equal(before.model, 'viggle_animate');
    await chooseWorkflow(page, 'Titles');
    await assertTitleModeHasOnlyItsOwnComposer(page);
    assert.deepEqual(await stateInputs(page), before, 'Titles leaves the active Viggle model and references unchanged.');
    await chooseWorkflow(page, 'Frames');
    after = await stateInputs(page);
    assert.equal(after.prompt, before.prompt);
    assert.deepEqual(after.imageRefs, before.imageRefs);
    assert.deepEqual(after.referenceFiles, before.referenceFiles);
    assert.notEqual(after.model, 'viggle_animate', 'Leaving Animate restores the regular video model.');

    // Create a portrait black title with custom-written motion. No Editor project is present.
    await chooseWorkflow(page, 'Titles');
    await assertTitleModeHasOnlyItsOwnComposer(page);
    assert.doesNotMatch(await sidebar.innerText(), /No Editor project needed|Save to/,
      'Titles omits the redundant save-location text.');
    const appearanceOrder = await page.evaluate(() => ({
      controls: document.querySelector('[data-testid="title-appearance-controls"]').getBoundingClientRect().top,
      prompt: document.querySelector('[data-testid="title-prompt-field"]').getBoundingClientRect().top,
    }));
    assert.ok(appearanceOrder.controls < appearanceOrder.prompt, 'Style and background selectors precede the prompt.');
    const ordinaryFormat = await page.evaluate(() => {
      const state = window.store.getState();
      return {resolution: state.resolutionPreset, aspect: state.aspectRatio,
        duration: state.durationSeconds, resolutionParam: state.params.resolution,
        frames: state.params.video_length};
    });
    const titleFormats = page.getByTestId('titles-output-format-controls');
    const titleCanvases = await page.evaluate(() => {
      const base = JSON.parse(localStorage.getItem('maestro-studio-title-draft-v1'));
      return [['480p', '9:16'], ['720p', '9:16'], ['540p', '4:3'], ['768p', '21:9'], ['1080p', '3:4']]
        .map(([resolution, aspect]) => window.titleHelpers.titleDimensions({...base, resolution, aspect}, {width: 1920, height: 1080}));
    });
    assert.deepEqual(titleCanvases, [{width: 480, height: 832}, {width: 704, height: 1280},
      {width: 736, height: 544}, {width: 1792, height: 768}, {width: 1088, height: 1440}],
      'Saved portrait title canvases stay compatible; added format choices use native H3 canvases.');
    assert.equal(await sidebar.locator('select[aria-label="Title resolution"]:visible').count(), 0,
      'Studio no longer repeats the footer resolution control in the composer body.');
    assert.equal(await sidebar.locator('select[aria-label="Title aspect ratio"]:visible').count(), 0,
      'Studio no longer repeats the footer aspect control in the composer body.');
    assert.equal(await sidebar.locator('select[aria-label="Title clip duration"]:visible').count(), 0,
      'Studio no longer repeats the footer duration control in the composer body.');
    await titleFormats.getByRole('button', {name: /^Resolution:/}).click();
    const titleResolutionMenu = page.getByRole('menu', {name: 'Resolution', exact: true});
    assert.deepEqual(await titleResolutionMenu.getByRole('menuitemradio').allTextContents()
      .then(rows => rows.map(row => /^\s*(\d+p)/.exec(row)?.[1])), ['480p', '540p', '720p', '768p', '1080p'],
      'Titles offers the standard video resolution tiers.');
    await titleResolutionMenu.getByRole('menuitemradio', {name: /540p/}).click();
    await titleFormats.getByRole('button', {name: 'Resolution: 540p', exact: true}).waitFor();
    await titleFormats.getByRole('button', {name: /^Aspect ratio:/}).click();
    await page.getByRole('menu', {name: 'Aspect ratio', exact: true})
      .getByRole('menuitemradio', {name: '4:3', exact: true}).click();
    await titleFormats.getByRole('button', {name: 'Aspect ratio: 4:3', exact: true}).waitFor();
    await titleFormats.getByRole('button', {name: /^Duration:/}).click();
    const titleDurationDialog = page.getByRole('dialog', {name: 'Title duration', exact: true});
    const titleDurationSlider = titleDurationDialog.getByRole('slider', {name: 'Title clip duration', exact: true});
    await titleDurationSlider.focus();
    await titleDurationSlider.press('Home');
    for (let step = 0; step < 7; step++) await titleDurationSlider.press('ArrowRight');
    await waitFor(page, () => JSON.parse(localStorage.getItem('maestro-studio-title-draft-v1')).sequenceFrames === 243,
      'native title duration selected from the footer slider');
    await page.keyboard.press('Escape');
    assert.equal(await titleDurationDialog.count(), 0, 'Escape closes the title duration popover.');
    assert.equal(await titleFormats.getByRole('button', {name: /^Duration:/}).evaluate(button => button === document.activeElement), true,
      'Closing the duration popover restores focus to its chip.');
    const selectedTitleFormat = await page.evaluate(() => {
      const draft = JSON.parse(localStorage.getItem('maestro-studio-title-draft-v1'));
      return {resolution: draft.resolution, aspect: draft.aspect, frames: draft.sequenceFrames};
    });
    assert.deepEqual(selectedTitleFormat, {resolution: '540p', aspect: '4:3', frames: 243},
      'The standard-looking controls persist the isolated title draft.');
    assert.deepEqual(await page.evaluate(() => {
      const state = window.store.getState();
      return {resolution: state.resolutionPreset, aspect: state.aspectRatio,
        duration: state.durationSeconds, resolutionParam: state.params.resolution,
        frames: state.params.video_length};
    }), ordinaryFormat, 'Changing title format never changes the ordinary Studio model format or frame count.');
    assert.equal(generationRequests.length, 0, 'Changing format never submits a generation.');
    await chooseWorkflow(page, 'Frames');
    await chooseWorkflow(page, 'Titles');
    await titleFormats.getByRole('button', {name: 'Resolution: 540p', exact: true}).waitFor();
    await titleFormats.getByRole('button', {name: 'Aspect ratio: 4:3', exact: true}).waitFor();
    assert.match(await titleFormats.getByRole('button', {name: /^Duration:/}).getAttribute('aria-label'), /10\.1/,
      'A native duration and the added format choices survive closing and reopening Titles.');
    await page.locator('[data-testid="studio-body-scroll"]').evaluate(element => { element.scrollTop = 0; });
    await page.screenshot({path: path.join(SCREENSHOTS, 'studio-titles-format-desktop.png'), fullPage: true});
    const advancedSidebar = await openTitleAdvanced(page);
    const presetSummary = advancedSidebar.getByText('Use an animation preset (optional)', {exact: true});
    await presetSummary.click();
    const preset = advancedSidebar.getByLabel('Animation preset');
    const presetCount = await preset.locator('option').count();
    assert.equal(presetCount, 21, 'The custom option and all 20 animation presets are available.');
    await preset.selectOption('dust');
    await advancedSidebar.getByLabel('Animated title text').fill('Maestro After Dark');
    await advancedSidebar.getByLabel('Title background').selectOption('black');
    await advancedSidebar.getByLabel('Title visual style').fill('Bright chrome lettering with a restrained blue rim light');
    await advancedSidebar.getByLabel('Title animation description').fill('The chrome words fly inward, hold, then fade to black.');
    await titleFormats.getByRole('button', {name: /^Duration:/}).click();
    await page.getByRole('dialog', {name: 'Title duration', exact: true})
      .getByLabel('Native title duration preset', {exact: true}).selectOption('209');
    await page.keyboard.press('Escape');
    await titleFormats.getByRole('button', {name: /^Aspect ratio:/}).click();
    await page.getByRole('menu', {name: 'Aspect ratio', exact: true})
      .getByRole('menuitemradio', {name: '9:16', exact: true}).click();
    await titleFormats.getByRole('button', {name: /^Resolution:/}).click();
    await page.getByRole('menu', {name: 'Resolution', exact: true})
      .getByRole('menuitemradio', {name: /720p/}).click();
    await openTitleAdvanced(page);
    assert.equal(await preset.inputValue(), 'custom',
      'Editing a preset description switches the title to its custom motion field.');
    const dimensions720 = await advancedSidebar.getByText(/704×1280/).count();
    assert.equal(dimensions720, 1, 'Portrait 720p title dimensions are shown before submit.');
    await titleFormats.getByRole('button', {name: /^Resolution:/}).click();
    await page.getByRole('menu', {name: 'Resolution', exact: true})
      .getByRole('menuitemradio', {name: /480p/}).click();
    await openTitleAdvanced(page);
    assert.equal(await advancedSidebar.getByText(/480×832/).count(), 1,
      'Portrait 480p title dimensions use the aligned output size.');
    await advancedSidebar.getByRole('button', {name: 'Use manual title plan'}).click();
    const bodyScroll = page.locator('[data-testid="studio-body-scroll"]');
    const desktopFooterBefore = await page.evaluate(() => ({
      top: document.querySelector('[data-testid="titles-generate-footer"]')?.getBoundingClientRect().top,
      hardwareTop: document.querySelector('.studio-hardware')?.getBoundingClientRect().top,
    }));
    const desktopScroll = await bodyScroll.evaluate(element => {
      element.scrollTop = element.scrollHeight;
      return {scrollTop: element.scrollTop, scrollHeight: element.scrollHeight, clientHeight: element.clientHeight};
    });
    await page.waitForTimeout(30);
    const desktopFooterAfter = await page.evaluate(() => ({
      top: document.querySelector('[data-testid="titles-generate-footer"]')?.getBoundingClientRect().top,
      hardwareTop: document.querySelector('.studio-hardware')?.getBoundingClientRect().top,
    }));
    assert.ok(desktopScroll.scrollHeight > desktopScroll.clientHeight && desktopScroll.scrollTop > 0,
      'The expanded title controls scroll inside the body.');
    assert.ok(Math.abs(desktopFooterAfter.top - desktopFooterBefore.top) < 1,
      'Scrolling title controls leaves the generate footer anchored at the sidebar bottom.');
    assert.ok(Math.abs(desktopFooterAfter.hardwareTop - desktopFooterBefore.hardwareTop) < 1,
      'The hardware monitor stays below the title footer while the form scrolls.');
    await bodyScroll.evaluate(element => { element.scrollTop = 0; });
    const requestedDimensions = await page.evaluate(() => window.titleHelpers.titleDimensions({
      text: 'Maestro After Dark', look: 'chrome', direction: 'fly inward', effect: 'custom',
      seconds: 8.4, aspect: '9:16', resolution: '480p', keepAudio: false, background: 'black',
    }, {width: 1920, height: 1080}));
    assert.deepEqual(requestedDimensions, {width: 480, height: 832});
    await page.locator('[data-testid="studio-body-scroll"]').evaluate(element => { element.scrollTop = 0; });
    await page.screenshot({path: path.join(SCREENSHOTS, 'studio-titles-desktop.png'), fullPage: true});

    // Simulate a lost HTTP response after acceptance. The client performs one exact lookup,
    // adopts the accepted job into the ordinary queue, and never posts again after reload.
    const generationButton = sidebar.getByRole('button', {name: 'Generate title'});
    await generationButton.dblclick({timeout: 2000}).catch(() => {});
    await waitFor(page, () => window.store.getState().jobs.some(job => job.id === 'studio-title-job-1'),
      'accepted title adoption into the normal generation queue');
    assert.equal(generationRequests.length, 1, 'A rapid double click sends only one generation POST.');
    const consumedEnhancement = await page.evaluate(() => ({
      oneShot: window.store.getState().enhanceOnGeneration,
      revision: window.store.getState().enhanceOnGenerationRevision,
    }));
    assert.equal(consumedEnhancement.oneShot, null, 'An accepted Studio title intent consumes its one-shot enhancement override.');
    assert.equal(consumedEnhancement.revision, oneShotEnhancementRevision + 1,
      'Consuming the one-shot enhancement increments its revision exactly once.');
    const firstRequest = generationRequests[0];
    assert.equal(firstRequest._studio_video_workflow, 'titles');
    assert.equal(firstRequest._studio_title_settings.background, 'black');
    assert.equal(firstRequest._studio_title_settings.aspect, '9:16');
    assert.equal(firstRequest._studio_title_settings.resolution, '480p');
    assert.equal(firstRequest.generation_mode, 'video');
    assert.equal(firstRequest.workspace, WORKSPACE);
    assert.equal(firstRequest.model_type, TITLE_MODEL);
    assert.equal(firstRequest.image_mode, 0);
    assert.deepEqual(firstRequest.image_refs, [], 'Normal Studio references do not leak into the title generation.');
    for (const key of ['_prompt_enhancement', 'h3_window_plan', 'enhanceOnGeneration', 'enhance_on_generation']) {
      assert.equal(Object.hasOwn(firstRequest, key), false, `Ordinary Studio enhancement state does not leak into title params (${key}).`);
    }
    assert.equal(uploadRequests.length, 1);
    assert.match(uploadRequests[0].body, /title-black-480x832\.png/);
    assert.equal(receiptLookups.length, 1, 'The lost receipt is looked up once by its exact client submission ID.');
    assert.equal(receiptLookups[0], firstRequest._client_submission_id);
    assert.deepEqual(projectApiRequests, [], 'Studio title generation never creates or saves an Editor project.');
    assert.equal(await page.evaluate(() => window.editorStore.getState().project), null,
      'A Studio title can generate with no Editor project.');
    assert.equal(await page.evaluate(() => window.store.getState().jobs.find(job => job.id === 'studio-title-job-1')?.showInGallery), true);

    // Change the active workspace only after the job was accepted; its receipt stays pinned to the original.
    await page.evaluate(() => window.store.setState({activeWorkspace: 'Changed While Running'}));
    await page.reload();
    await mount(page, bundle, css, {workflow: 'titles', workspace: 'Changed While Running'});
    await waitFor(page, () => window.titleStore.getState().intent?.jobId === 'studio-title-job-1',
      'title intent restoration after browser reload');
    await page.waitForTimeout(100);
    assert.equal(generationRequests.length, 1, 'Reloading a queued title never resubmits the generation POST.');
    assert.equal(receiptLookups.length, 1, 'The known job ID avoids another receipt lookup on reload.');
    assert.equal(await page.evaluate(() => window.titleStore.getState().intent?.workspace), WORKSPACE,
      'The accepted title retains its original workspace after the user switches workspaces.');

    statuses.set('studio-title-job-1', jobStatus('studio-title-job-1', firstRequest._client_submission_id,
      'completed', [`/outputs/${WORKSPACE}/title-black-480x832.mp4`]));
    currentOutput = {id: 'title-gallery-entry', name: 'title-black-480x832.mp4',
      type: 'video', mode: 'video', size: 1234, created_at: 100, url: '/outputs/title-black-480x832.mp4',
      path: `/outputs/${WORKSPACE}/title-black-480x832.mp4`, workspace: WORKSPACE};
    const refreshesBeforeComplete = outputRefreshes.length;
    await page.evaluate(() => window.titleStore.getState().reconcile());
    await waitFor(page, () => window.titleStore.getState().intent?.status === 'completed',
      'completed title verification');
    await waitFor(page, () => window.store.getState().outputs.some(output => output.name === 'title-black-480x832.mp4'),
      'normal gallery refresh');
    assert.ok(outputRefreshes.length > refreshesBeforeComplete, 'Completion refreshes the ordinary gallery.');
    assert.equal(probes.at(-1).workspace, WORKSPACE, 'Media verification uses the frozen title workspace.');
    assert.equal(await page.evaluate(() => window.store.getState().activeWorkspace), 'Changed While Running');
    assert.equal(await page.evaluate(() => window.editorStore.getState().project), null,
      'Completion leaves the Editor project unchanged.');

    // Failed and cancelled jobs report errors without a success state. Use green on the failure case.
    await page.evaluate(() => window.titleStore.getState().dismiss());
    await page.evaluate(() => window.store.getState().setEnhanceOnGeneration(true));
    const prepareFailureRevision = await page.evaluate(() => window.store.getState().enhanceOnGenerationRevision);
    failNextTitleUpload = true;
    await page.getByRole('button', {name: 'Generate title'}).click();
    await waitFor(page, () => window.titleStore.getState().error?.includes('Upload failed'),
      'title upload preparation failure');
    assert.equal(uploadFailures.length, 1, 'The title background upload failure is synthetic and isolated.');
    assert.equal(generationRequests.length, 1, 'A title preparation failure never sends a GPU generation request.');
    assert.equal(await page.evaluate(() => window.titleStore.getState().intent), null,
      'A pre-intent upload failure does not create a title submission identity.');
    const afterPrepareFailure = await page.evaluate(() => ({
      oneShot: window.store.getState().enhanceOnGeneration,
      revision: window.store.getState().enhanceOnGenerationRevision,
    }));
    assert.equal(afterPrepareFailure.oneShot, true, 'A pre-intent preparation failure preserves the one-shot enhancement choice.');
    assert.equal(afterPrepareFailure.revision, prepareFailureRevision,
      'A pre-intent failure does not consume or revise the one-shot choice.');
    await page.evaluate(() => window.titleStore.getState().dismiss());
    nextSubmissionMode = {status: 'failed', loseResponse: false};
    await page.getByLabel('Title background').first().selectOption('chroma');
    await acceptManualTitlePlan(sidebar);
    await page.evaluate(() => window.store.getState().setEnhanceOnGeneration(true));
    const raceRevision = await page.evaluate(() => window.store.getState().enhanceOnGenerationRevision);
    const racedGeneration = {started: deferred(), release: deferred()};
    generationGate = racedGeneration;
    await page.getByRole('button', {name: 'Generate title'}).click();
    await racedGeneration.started.promise;
    await page.evaluate(() => window.store.getState().setEnhanceOnGeneration(false));
    const laterPreferenceRevision = await page.evaluate(() => window.store.getState().enhanceOnGenerationRevision);
    assert.equal(laterPreferenceRevision, raceRevision + 1, 'A later user choice advances the enhancement revision during submit.');
    racedGeneration.release.resolve();
    await waitFor(page, () => window.titleStore.getState().intent?.status === 'failed',
      'failed title generation');
    const preservedLaterChoice = await page.evaluate(() => ({
      oneShot: window.store.getState().enhanceOnGeneration,
      revision: window.store.getState().enhanceOnGenerationRevision,
    }));
    assert.equal(preservedLaterChoice.oneShot, false,
      'Submitting an older title intent preserves a later explicit enhancement choice.');
    assert.equal(preservedLaterChoice.revision, laterPreferenceRevision,
      'The submission completion does not increment or clear a newer enhancement revision.');
    assert.equal(generationRequests.length, 2);
    assert.equal(generationRequests[1]._studio_title_settings.background, 'chroma');
    assert.match(uploadRequests[1].body, /title-green-/);
    assert.equal(await page.getByRole('alert').filter({hasText: 'Synthetic generation failure'}).count(), 1);
    assert.equal(await page.getByRole('button', {name: 'Dismiss', exact: true}).count(), 0,
      'The title sidebar does not show the old status banner actions.');

    await page.evaluate(() => window.titleStore.getState().dismiss());
    nextSubmissionMode = {status: 'cancelled', loseResponse: false};
    await page.getByRole('button', {name: 'Generate title'}).click();
    await waitFor(page, () => window.titleStore.getState().intent?.status === 'failed'
      && window.titleStore.getState().intent?.error === 'Synthetic cancellation', 'cancelled title generation');
    assert.equal(generationRequests.length, 3);
    assert.equal(await page.getByRole('alert').filter({hasText: 'Synthetic cancellation'}).count(), 1);
    assert.equal(await page.getByRole('link', {name: 'Preview', exact: true}).count(), 0);
    assert.deepEqual(projectApiRequests, [], 'No Studio title path calls Editor project APIs.');

    // Phone layout keeps the title form scrollable while the keyboard is open, and the
    // anchored workflow chooser remains a non-modal popover without a focus trap.
    await page.setViewportSize({width: 390, height: 844});
    await page.waitForTimeout(100);
    await assertHardwareExpandable(page, 'Mobile');
    if (await page.getByRole('dialog', {name: 'Advanced title settings', exact: true}).isVisible()) await page.keyboard.press('Escape');
    await page.locator('[data-testid="studio-body-scroll"]').evaluate(element => { element.scrollTop = 0; });
    await titleFormats.getByRole('button', {name: /^Resolution:/}).click();
    const phoneResolutionMenu = page.getByRole('menu', {name: 'Resolution', exact: true});
    const phoneResolutionBounds = await phoneResolutionMenu.boundingBox();
    assert.ok(phoneResolutionBounds && phoneResolutionBounds.x >= 0
      && phoneResolutionBounds.x + phoneResolutionBounds.width <= 391,
      'The standard title resolution menu fits the phone viewport.');
    await page.screenshot({path: path.join(SCREENSHOTS, 'studio-titles-mobile-resolution.png'), fullPage: true});
    await page.keyboard.press('Escape');
    await titleFormats.getByRole('button', {name: /^Duration:/}).click();
    const phoneDurationDialog = page.getByRole('dialog', {name: 'Title duration', exact: true});
    const phoneDurationBounds = await phoneDurationDialog.boundingBox();
    assert.ok(phoneDurationBounds && phoneDurationBounds.x >= 0 && phoneDurationBounds.y >= 0
      && phoneDurationBounds.x + phoneDurationBounds.width <= 391
      && phoneDurationBounds.y + phoneDurationBounds.height <= 845,
      'The title duration popover fits above the fixed footer on a phone.');
    await page.screenshot({path: path.join(SCREENSHOTS, 'studio-titles-mobile-duration.png'), fullPage: true});
    await page.keyboard.press('Escape');
    await page.screenshot({path: path.join(SCREENSHOTS, 'studio-titles-mobile.png'), fullPage: true});
    await page.locator('.maestro-sidebar').locator('.studio-workflow-header button[title]').click();
    const mobileDialog = page.getByRole('dialog', {name: 'Choose a workflow'});
    assert.notEqual(await mobileDialog.getAttribute('aria-modal'), 'true', 'Mobile workflow selection is not a modal focus trap.');
    const lastOption = mobileDialog.getByRole('button', {name: /^Titles/});
    await lastOption.focus();
    await page.evaluate(() => {
      window.lastTabPrevented = null;
      document.addEventListener('keydown', event => {
        if (event.key === 'Tab') window.lastTabPrevented = event.defaultPrevented;
      }, {once: true});
    });
    await page.keyboard.press('Tab');
    assert.equal(await page.evaluate(() => window.lastTabPrevented), false,
      'The phone workflow popover lets keyboard focus continue naturally.');
    await page.keyboard.press('Escape');
    await openTitleAdvanced(page);
    await page.getByLabel('Title animation description').waitFor();
    await page.keyboard.press('Escape');
    await page.getByLabel('Describe your title animation').focus();
    await page.getByLabel('Describe your title animation').fill('Mobile title motion stays in the visible sidebar.');
    await page.setViewportSize({width: 390, height: 500});
    await waitFor(page, () => document.querySelector('.maestro-sidebar')?.getAttribute('data-keyboard-open') === 'true',
      'mobile keyboard viewport handling');
    const scroll = await page.locator('[data-testid="studio-body-scroll"]').evaluate(element => ({
      scrollHeight: element.scrollHeight, clientHeight: element.clientHeight,
      width: element.scrollWidth, clientWidth: element.clientWidth,
    }));
    assert.ok(scroll.scrollHeight > scroll.clientHeight, 'The mobile Studio Titles panel remains vertically scrollable.');
    assert.ok(scroll.width <= scroll.clientWidth, 'The mobile title form does not overflow horizontally.');
    const mobileFooterBefore = await page.evaluate(() => ({
      hostHidden: document.querySelector('[data-testid="titles-footer-host"]')?.hidden,
      footerTop: document.querySelector('[data-testid="titles-generate-footer"]')?.getBoundingClientRect().top,
      footerBottom: document.querySelector('[data-testid="titles-generate-footer"]')?.getBoundingClientRect().bottom,
      sidebarBottom: document.querySelector('.maestro-sidebar')?.getBoundingClientRect().bottom,
    }));
    const mobileScroll = await page.locator('[data-testid="studio-body-scroll"]').evaluate(element => {
      element.scrollTop = element.scrollHeight;
      return {scrollTop: element.scrollTop};
    });
    await page.waitForTimeout(30);
    const mobileFooterAfter = await page.evaluate(() => ({
      footerTop: document.querySelector('[data-testid="titles-generate-footer"]')?.getBoundingClientRect().top,
      footerBottom: document.querySelector('[data-testid="titles-generate-footer"]')?.getBoundingClientRect().bottom,
      sidebarBottom: document.querySelector('.maestro-sidebar')?.getBoundingClientRect().bottom,
    }));
    assert.equal(mobileFooterBefore.hostHidden, false, 'The title footer remains visible while the mobile keyboard is open.');
    assert.ok(Math.abs(mobileFooterAfter.footerTop - mobileFooterBefore.footerTop) < 1,
      'Mobile body scrolling does not move the fixed title footer.');
    assert.ok(mobileFooterAfter.footerBottom <= mobileFooterAfter.sidebarBottom + 1,
      'The title footer stays within the visible mobile sidebar while the keyboard is open.');
    assert.ok(mobileScroll.scrollTop > 0, 'Mobile scrolling continues inside the title form rather than the footer.');

    // Sequence settings survive the real Studio normalization, submission and
    // browser reload paths. All GPU endpoints remain synthetic.
    await page.setViewportSize({width: 1360, height: 900});
    await page.evaluate(() => window.titleStore.getState().dismiss());
    const sequenceSettings = {mode: 'sequence', text: 'STALE SINGLE TITLE', look: 'Hand-drawn blue ink',
      effect: 'fire', direction: 'Sparks trace the complete title and hold it together.', sequenceDirection: 'Keep the camera locked.',
      seconds: 5.2, aspect: '9:16', resolution: '480p', keepAudio: false, background: 'black',
      phrases: [
        {id: 'seq-a', text: 'CREATE', seconds: 3, animation: 'Scale in, hold, then clear.'},
        {id: 'seq-b', text: 'EXPLORE', seconds: 3, animation: 'Rotate in, hold, then clear.'},
        {id: 'seq-c', text: 'INSPIRE', seconds: 4, animation: ''},
      ]};
    const planning = await page.evaluate(settings => {
      const h = window.titleHelpers;
      const invalids = [[], settings.phrases.slice(0, 1), [...settings.phrases, ...settings.phrases, settings.phrases[0]],
        settings.phrases.map(p => ({...p, seconds: .5})), settings.phrases.map(p => ({...p, seconds: 7})),
        settings.phrases.map(p => ({...p, text: ''})), settings.phrases.map(p => ({...p, id: 'duplicate'})),
        settings.phrases.map(p => ({...p, text: 'x'.repeat(60)}))];
      return {timeline: h.titleSequenceTimeline(settings),
        prompt: h.animatedTitlePrompt(settings),
        body: h.animatedTitleParams(settings, {width: 1920, height: 1080}, '/uploads/black.png', 'test', 'synthetic-id'),
        invalids: invalids.map(phrases => {try { h.animatedTitlePrompt({...settings, phrases}); return ''; } catch (error) {return error.message;}}),
        converted: h.newTitlePhrases('One\nTwo\nThree\nFour\nFive\nSix', 5.2)};
    }, sequenceSettings);
    assert.equal(planning.timeline.requestedSeconds, 10);
    assert.equal(planning.timeline.duration, 243 / 24);
    assert.equal(planning.timeline.phrases[0].start, 0);
    assert.equal(planning.timeline.phrases.at(-1).end, planning.timeline.duration);
    for (let i = 1; i < 3; i++) assert.equal(planning.timeline.phrases[i].start, planning.timeline.phrases[i - 1].end);
    assert.equal(planning.body.video_length, 243, 'The summed phrase duration overrides the hidden single-title duration.');
    assert.match(planning.prompt, /exactly ONE phrase at a time/);
    assert.match(planning.prompt, /Final slot: show ONLY "INSPIRE"; no earlier phrase returns/);
    assert.doesNotMatch(planning.prompt, /complete-title hold|Sparks trace|STALE SINGLE TITLE/,
      'Sequence prompts do not inherit the previous title or full-title preset instructions.');
    assert.ok(planning.invalids.every(Boolean), 'Invalid phrase counts, durations, texts and identities block submission.');
    assert.ok(planning.converted.every(phrase => phrase.seconds >= 1), 'Converting six lines allocates usable slots.');
    assert.equal(new Set(planning.converted.map(phrase => phrase.id)).size, 6);
    const nativeChecks = await page.evaluate(settings => {
      const h = window.titleHelpers;
      let checked = 0;
      const check = (value, frames) => {
        const counts = value.phrases.map(phrase => Math.round(phrase.seconds * 24));
        if (counts.some(count => count < 24 || count > 240)
          || counts.reduce((sum, count) => sum + count, 0) !== frames) throw new Error('Invalid bounded frame allocation');
        const timeline = h.titleSequenceTimeline(value);
        if (timeline.duration !== frames / 24 || timeline.phrases.at(-1).end !== frames / 24
          || timeline.phrases.some((row, i) => i && row.start !== timeline.phrases[i - 1].end)) throw new Error('Non-contiguous native slots');
        const params = h.animatedTitleParams(value, {width: 1920, height: 1080}, '/uploads/black.png', 'test', 'synthetic-native-id');
        if (params.video_length !== frames) throw new Error('Payload changed the selected native duration');
        checked++;
      };
      for (let count = 2; count <= 6; count++) for (const choice of h.TITLE_SEQUENCE_DURATIONS) {
        if (choice.frames < count * 24) continue;
        for (const skewed of [false, true]) {
          const original = {...settings, sequenceFrames: undefined,
            phrases: Array.from({length: count}, (_, i) => ({id: `n-${i}`, text: `WORD ${i}`, animation: '', seconds: skewed && i === 0 ? 10 : 1}))};
          const before = JSON.stringify(original);
          const aligned = h.alignTitleSequence(original, choice.frames);
          check(aligned, choice.frames);
          if (JSON.stringify(original) !== before) throw new Error('Alignment mutated an existing draft');
          for (const phrase of aligned.phrases) for (const seconds of [0, 1, 3.333, 10, 99]) {
            check(h.setTitlePhraseSeconds(aligned, phrase.id, seconds), choice.frames);
          }
        }
      }
      const six = {...settings, phrases: Array.from({length: 6}, (_, i) => ({id: `six-${i}`, text: `TEXT ${i}`, seconds: 1, animation: ''}))};
      const sixAligned = h.alignTitleSequence(six, 124);
      return {checked, sixFrames: sixAligned.sequenceFrames,
        invalidFrames: [123, 142, 346, NaN].map(sequenceFrames => {
          try {h.animatedTitlePrompt({...settings, sequenceFrames}); return false;} catch {return true;}
        })};
    }, sequenceSettings);
    assert.ok(nativeChecks.checked > 2000, 'Every compatible native length is checked with even/skewed slots and edited phrase limits.');
    assert.equal(nativeChecks.sixFrames, 158, 'Six phrases get the next native duration that can fit at least one second each.');
    assert.ok(nativeChecks.invalidFrames.every(Boolean), 'Invalid native frame selections cannot generate.');
    await page.evaluate(settings => window.store.setState(state => ({params: {...state.params, _studio_title_settings: settings}})), sequenceSettings);
    await openTitleAdvanced(page);
    await waitFor(page, () => document.querySelector('[aria-label="Phrase 1 text"]')?.value === 'CREATE', 'gallery sequence settings restore');
    assert.equal(await page.getByLabel('Phrase 3 text', {exact: true}).inputValue(), 'INSPIRE');
    await page.getByLabel('Phrase 1 text', {exact: true}).fill('CREATE MORE');
    await waitFor(page, () => window.store.getState().params._studio_title_settings === undefined, 'sequence override becomes local draft');
    nextSubmissionMode = {status: 'running', loseResponse: true};
    await page.getByRole('button', {name: 'Generate title', exact: true}).click();
    await waitFor(page, () => window.titleStore.getState().intent?.jobId === 'studio-title-job-4', 'sequence lost-receipt adoption');
    const sequenceBody = generationRequests.at(-1);
    assert.equal(sequenceBody._studio_title_settings.mode, 'sequence');
    assert.deepEqual(sequenceBody._studio_title_settings.phrases.map(p => p.text), ['CREATE MORE', 'EXPLORE', 'INSPIRE']);
    assert.equal(sequenceBody._studio_title_settings.text, 'CREATE MORE\nEXPLORE\nINSPIRE');
    assert.equal(sequenceBody.video_length, 243);
    assert.equal(sequenceBody._studio_title_settings.sequenceFrames, 243, 'The saved title records the selected native duration.');
    assert.match(sequenceBody.prompt, /Keep the camera locked/);
    assert.equal(generationRequests.length, 4, 'One sequence sends one POST, without per-phrase jobs.');
    assert.equal(receiptLookups.at(-1), sequenceBody._client_submission_id);
    const beforeSequenceReload = generationRequests.length;
    await page.reload();
    await mount(page, bundle, css, {workflow: 'titles', workspace: 'Changed While Running'});
    await waitFor(page, () => window.titleStore.getState().intent?.jobId === 'studio-title-job-4', 'sequence receipt reload');
    await openTitleAdvanced(page);
    assert.equal(await page.getByLabel('Phrase 1 text', {exact: true}).inputValue(), 'CREATE MORE');
    assert.equal(await page.getByLabel('Sequence duration', {exact: true}).inputValue(), '243');
    const recoveredPhrases = await page.evaluate(() => window.titleStore.getState().intent.settings.phrases);
    assert.equal(recoveredPhrases.reduce((sum, phrase) => sum + Math.round(phrase.seconds * 24), 0), 243,
      'Recovered phrase slots add up to the selected native frame count.');
    assert.equal(generationRequests.length, beforeSequenceReload, 'Sequence reload never resubmits.');
    assert.deepEqual(projectApiRequests, [], 'Sequence generation remains independent of Editor projects.');

    // Controlled Frames catalog, native timing/defaults, and all chosen background colors.
    await page.keyboard.press('Escape');
    await page.evaluate(() => {
      window.titleStore.getState().dismiss();
      window.store.setState(state => ({enhanceOnGeneration: false, enhanceOnGenerationDefault: false,
        params: {...state.params, _studio_title_settings: {
          brief: 'Show “COLOR” with a clean reveal, readable hold, then dissolve.', text: '', look: '', direction: '',
          effect: 'custom', seconds: 5.2, aspect: '16:9', resolution: '480p', background: 'white', keepAudio: false,
        }}}));
    });
    const normalBeforeModel = await stateInputs(page);
    await page.getByRole('button', {name: 'Choose model', exact: true}).click();
    await page.getByRole('dialog', {name: 'Choose a model', exact: true}).getByTitle('Test Frames T2V', {exact: true}).click();
    await waitFor(page, () => document.querySelector('[aria-label="Choose model"]')?.dataset.modelType === 'test_frames_text', 'controlled T2V title model');
    assert.deepEqual(await stateInputs(page), normalBeforeModel, 'Selecting a title model preserves the ordinary Studio model and inputs.');
    const modelAdvanced = await openTitleAdvanced(page);
    assert.equal(await modelAdvanced.getByLabel('Title inference steps').inputValue(), '12');
    assert.equal(await modelAdvanced.getByLabel('Title guidance', {exact: true}).inputValue(), '2.5');
    await modelAdvanced.getByLabel('Title seed', {exact: true}).fill('314159');
    await page.screenshot({path: path.join(SCREENSHOTS, 'studio-title-model-advanced.png'), fullPage: true});
    await page.keyboard.press('Escape');
    nextSubmissionMode = {status: 'completed', loseResponse: false};
    const beforeTextUploads = uploadRequests.length;
    await page.getByRole('button', {name: 'Generate title', exact: true}).click();
    await waitFor(page, () => window.titleStore.getState().intent?.status === 'completed', 'text-model title completion');
    const textBody = generationRequests.at(-1);
    assert.equal(textBody.model_type, 'test_frames_text');
    assert.equal(textBody._studio_title_settings.model.fps, 16);
    assert.equal(textBody.video_length % 4, 1, 'The text model uses its own native frame lattice.');
    assert.equal(textBody.num_inference_steps, 12);
    assert.equal(textBody.guidance_scale, 2.5);
    assert.equal(textBody.seed, 314159);
    assert.equal(textBody.sample_solver, 'euler');
    assert.equal(textBody.multi_prompts_gen_type, 2, 'A multiline title prompt always remains one clip.');
    assert.ok(!textBody.image_start && !textBody.image_end && !textBody.image_prompt_type);
    assert.notEqual(textBody.override_attention, 'sla', 'Foreign models never inherit the Fused SLA override.');
    assert.match(textBody.prompt, /#FFFFFF/i);
    assert.equal(uploadRequests.length, beforeTextUploads, 'A text-only model is not sent unsupported boundary images.');
    assert.equal(await page.getByRole('button', {name: 'Dismiss', exact: true}).count(), 0);
    assert.equal(await page.getByRole('link', {name: 'Preview', exact: true}).count(), 0);
    await page.getByRole('button', {name: 'Choose model', exact: true}).click();
    await page.getByRole('dialog', {name: 'Choose a model', exact: true}).getByTitle('Test Frames I2V', {exact: true}).click();
    await waitFor(page, () => document.querySelector('[aria-label="Choose model"]')?.dataset.modelType === 'test_frames_image', 'controlled I2V title model');
    await page.getByLabel('Title background', {exact: true}).first().selectOption('blue');
    nextSubmissionMode = {status: 'completed', loseResponse: false};
    await page.getByRole('button', {name: 'Generate title', exact: true}).click();
    await waitFor(page, () => window.titleStore.getState().intent?.status === 'completed'
      && window.titleStore.getState().intent?.settings.background === 'blue', 'blue I2V title completion');
    const imageBody = generationRequests.at(-1);
    assert.equal(imageBody.model_type, 'test_frames_image');
    assert.equal(imageBody.num_inference_steps, 8);
    assert.ok(imageBody.image_start);
    assert.ok(!imageBody.image_end, 'A start-only model is not sent an unsupported last frame.');
    assert.equal(imageBody.image_prompt_type, 'S');
    assert.equal(imageBody.multi_prompts_gen_type, 2);
    assert.deepEqual(await uploadedPixel(page, uploadRequests.at(-1)), [0, 0, 255, 255]);
    assert.match(imageBody.prompt, /#0000FF/i);
    await page.getByRole('button', {name: 'Choose model', exact: true}).click();
    await page.getByRole('dialog', {name: 'Choose a model', exact: true}).getByTitle('H3 Fused 4-Step · Frames', {exact: true}).click();
    await waitFor(page, () => document.querySelector('[aria-label="Choose model"]')?.dataset.modelType === 'minimax_h3_fused_turbo', 'return to H3 title model');
    await page.getByLabel('Title background', {exact: true}).first().selectOption('custom');
    await page.getByLabel('Title background color', {exact: true}).first().evaluate(input => {
      Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value').set.call(input, '#334455');
      input.dispatchEvent(new Event('input', {bubbles: true}));
    });
    await page.setViewportSize({width: 390, height: 844});
    await page.locator('[data-testid="studio-body-scroll"]').evaluate(element => { element.scrollTop = 0; });
    await page.screenshot({path: path.join(SCREENSHOTS, 'studio-title-custom-mobile.png'), fullPage: true});
    await page.getByRole('button', {name: 'Choose model', exact: true}).click();
    await page.screenshot({path: path.join(SCREENSHOTS, 'studio-title-model-picker-mobile.png'), fullPage: true});
    await page.keyboard.press('Escape');
    await page.setViewportSize({width: 1360, height: 900});
    nextSubmissionMode = {status: 'completed', loseResponse: false};
    await page.getByRole('button', {name: 'Generate title', exact: true}).click();
    await waitFor(page, () => window.titleStore.getState().intent?.status === 'completed'
      && window.titleStore.getState().intent?.settings.background === 'custom', 'custom-color title completion');
    assert.deepEqual(await uploadedPixel(page, uploadRequests.at(-1)), [51, 68, 85, 255]);
    assert.match(generationRequests.at(-1).prompt, /#334455/i);
    const editorKeying = await page.evaluate(() => ['blue', 'white', 'custom'].map(background => {
      const project = {assets: {}, tracks: []};
      window.titleHelpers.insertAnimatedTitle(project, {id: 'overlay', type: 'video', duration: 5}, {
        id: 'title', at: 0, text: 'COLOR', keepAudio: false, itemId: 'item', trackId: 'track', background, backgroundColor: '#334455',
      });
      const key = project.tracks[0].items[0].keying;
      return {background, enabled: key.enabled, ...(key.enabled ? {color: key.color} : {})};
    }));
    assert.deepEqual(editorKeying, [{background: 'blue', enabled: true, color: '#0000ff'},
      {background: 'white', enabled: false}, {background: 'custom', enabled: false}],
      'Blue keys with blue; white and custom title backgrounds remain opaque in the Editor.');
    assert.deepEqual(await stateInputs(page), normalBeforeModel, 'All title model/color submissions preserve unrelated Studio inputs.');
    assert.ok(preferenceUpdates.some(preference => preference.studio_video_workflow === 'titles'),
      'The title workflow is persisted as a normal Studio preference.');
    assert.deepEqual(pageErrors, [], 'The isolated Studio Titles sidebar has no runtime errors.');
    console.log(`Studio Titles workflow passed: Frames/Blend/Viggle routing, isolated inputs, portrait settings, receipt recovery, queue/gallery adoption, failures, and mobile layout. Screenshots: ${path.join(SCREENSHOTS, 'studio-titles-desktop.png')} and ${path.join(SCREENSHOTS, 'studio-titles-mobile.png')}`);
  } finally {
    await browser.close();
  }
})().catch(error => {
  console.error(error.stack || error);
  process.exitCode = 1;
});
