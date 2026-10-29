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
const TEST_LORA = 'studio-title-style.safetensors';
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
  model_type: TITLE_MODEL, architecture: 'minimax_h3',
  t2v_class: 'text', i2v_class: 'image', image_ref_choices: ['I'],
  video_prompt_type_sources: [], image_prompt_type_sources: [],
  resolution_presets: {}, default_num_inference_steps: 4,
  inference_steps_min: 1, inference_steps_max: 50,
  sla_attention: true, sla_attention_default: false,
  sla_attention_status: {supported: true},
  first_block_cache: true,
  default_skip_steps_multiplier: 0.08,
  skip_steps_multiplier_choices: [['Conservative (0.06)', 0.06], ['Balanced (0.08)', 0.08], ['Fast (0.12)', 0.12]],
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
        import {GlobalQueuePopover} from './src/components/GlobalQueuePopover';
        import {titleDimensions, insertAnimatedTitle, titlePlanningKey, titlePlanIsCurrent} from './src/editor/editorAnimatedTitles';
        window.store = useStore;
        window.editorStore = useEditorStore;
        window.titleStore = useEditorTitleStore;
        window.titleHelpers = {titleDimensions, insertAnimatedTitle, titlePlanningKey, titlePlanIsCurrent};
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
          window.reactRoot.render(<><Sidebar onOpenWorlds={() => {window.worldsOpened = (window.worldsOpened || 0) + 1;}}/><div style={{position:'fixed',top:8,right:8,zIndex:100}}><GlobalQueuePopover/></div></>);
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
      voiceA: state.params.audio_guide,
      voiceB: state.params.audio_guide2,
      voiceFilenameA: state.audioGuideFilename,
      voiceFilenameB: state.audioGuide2Filename,
      speakers: state.params.speakers_locations,
      advanced: {
        steps: state.params.num_inference_steps,
        seed: state.params.seed,
        activatedLoras: [...(state.params.activated_loras || [])],
        loraMultipliers: state.params.loras_multipliers,
        loraWeights: JSON.parse(JSON.stringify(state.loraWeights || {})),
        attention: state.params.override_attention ?? '',
        cache: state.params.skip_steps_cache_type,
        cacheThreshold: state.params.skip_steps_multiplier,
        cacheWarmup: state.params.skip_steps_start_step_perc,
        spatialUpsampling: state.spatialUpsampling,
      },
    };
  });
}

async function assertTitleModeHasOnlyItsOwnComposer(page) {
  const sidebar = page.locator('.maestro-sidebar');
  await page.getByRole('region', {name: 'Studio Titles'}).waitFor();
  await waitFor(page, () => {
    const state = window.store.getState();
    return !!state.titleGenerationContext && !state.modelOptionsLoading
      && state.modelOptions?.model_type === state.params.model_type;
  }, 'active title model controls');
  assert.equal(await sidebar.locator('textarea[aria-label="Generation prompt"]').count(), 0,
    'Titles replaces the ordinary generation prompt.');
  assert.equal(await sidebar.locator('[data-studio-footer]').count(), 0,
    'Titles hides the ordinary settings/footer controls.');
  assert.equal(await sidebar.getByText('Control video', {exact: true}).count(), 0,
    'Titles hides the ordinary control-video input.');
  const advancedTrigger = sidebar.locator('button[aria-label^="Advanced settings"]');
  assert.equal(await advancedTrigger.count(), 1,
    'Titles uses the shared compact Advanced settings control.');
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
    'Resolution, aspect, duration and shared Advanced use the same compact chips as the other Studio modes.');
  assert.equal(await formatControls.evaluate(element => !!element.closest('[data-testid="titles-generate-footer"]')), true,
    'The everyday format controls remain in the fixed footer rather than the scrolling title form.');
  assert.equal(await formatControls.locator('button[aria-label^="Advanced settings"]').count(), 1,
    'Titles mounts the shared compact Advanced settings trigger in the format row.');
  assert.equal(await page.getByRole('button', {name: 'Generate', exact: true}).count(), 1,
    'Titles uses the shared Generate split button.');
  assert.equal(await page.getByRole('button', {name: 'Add current Studio settings to the queue', exact: true}).count(), 1,
    'Titles exposes the shared Add current Studio settings to the queue action.');
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

async function openTitleAdvanced(page) {
  const trigger = page.locator('button[aria-label^="Advanced settings"]');
  const advanced = page.getByRole('dialog', {name: 'Advanced settings', exact: true});
  if (!await advanced.isVisible()) await trigger.click();
  await advanced.waitFor();
  assert.match(await trigger.getAttribute('aria-label'), /^Advanced settings/,
    'The compact trigger exposes the shared Advanced settings label.');
  return advanced;
}

async function setRangeValue(locator, value) {
  await locator.evaluate((input, next) => {
    Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value').set.call(input, String(next));
    input.dispatchEvent(new Event('input', {bubbles: true}));
    input.dispatchEvent(new Event('change', {bubbles: true}));
  }, value);
}

async function setStudioSeed(page, value) {
  const advanced = await openTitleAdvanced(page);
  const generation = advanced.locator('[data-testid="advanced-generation"]');
  const seed = generation.getByText('Seed', {exact: true}).locator('..').locator('..').locator('input[type="number"]');
  if (!await seed.isVisible()) await generation.locator('summary').click();
  await generation.getByText('Seed', {exact: true}).locator('..').locator('..').locator('input[type="number"]').fill(String(value));
  await waitFor(page, expected => window.store.getState().params.seed === Number(expected),
    'shared Studio seed setting', 10000, value);
  await page.keyboard.press('Escape');
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
  const startQueueRequests = [];
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
        model_type: endpoint.endsWith('_text') ? 'test_frames_text' : 'test_frames_image',
        architecture: 'wan', sla_attention: false, first_block_cache: false,
        fps: endpoint.endsWith('_text') ? 16 : 24,
        frames_minimum: 1, frames_steps: endpoint.endsWith('_text') ? 4 : 8, frames_maximum: endpoint.endsWith('_text') ? 161 : 241,
        default_num_inference_steps: endpoint.endsWith('_text') ? 12 : 8, default_guidance_scale: 2.5});
      if (/^\/api\/v1\/loras\/[^/]+$/.test(endpoint)) return json({loras: [TEST_LORA], guidance_max_phases: 1});
      if (/^\/api\/v1\/loras\/[^/]+\/details$/.test(endpoint)) return json({loras: [{filename: TEST_LORA,
        recommended_weights: {min: 0.4, max: 1.2, default: 0.75, source: 'default'}}], guidance_max_phases: 1});
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
        const outputs = Array.isArray(currentOutput) ? currentOutput : currentOutput ? [currentOutput] : [];
        return json({outputs, total: outputs.length, next_cursor: null});
      }
      if (endpoint === '/api/v1/jobs/queue/start' && method === 'POST') {
        startQueueRequests.push({method, endpoint});
        const jobIds = [...statuses.entries()].filter(([, status]) => status.status === 'held').map(([jobId]) => jobId);
        for (const jobId of jobIds) {
          const held = statuses.get(jobId);
          statuses.set(jobId, {...held, status: 'queued', message: 'Synthetic queue release'});
        }
        return json({status: jobIds.length ? 'started' : 'idle', job_ids: jobIds, released: jobIds.length});
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
        const statusName = body._queue_mode === 'held' ? 'held' : mode.status;
        submissionModes.push({...mode, queueMode: body._queue_mode || 'now'});
        nextSubmissionMode = {status: 'queued', loseResponse: false};
        const outputPath = `/outputs/${WORKSPACE}/studio-title-${generationRequests.length}.mp4`;
        const status = jobStatus(jobId, clientId, statusName,
          statusName === 'completed' ? [outputPath] : []);
        statuses.set(jobId, status);
        receipts.set(clientId, {job_id: jobId, client_submission_id: clientId, status: statusName});
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
    assert.equal(await page.getByRole('region', {name: 'Studio Titles'}).count(), 1, 'Titles shows its dedicated creative composer.');
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
    let oneShotEnhancementRevision = await page.evaluate(() => window.store.getState().enhanceOnGenerationRevision);
    assert.equal(await page.evaluate(() => window.store.getState().enhanceOnGeneration), true,
      'A one-shot title enhancement can override the saved default.');

    await page.getByLabel('Describe your title animation').fill('Edited loaded gallery title');
    await page.locator('[data-testid="titles-generate-footer"]').getByRole('button', {name: 'Generate'}).click();
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

    await waitFor(page, () => JSON.parse(localStorage.getItem('maestro-studio-title-draft-v1') || 'null')?.titleReviewRequired === true,
      'reviewed Studio title draft is saved separately');
    const beforeStudioClear = await page.evaluate(() =>
      JSON.parse(localStorage.getItem('maestro-studio-title-draft-v1') || 'null'));
    const plannerRequestsBeforeStudioClear = titlePlanRequests.length;
    await page.screenshot({path: path.join(SCREENSHOTS, 'studio-title-clear-plan-review.png'), fullPage: true});
    await page.getByRole('button', {name: 'Clear plan', exact: true}).click();
    await waitFor(page, () => {
      const saved = JSON.parse(localStorage.getItem('maestro-studio-title-draft-v1') || 'null');
      return saved && !saved.plannedKey && saved.titleReviewRequired !== true
        && !saved.rawTitleDraft && !saved.titleDraftHistoryId;
    }, 'cleared Studio plan saved to the separate title draft');
    const clearedStudioDraft = await page.evaluate(() =>
      JSON.parse(localStorage.getItem('maestro-studio-title-draft-v1') || 'null'));
    assert.equal(clearedStudioDraft.brief, beforeStudioClear.brief,
      'Clear plan preserves the Studio brief.');
    for (const field of ['stylePreset', 'background', 'backgroundColor', 'model', 'seconds', 'aspect', 'resolution', 'generationParams']) {
      assert.deepEqual(clearedStudioDraft[field], beforeStudioClear[field],
        `Clear plan preserves the Studio ${field} setting.`);
    }
    assert.equal(clearedStudioDraft.text, '');
    assert.equal(clearedStudioDraft.look, '');
    assert.equal(clearedStudioDraft.direction, '');
    assert.deepEqual(clearedStudioDraft.phrases || [], []);
    assert.equal(clearedStudioDraft.mode, 'single');
    assert.equal(clearedStudioDraft.titleReviewRequired, false);
    assert.equal(clearedStudioDraft.plannedKey, undefined);
    assert.deepEqual(clearedStudioDraft.plannerWarnings || [], []);
    assert.equal(clearedStudioDraft.rawTitleDraft, undefined);
    assert.equal(clearedStudioDraft.titleDraftHistoryId, undefined);
    assert.equal(await page.getByTestId('title-plan-review').count(), 0);
    assert.equal(await page.getByText('Raw planner draft diagnostics', {exact: true}).count(), 0);
    assert.equal(titlePlanRequests.length, plannerRequestsBeforeStudioClear,
      'Clear plan makes no title planner request.');
    assert.equal(generationRequests.length, 0, 'Clear plan does not submit or cancel a title job.');
    assert.equal(await page.evaluate(() => window.titleStore.getState().intent), null,
      'Clearing a plan leaves the Studio title job and receipt state untouched.');
    assert.equal(await page.evaluate(() => window.store.getState().enhanceOnGeneration), false,
      'Clear plan turns off the one-shot enhancement choice.');
    assert.equal(await page.evaluate(() => window.store.getState().enhanceOnGenerationDefault), false,
      'Clear plan leaves the saved enhancement default unchanged.');
    oneShotEnhancementRevision = await page.evaluate(() => window.store.getState().enhanceOnGenerationRevision);

    const studioBrief = page.getByLabel('Describe your title animation');
    await studioBrief.fill('');
    await waitFor(page, () => JSON.parse(localStorage.getItem('maestro-studio-title-draft-v1') || 'null')?.brief === '',
      'empty Studio brief reaches persistent title storage');
    const emptyStudioDraft = await page.evaluate(() =>
      JSON.parse(localStorage.getItem('maestro-studio-title-draft-v1') || 'null'));
    assert.equal(emptyStudioDraft.text, '');
    assert.equal(emptyStudioDraft.look, '');
    assert.equal(emptyStudioDraft.direction, '');
    assert.deepEqual(emptyStudioDraft.phrases || [], []);
    assert.equal(emptyStudioDraft.titleReviewRequired, false);
    assert.equal(emptyStudioDraft.plannedKey, undefined);
    assert.deepEqual(emptyStudioDraft.plannerWarnings || [], []);
    assert.equal(emptyStudioDraft.rawTitleDraft, undefined);
    assert.equal(emptyStudioDraft.titleDraftHistoryId, undefined);
    assert.equal(await page.getByTestId('title-plan-review').count(), 0);
    assert.equal(await page.getByText('Raw planner draft diagnostics', {exact: true}).count(), 0);
    const titleFooter = page.getByTestId('titles-generate-footer');
    const needInputs = titleFooter.getByRole('button', {name: 'Need inputs', exact: true});
    await needInputs.waitFor();
    assert.equal(await needInputs.isDisabled(), true, 'Studio cannot generate an empty title description.');
    const queueUnavailable = titleFooter.getByRole('button', {name: 'Add to queue unavailable', exact: true});
    assert.equal(await queueUnavailable.isDisabled(), true, 'Studio cannot queue an empty title description.');
    assert.equal(titlePlanRequests.length, plannerRequestsBeforeStudioClear,
      'Deleting the Studio brief makes no planner request.');
    assert.equal(generationRequests.length, 0, 'Deleting the Studio brief makes no generation request.');

    await chooseWorkflow(page, 'Frames');
    await chooseWorkflow(page, 'Titles');
    await waitFor(page, () => document.querySelector('[aria-label="Describe your title animation"]')?.value === '',
      'empty title draft after a Studio workflow round trip');
    assert.equal(await page.evaluate(() => window.store.getState().params._studio_title_settings), undefined,
      'A cleared title draft has no one-shot gallery settings to restore.');

    await chooseWorkflow(page, 'Frames');
    const staleModeTitleSettings = {
      brief: 'STALE per-mode title settings', text: 'STALE', look: 'Stale look', direction: 'Stale motion',
      mode: 'single', seconds: 8, aspect: '16:9', resolution: '480p', background: 'black',
    };
    await page.evaluate(settings => window.store.getState().setParam('_studio_title_settings', settings), staleModeTitleSettings);
    assert.equal(await page.evaluate(() => window.store.getState().savedParamsPerMode.video?._studio_title_settings?.brief),
      staleModeTitleSettings.brief, 'The regression fixture places stale title settings in the video mode snapshot.');
    await page.evaluate(() => window.store.getState().setGenerationMode('tools'));
    const persistedModeSettings = await page.evaluate(() => JSON.parse(localStorage.getItem('maestro_mode_settings') || 'null'));
    assert.ok(persistedModeSettings, 'The real per-mode settings snapshot was written to localStorage.');
    assert.equal(persistedModeSettings.savedParamsPerMode?.video?._studio_title_settings, undefined,
      'Per-mode preference snapshots strip stale Studio title settings.');

    await page.reload();
    await mount(page, bundle, css, {workflow: 'titles', workspace: WORKSPACE});
    await page.evaluate(async workspace => {
      await window.store.getState().loadModels();
      window.seedScenario({workflow: 'titles', workspace});
    }, WORKSPACE);
    await waitFor(page, () => document.querySelector('[aria-label="Describe your title animation"]')?.value === '',
      'empty title draft remains empty after browser reload');
    const reloadedTitleStorage = await page.evaluate(() => ({
      draft: JSON.parse(localStorage.getItem('maestro-studio-title-draft-v1') || 'null'),
      savedMode: JSON.parse(localStorage.getItem('maestro_mode_settings') || 'null'),
      currentTitleSettings: window.store.getState().params._studio_title_settings,
      savedVideoTitleSettings: window.store.getState().savedParamsPerMode.video?._studio_title_settings,
    }));
    assert.equal(reloadedTitleStorage.draft.brief, '', 'Reload uses the real, empty title draft from localStorage.');
    assert.equal(reloadedTitleStorage.draft.text, '');
    assert.equal(reloadedTitleStorage.draft.rawTitleDraft, undefined);
    assert.equal(reloadedTitleStorage.draft.titleDraftHistoryId, undefined);
    assert.equal(reloadedTitleStorage.savedMode.savedParamsPerMode?.video?._studio_title_settings, undefined);
    assert.equal(reloadedTitleStorage.currentTitleSettings, undefined,
      'Reload does not let a stale per-mode title object override the empty title draft.');
    assert.equal(reloadedTitleStorage.savedVideoTitleSettings, undefined);
    assert.equal(await page.getByTestId('titles-generate-footer')
      .getByRole('button', {name: 'Need inputs', exact: true}).isDisabled(), true);
    assert.equal(titlePlanRequests.length, plannerRequestsBeforeStudioClear,
      'Reload does not re-plan the cleared title.');
    assert.equal(generationRequests.length, 0, 'Reload does not submit a title generation.');
    await page.evaluate(() => window.store.getState().setEnhanceOnGeneration(true));
    oneShotEnhancementRevision = await page.evaluate(() => window.store.getState().enhanceOnGenerationRevision);
    await page.getByLabel('Describe your title animation').fill('Keep a fresh title description for the remaining Studio workflow checks.');
    await waitFor(page, text => JSON.parse(localStorage.getItem('maestro-studio-title-draft-v1') || 'null')?.brief === text,
      'fresh title description stored for subsequent workflow checks', 10000,
      'Keep a fresh title description for the remaining Studio workflow checks.');
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
    assert.equal(await page.getByRole('region', {name: 'Studio Titles'}).count(), 1,
      'The Avatar detour presents the dedicated title composer while active.');
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
    assert.equal(after.model, TITLE_MODEL, 'Titles runs with its active video model.');
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
    await chooseWorkflow(page, 'Animate');
    assert.deepEqual(await stateInputs(page), before, 'Returning from Titles restores Viggle model, prompt and reference.');
    await chooseWorkflow(page, 'Frames');
    after = await stateInputs(page);
    assert.equal(after.prompt, before.prompt);
    assert.deepEqual(after.imageRefs, before.imageRefs);
    assert.deepEqual(after.referenceFiles, before.referenceFiles);
    assert.notEqual(after.model, 'viggle_animate', 'Leaving Animate restores the regular video model.');
    const ordinaryFormat = await page.evaluate(() => {
      const state = window.store.getState();
      return {resolution: state.resolutionPreset, aspect: state.aspectRatio,
        duration: state.durationSeconds, resolutionParam: state.params.resolution,
        frames: state.params.video_length};
    });

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

    assert.equal(generationRequests.length, 0, 'Changing format never submits a generation.');
    await chooseWorkflow(page, 'Frames');
    assert.deepEqual(await page.evaluate(() => {
      const state = window.store.getState();
      return {resolution: state.resolutionPreset, aspect: state.aspectRatio,
        duration: state.durationSeconds, resolutionParam: state.params.resolution,
        frames: state.params.video_length};
    }), ordinaryFormat, 'Leaving Titles restores the ordinary Studio model format and frame count.');
    await chooseWorkflow(page, 'Titles');
    await titleFormats.getByRole('button', {name: 'Resolution: 540p', exact: true}).waitFor();
    await titleFormats.getByRole('button', {name: 'Aspect ratio: 4:3', exact: true}).waitFor();
    assert.match(await titleFormats.getByRole('button', {name: /^Duration:/}).getAttribute('aria-label'), /10\.1/,
      'A native duration and the added format choices survive closing and reopening Titles.');
    await page.locator('[data-testid="studio-body-scroll"]').evaluate(element => { element.scrollTop = 0; });
    await page.screenshot({path: path.join(SCREENSHOTS, 'studio-titles-format-desktop.png'), fullPage: true});
    const advancedDialog = await openTitleAdvanced(page);
    const sectionNames = await advancedDialog.locator('details[data-testid^="advanced-"]:not([hidden]) summary').allTextContents();
    for (const section of ['LoRAs & presets', 'Performance', 'Finishing', 'Generation']) {
      assert.ok(sectionNames.some(text => text.trim().startsWith(section)), `Shared Advanced includes ${section}.`);
    }
    assert.doesNotMatch(await advancedDialog.innerText(), /single\s+title|phrase\s+sequence|use\s+manual\s+title\s+plan/i,
      'The shared Studio Advanced dialog contains no title mode or manual-plan controls.');

    const loras = advancedDialog.locator('[data-testid="advanced-loras"]');
    if (!await advancedDialog.getByTitle('Default settings').isVisible()) await loras.locator('summary').click();
    await advancedDialog.getByTitle('Default settings').waitFor();
    await advancedDialog.getByTitle(`File: ${TEST_LORA}`, {exact: true}).click();
    await waitFor(page, filename => window.store.getState().params.activated_loras.includes(filename),
      'Studio title LoRA activation', 10000, TEST_LORA);
    await waitFor(page, filename => window.store.getState().loraWeights[filename]?.[0] === 0.75,
      'recommended title LoRA weight', 10000, TEST_LORA);
    const loraWeight = loras.locator('input[type="range"]:visible').first();
    await loraWeight.waitFor();
    await setRangeValue(loraWeight, 0.65);
    await waitFor(page, filename => window.store.getState().loraWeights[filename]?.[0] === 0.65,
      'title LoRA weight from its Advanced slider', 10000, TEST_LORA);

    const performance = advancedDialog.locator('[data-testid="advanced-performance"]');
    if (!await performance.locator('input[type="checkbox"]').first().isVisible()) await performance.locator('summary').click();
    const slaToggle = performance.locator('label').filter({hasText: 'SLA Sparse Attention'}).locator('input[type="checkbox"]');
    await slaToggle.check();
    const cacheToggle = performance.locator('label').filter({hasText: 'First Block Cache'}).locator('input[type="checkbox"]');
    await cacheToggle.check();
    await performance.locator('select').first().selectOption('0.12');
    const warmup = performance.locator('input[type="range"]:visible').first();
    await setRangeValue(warmup, 35);
    await waitFor(page, () => window.store.getState().params.override_attention === 'sla'
      && window.store.getState().params.skip_steps_cache_type === 'first_block'
      && window.store.getState().params.skip_steps_multiplier === 0.12
      && window.store.getState().params.skip_steps_start_step_perc === 35,
    'title attention and First Block Cache settings');

    const finishing = advancedDialog.locator('[data-testid="advanced-finishing"]');
    const spatialUpsampling = finishing.getByText('Spatial Upsampling', {exact: true}).locator('..').locator('select');
    if (!await spatialUpsampling.isVisible()) await finishing.locator('summary').click();
    await spatialUpsampling.selectOption('lanczos2');
    await waitFor(page, () => window.store.getState().spatialUpsampling === 'lanczos2',
      'title finishing choice');

    const generation = advancedDialog.locator('[data-testid="advanced-generation"]');
    const h3Steps = generation.getByText('Inference Steps', {exact: true}).locator('..').locator('input[type="number"]');
    const seed = generation.getByText('Seed', {exact: true}).locator('..').locator('..').locator('input[type="number"]');
    if (!await h3Steps.isVisible()) await generation.locator('summary').click();
    assert.equal(await h3Steps.inputValue(), '4', 'The shared control reflects the selected H3 4-step model default.');
    assert.equal(await h3Steps.isDisabled(), false, 'The shared H3 step control follows the active model options.');
    await h3Steps.fill('17');
    await seed.fill('314159');
    await waitFor(page, () => window.store.getState().params.num_inference_steps === 17
      && window.store.getState().params.seed === 314159, 'title generation steps and seed');
    await page.screenshot({path: path.join(SCREENSHOTS, 'studio-title-standard-advanced.png'), fullPage: true});
    await page.keyboard.press('Escape');

    // Native title duration remains bounded and separate from ordinary Studio format.
    await titleFormats.getByRole('button', {name: /^Duration:/}).click();
    const durationDialog = page.getByRole('dialog', {name: 'Title duration', exact: true});
    await durationDialog.getByLabel('Native title duration preset', {exact: true}).selectOption('209');
    await page.keyboard.press('Escape');
    await titleFormats.getByRole('button', {name: /^Aspect ratio:/}).click();
    await page.getByRole('menu', {name: 'Aspect ratio', exact: true})
      .getByRole('menuitemradio', {name: '9:16', exact: true}).click();
    await titleFormats.getByRole('button', {name: /^Resolution:/}).click();
    await page.getByRole('menu', {name: 'Resolution', exact: true})
      .getByRole('menuitemradio', {name: /720p/}).click();
    await titleFormats.getByRole('button', {name: /^Resolution:/}).click();
    await page.getByRole('menu', {name: 'Resolution', exact: true})
      .getByRole('menuitemradio', {name: /480p/}).click();
    await page.setViewportSize({width: 1360, height: 700});
    await page.waitForTimeout(40);
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
    await page.setViewportSize({width: 1360, height: 900});
    const requestedDimensions = await page.evaluate(() => window.titleHelpers.titleDimensions({
      text: 'Maestro After Dark', look: 'chrome', direction: 'fly inward', effect: 'custom',
      seconds: 8.4, aspect: '9:16', resolution: '480p', keepAudio: false, background: 'black',
    }, {width: 1920, height: 1080}));
    assert.deepEqual(requestedDimensions, {width: 480, height: 832});
    await page.locator('[data-testid="studio-body-scroll"]').evaluate(element => { element.scrollTop = 0; });
    await page.screenshot({path: path.join(SCREENSHOTS, 'studio-titles-desktop.png'), fullPage: true});

    // Simulate a lost HTTP response after acceptance. The client performs one exact lookup,
    // adopts the accepted job into the ordinary queue, and never posts again after reload.
    const generationButton = sidebar.getByRole('button', {name: 'Generate'});
    const plansBeforeGeneration = titlePlanRequests.length;
    await generationButton.dblclick({timeout: 2000}).catch(() => {});
    await waitForNode(() => generationRequests.length > 0 || titlePlanRequests.length > plansBeforeGeneration,
      'title generation submission or AI-plan review');
    if (generationRequests.length === 0) {
      assert.equal(titlePlanRequests.length, plansBeforeGeneration + 1,
        'A stale creative plan returns to the AI review flow before submission.');
      assert.equal(await page.evaluate(() => window.titleStore.getState().intent), null,
        'An AI-plan review pause still creates no generation intent.');
      await generationButton.click();
      await waitForNode(() => generationRequests.length > 0, 'accepted reviewed title submission');
    }
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
    assert.equal(firstRequest.num_inference_steps, 17, 'The shared H3 inference-step edit reaches the title payload.');
    assert.equal(firstRequest.seed, 314159, 'The shared Advanced seed is included in the title payload.');
    assert.equal(firstRequest._studio_title_settings.generationParams.num_inference_steps, 17,
      'The Title payload preserves the shared inference-step value inside its settings snapshot.');
    assert.equal(firstRequest._studio_title_settings.generationParams.seed, 314159,
      'The Title payload keeps the same seed inside its immutable standard-settings snapshot.');
    assert.ok(firstRequest.activated_loras.includes(TEST_LORA), 'The shared LoRA selection reaches the title payload.');
    assert.match(String(firstRequest.loras_multipliers), /0\.65/, 'The chosen LoRA weight reaches the title payload.');
    assert.equal(firstRequest.override_attention, 'sla', 'The selected H3 attention setting reaches the title payload.');
    assert.equal(firstRequest.skip_steps_cache_type, 'first_block', 'First Block Cache activation reaches the title payload.');
    assert.equal(firstRequest.skip_steps_multiplier, 0.12, 'The chosen cache threshold reaches the title payload.');
    assert.equal(firstRequest.skip_steps_start_step_perc, 35, 'The chosen cache warmup reaches the title payload.');
    assert.equal(firstRequest.spatial_upsampling, 'lanczos2', 'The finishing choice reaches the title payload.');
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

    // An already-running title does not block adding multiple immutable held jobs.
    assert.equal(await page.evaluate(() => window.titleStore.getState().intent?.status), 'queued',
      'The first title remains active while queued titles are added.');
    const queueAction = sidebar.getByRole('button', {name: 'Add current Studio settings to the queue', exact: true});
    const setSeedBeforeQueue = async value => setStudioSeed(page, value);
    await setSeedBeforeQueue(271828);
    const queueOneIndex = generationRequests.length + 1;
    const lookupsBeforeLostQueueReceipt = receiptLookups.length;
    nextSubmissionMode = {status: 'queued', loseResponse: true};
    await queueAction.click();
    await waitFor(page, expectedId => window.store.getState().jobs.some(job => job.id === expectedId && job.status === 'held')
      && window.titleStore.getState().queueIntent === null,
    'exact lost-receipt recovery for held title one', 10000, `studio-title-job-${queueOneIndex}`);
    const queuedOne = generationRequests.at(-1);
    assert.equal(generationRequests.length, queueOneIndex, 'Recovering the lost held receipt does not duplicate its POST.');
    assert.equal(queuedOne._queue_mode, 'held', 'The shared queue action asks the backend to hold this title.');
    assert.equal(queuedOne.seed, 271828, 'The first held title snapshots its current shared seed.');
    assert.equal(queuedOne._studio_title_settings.generationParams.seed, 271828,
      'The held payload carries the same immutable shared-settings snapshot.');
    assert.equal(statuses.get(`studio-title-job-${queueOneIndex}`).status, 'held');
    assert.equal(receiptLookups.length, lookupsBeforeLostQueueReceipt + 1,
      'An ambiguous held receipt is looked up exactly once by its client submission ID.');
    assert.equal(receiptLookups.at(-1), queuedOne._client_submission_id);
    assert.ok(statusLookups.includes(`studio-title-job-${queueOneIndex}`),
      'The exact receipt is confirmed through the job status endpoint before queue adoption.');
    assert.equal(await page.evaluate(() => window.titleStore.getState().intent?.clientId), firstRequest._client_submission_id,
      'Adding a held title keeps the active generation intent intact.');

    await setSeedBeforeQueue(123456);
    const queueTwoIndex = generationRequests.length + 1;
    await queueAction.click();
    await waitFor(page, expectedId => window.store.getState().jobs.some(job => job.id === expectedId && job.status === 'held')
      && window.titleStore.getState().queueIntent === null,
    'second held title adoption while the first is active', 10000, `studio-title-job-${queueTwoIndex}`);
    const queuedTwo = generationRequests.at(-1);
    assert.equal(generationRequests.length, queueTwoIndex, 'Each held title is posted once.');
    assert.equal(queuedTwo._queue_mode, 'held');
    assert.equal(queuedTwo.seed, 123456, 'The second held title captures the later shared seed.');
    assert.equal(queuedOne.seed, 271828, 'Changing shared settings after the first POST does not mutate its payload snapshot.');
    assert.deepEqual(generationRequests.slice(1).map(body => body._queue_mode), ['held', 'held'],
      'Both Add to queue actions remain held and never start generation.');

    const generationQueueButton = page.getByRole('button', {name: /^Generation queue,/});
    await generationQueueButton.click();
    const generationQueue = page.getByRole('dialog', {name: 'Generation queue', exact: true});
    await generationQueue.getByRole('button', {name: 'Start queue', exact: true}).click();
    const heldIds = [`studio-title-job-${queueOneIndex}`, `studio-title-job-${queueTwoIndex}`];
    await waitFor(page, ids => ids.every(id => window.store.getState().jobs.find(job => job.id === id)?.status === 'queued'),
      'normal Studio Start queue releases the held titles', 10000, heldIds);
    assert.equal(startQueueRequests.length, 1, 'Start queue calls the normal mocked Studio queue endpoint once.');
    assert.equal(generationRequests.length, queueTwoIndex, 'Releasing held jobs does not issue another generation POST.');

    const nativeTitleOutput = `/outputs/${WORKSPACE}/title-black-480x832-native.mp4`;
    const processedTitleOutput = `/outputs/${WORKSPACE}/title-black-960x1664.mp4`;
    statuses.set('studio-title-job-1', jobStatus('studio-title-job-1', firstRequest._client_submission_id,
      'completed', [nativeTitleOutput, processedTitleOutput]));
    currentOutput = [
      {id: 'title-gallery-entry-native', name: 'title-black-480x832-native.mp4',
        type: 'video', mode: 'video', size: 1234, created_at: 101, url: '/outputs/title-black-480x832-native.mp4',
        path: nativeTitleOutput, workspace: WORKSPACE},
      {id: 'title-gallery-entry-processed', name: 'title-black-960x1664.mp4',
        type: 'video', mode: 'video', size: 2345, created_at: 100, url: '/outputs/title-black-960x1664.mp4',
        path: processedTitleOutput, workspace: WORKSPACE},
    ];
    const probesBeforeComplete = probes.length;
    const refreshesBeforeComplete = outputRefreshes.length;
    await page.evaluate(() => window.titleStore.getState().reconcile());
    await waitFor(page, () => window.titleStore.getState().intent?.status === 'completed',
      'completed title verification');
    await waitFor(page, () => window.store.getState().outputs.some(output => output.name === 'title-black-960x1664.mp4'),
      'normal gallery refresh');
    assert.ok(outputRefreshes.length > refreshesBeforeComplete, 'Completion refreshes the ordinary gallery.');
    assert.deepEqual(probes.slice(probesBeforeComplete).map(probe => probe.asset?.path).sort(),
      [nativeTitleOutput, processedTitleOutput].sort(),
      'Studio completion probes every native and processed video in a multi-output result.');
    assert.ok(await page.evaluate(() => window.store.getState().outputs.some(output => output.name === 'title-black-480x832-native.mp4')),
      'The ordinary gallery retains the native output alongside its processed video.');
    assert.equal(probes.at(-1).workspace, WORKSPACE, 'Media verification uses the frozen title workspace.');
    assert.equal(await page.evaluate(() => window.store.getState().activeWorkspace), 'Changed While Running');
    assert.equal(await page.evaluate(() => window.editorStore.getState().project), null,
      'Completion leaves the Editor project unchanged.');

    // Failed and cancelled jobs report errors without a success state. Use green on the failure case.
    await page.evaluate(() => window.titleStore.getState().dismiss());
    await page.evaluate(() => window.store.getState().setEnhanceOnGeneration(true));
    const prepareFailureRevision = await page.evaluate(() => window.store.getState().enhanceOnGenerationRevision);
    failNextTitleUpload = true;
    await page.getByRole('button', {name: 'Generate'}).click();
    await waitFor(page, () => window.titleStore.getState().error?.includes('Upload failed'),
      'title upload preparation failure');
    assert.equal(uploadFailures.length, 1, 'The title background upload failure is synthetic and isolated.');
    assert.equal(generationRequests.length, queueTwoIndex, 'A title preparation failure never sends a GPU generation request.');
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

    await page.evaluate(() => window.store.getState().setEnhanceOnGeneration(true));
    const raceRevision = await page.evaluate(() => window.store.getState().enhanceOnGenerationRevision);
    const racedGeneration = {started: deferred(), release: deferred()};
    generationGate = racedGeneration;
    await page.getByRole('button', {name: 'Generate'}).click();
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
    assert.equal(generationRequests.length, queueTwoIndex + 1);
    assert.equal(generationRequests.at(-1)._studio_title_settings.background, 'black');
    assert.match(uploadRequests.at(-1).body, /title-black-/);
    assert.equal(await page.getByRole('alert').filter({hasText: 'Synthetic generation failure'}).count(), 1);
    assert.equal(await page.getByRole('button', {name: 'Dismiss', exact: true}).count(), 0,
      'The title sidebar does not show the old status banner actions.');

    await page.evaluate(() => window.titleStore.getState().dismiss());
    nextSubmissionMode = {status: 'cancelled', loseResponse: false};
    await page.getByRole('button', {name: 'Generate'}).click();
    await waitFor(page, () => window.titleStore.getState().intent?.status === 'failed'
      && window.titleStore.getState().intent?.error === 'Synthetic cancellation', 'cancelled title generation');
    assert.equal(generationRequests.length, queueTwoIndex + 2);
    assert.equal(await page.getByRole('alert').filter({hasText: 'Synthetic cancellation'}).count(), 1);
    assert.equal(await page.getByRole('link', {name: 'Preview', exact: true}).count(), 0);
    assert.deepEqual(projectApiRequests, [], 'No Studio title path calls Editor project APIs.');

    // Phone layout keeps the title form scrollable while the keyboard is open, and the
    // anchored workflow chooser remains a non-modal popover without a focus trap.
    await page.setViewportSize({width: 390, height: 844});
    await page.waitForTimeout(100);
    await assertHardwareExpandable(page, 'Mobile');
    if (await page.getByRole('dialog', {name: 'Advanced settings', exact: true}).isVisible()) await page.keyboard.press('Escape');
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
    const mobileAdvanced = page.getByRole('dialog', {name: 'Advanced settings', exact: true});
    const mobileAdvancedSections = [
      ['advanced-loras', advancedDialog.getByTitle('Default settings')],
      ['advanced-performance', page.locator('[data-testid="advanced-performance"] input[type="checkbox"]').first()],
      ['advanced-finishing', page.locator('[data-testid="advanced-finishing"]').getByText('Spatial Upsampling', {exact: true}).locator('..').locator('select')],
      ['advanced-generation', page.locator('[data-testid="advanced-generation"]').getByText('Seed', {exact: true})],
    ];
    for (const [id, control] of mobileAdvancedSections) {
      const section = mobileAdvanced.locator(`[data-testid="${id}"]`);
      if (!await control.isVisible()) await section.locator('summary').click();
    }
    assert.doesNotMatch(await mobileAdvanced.innerText(), /single\s+title|phrase\s+sequence|use\s+manual\s+title\s+plan/i);
    await page.screenshot({path: path.join(SCREENSHOTS, 'studio-title-standard-advanced-mobile.png'), fullPage: true});
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

    await page.setViewportSize({width: 1360, height: 900});
    await page.evaluate(() => window.titleStore.getState().dismiss());

    // Controlled Frames catalog, native timing/defaults, and all chosen background colors.
    await page.keyboard.press('Escape');
    await chooseWorkflow(page, 'Frames');
    const normalBeforeModel = await stateInputs(page);
    await chooseWorkflow(page, 'Titles');
    await page.evaluate(() => {
      window.titleStore.getState().dismiss();
      window.store.setState(state => ({enhanceOnGeneration: false, enhanceOnGenerationDefault: false,
        params: {...state.params, _studio_title_settings: {
          brief: 'Show “COLOR” with a clean reveal, readable hold, then dissolve.', text: '', look: '', direction: '',
          effect: 'custom', seconds: 5.2, aspect: '16:9', resolution: '480p', background: 'white', keepAudio: false,
        }}}));
    });
    await page.getByRole('button', {name: 'Choose model', exact: true}).click();
    await page.getByRole('dialog', {name: 'Choose a model', exact: true}).getByTitle('Test Frames T2V', {exact: true}).click();
    await waitFor(page, () => document.querySelector('[aria-label="Choose model"]')?.dataset.modelType === 'test_frames_text', 'controlled T2V title model');
    await waitFor(page, () => JSON.parse(localStorage.getItem('maestro-studio-title-draft-v1') || 'null')?.model?.modelType === 'test_frames_text',
      'T2V profile reaches the title payload snapshot');
    const modelAdvanced = await openTitleAdvanced(page);
    const textGeneration = modelAdvanced.locator('[data-testid="advanced-generation"]');
    const textSteps = textGeneration.getByText('Inference Steps', {exact: true}).locator('..').locator('input[type="number"]');
    const textGuidance = textGeneration.getByText('Guidance Scale', {exact: true}).locator('..').locator('input[type="number"]');
    if (!await textSteps.isVisible()) await textGeneration.locator('summary').click();
    assert.equal(await textSteps.inputValue(), '12', 'T2V loads its own inference-step default in the shared panel.');
    assert.equal(await textGuidance.inputValue(), '2.5', 'T2V loads its own guidance default in the shared panel.');
    await textSteps.fill('15');
    await textGuidance.fill('3.1');
    await setStudioSeed(page, 314159);
    await waitFor(page, () => window.store.getState().params.num_inference_steps === 15
      && window.store.getState().params.guidance_scale === 3.1, 'editable shared title generation settings');
    await page.screenshot({path: path.join(SCREENSHOTS, 'studio-title-model-advanced.png'), fullPage: true});
    if (await page.getByRole('dialog', {name: 'Advanced settings', exact: true}).isVisible()) await page.keyboard.press('Escape');
    nextSubmissionMode = {status: 'completed', loseResponse: false};
    const beforeTextUploads = uploadRequests.length;
    const beforeTextRequests = generationRequests.length;
    await page.getByRole('button', {name: 'Generate', exact: true}).click();
    await waitForNode(() => generationRequests.length > beforeTextRequests, 'new T2V title generation POST');
    const textBody = generationRequests.at(-1);
    const t2vContext = await page.evaluate(() => ({
      paramsModelType: window.store.getState().params.model_type,
      modelOptionsType: window.store.getState().modelOptions?.model_type,
      draftModel: JSON.parse(localStorage.getItem('maestro-studio-title-draft-v1') || 'null')?.model,
      intent: (() => {
        const intent = window.titleStore.getState().intent;
        return intent && {status: intent.status, error: intent.error, settingsModel: intent.settings?.model};
      })(),
      queueIntent: window.titleStore.getState().queueIntent,
    }));
    assert.equal(textBody.model_type, 'test_frames_text',
      `The new T2V submission uses its selected profile: ${JSON.stringify({
        ...t2vContext, requestModelType: textBody.model_type,
        payloadModel: textBody._studio_title_settings?.model,
        workflow: textBody._studio_video_workflow,
      })}`);
    await waitFor(page, () => window.titleStore.getState().intent?.status === 'completed'
      && window.titleStore.getState().intent?.settings.model?.modelType === 'test_frames_text', 'text-model title completion');
    assert.equal(textBody.model_type, 'test_frames_text');
    assert.equal(textBody._studio_title_settings.model.fps, 16);
    assert.equal(textBody.video_length % 4, 1, 'The text model uses its own native frame lattice.');
    assert.ok(textBody.video_length <= 161, 'T2V titles stay within the model’s native frame cap.');
    assert.equal(textBody.num_inference_steps, 15);
    assert.equal(textBody.guidance_scale, 3.1);
    assert.equal(textBody.seed, 314159);
    assert.equal(textBody.sample_solver, 'euler');
    assert.equal(textBody.multi_prompts_gen_type, 2, 'A multiline title prompt always remains one clip.');
    assert.ok(!textBody.image_start && !textBody.image_end && !textBody.image_prompt_type);
    assert.notEqual(textBody.override_attention, 'sla', 'Foreign models never inherit the Fused SLA override.');
    assert.match(textBody.prompt, /#FFFFFF/i);
    assert.equal(uploadRequests.length, beforeTextUploads, 'A text-only model is not sent unsupported boundary images.');
    assert.equal(await page.getByRole('button', {name: 'Dismiss', exact: true}).count(), 0);
    assert.equal(await page.getByRole('link', {name: 'Preview', exact: true}).count(), 0);
    const beforeInvalidState = await page.evaluate(() => {
      const {intent, queueIntent} = window.titleStore.getState();
      return {
        activeModel: window.store.getState().params.model_type,
        optionsModel: window.store.getState().modelOptions?.model_type,
        intent: intent && {id: intent.id, clientId: intent.clientId, status: intent.status,
          error: intent.error, settingsModel: intent.settings?.model},
        queueIntent,
      };
    });
    const priorTextClientId = beforeInvalidState.intent?.clientId;
    const beforeInvalidTextRequests = generationRequests.length;
    const invalidTextSettings = {...textBody._studio_title_settings,
      generationParams: {...textBody._studio_title_settings.generationParams, num_inference_steps: 0}};
    invalidTextSettings.plannedKey = await page.evaluate(settings => window.titleHelpers.titlePlanningKey(settings), invalidTextSettings);
    await page.evaluate(settings => window.titleStore.getState().submitStudio(settings, 'now'), invalidTextSettings);
    await waitFor(page, () => window.titleStore.getState().error?.toLowerCase().includes('inference step'),
      'invalid T2V steps validation');
    assert.equal(generationRequests.length, beforeInvalidTextRequests,
      'Invalid shared Advanced settings never POST a generation.');
    assert.equal(await page.evaluate(() => window.titleStore.getState().intent?.clientId), priorTextClientId,
      'Invalid steps do not replace the prior completed intent with a durable new intent.');
    const afterInvalidState = await page.evaluate(() => {
      const {intent, queueIntent} = window.titleStore.getState();
      return {
        activeModel: window.store.getState().params.model_type,
        optionsModel: window.store.getState().modelOptions?.model_type,
        intent: intent && {id: intent.id, clientId: intent.clientId, status: intent.status,
          error: intent.error, settingsModel: intent.settings?.model},
        queueIntent,
      };
    });
    assert.equal(afterInvalidState.activeModel, beforeInvalidState.activeModel,
      'Rejected invalid settings do not change the selected Studio model.');
    assert.equal(afterInvalidState.optionsModel, beforeInvalidState.optionsModel,
      'Rejected invalid settings do not change the active model options.');
    assert.deepEqual(afterInvalidState.intent, beforeInvalidState.intent,
      'Rejected invalid settings preserve the prior intent status, error, and model snapshot.');
    assert.deepEqual(afterInvalidState.queueIntent, beforeInvalidState.queueIntent,
      'Rejected invalid settings do not mutate a held queue receipt.');
    nextSubmissionMode = {status: 'completed', loseResponse: false};
    await page.getByRole('button', {name: 'Generate', exact: true}).click();
    await waitForNode(() => generationRequests.length > beforeInvalidTextRequests,
      'corrected T2V settings can be submitted after invalid-step rejection');
    await waitFor(page, id => window.titleStore.getState().intent?.status === 'completed'
      && window.titleStore.getState().intent?.clientId !== id,
    'corrected T2V retry completes', 10000, priorTextClientId);
    assert.equal(generationRequests.at(-1).num_inference_steps, 15,
      'The valid shared Advanced value remains available for a corrected retry.');
    await page.getByRole('button', {name: 'Choose model', exact: true}).click();
    await page.getByRole('dialog', {name: 'Choose a model', exact: true}).getByTitle('Test Frames I2V', {exact: true}).click();
    await waitFor(page, () => document.querySelector('[aria-label="Choose model"]')?.dataset.modelType === 'test_frames_image', 'controlled I2V title model');
    await page.getByLabel('Title background', {exact: true}).first().selectOption('blue');
    nextSubmissionMode = {status: 'completed', loseResponse: false};
    await page.getByRole('button', {name: 'Generate', exact: true}).click();
    await waitFor(page, () => window.titleStore.getState().intent?.status === 'completed'
      && window.titleStore.getState().intent?.settings.background === 'blue', 'blue I2V title completion');
    const imageBody = generationRequests.at(-1);
    assert.equal(imageBody.model_type, 'test_frames_image');
    assert.equal(imageBody.num_inference_steps, 8, 'I2V loads the image model’s own default inference steps.');
    assert.ok(imageBody.video_length <= 241 && imageBody.video_length % 8 === 1,
      'I2V titles respect its native frame cap and step lattice.');
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
    await page.getByRole('button', {name: 'Generate', exact: true}).click();
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
    const longStudioBrief = 'Show “ORBIT” on black, then transform its lettering.\n'
      + 'Keep the lettering flat, readable and inside the frame throughout the animation.\n'.repeat(160);
    assert.ok(longStudioBrief.length > 9000 && longStudioBrief.length < 16000);
    assert.equal(await page.getByLabel('Describe your title animation').getAttribute('maxlength'), '16000');
    await page.getByLabel('Describe your title animation').fill(longStudioBrief);
    await page.evaluate(() => window.store.setState({enhanceOnGeneration: false, enhanceOnGenerationDefault: false}));
    const beforeLongBriefRequests = generationRequests.length;
    nextSubmissionMode = {status: 'completed', loseResponse: false};
    await page.getByRole('button', {name: 'Generate', exact: true}).click();
    await waitForNode(() => generationRequests.length === beforeLongBriefRequests + 1, 'long title generation POST');
    await waitFor(page, brief => window.titleStore.getState().intent?.status === 'completed'
      && window.titleStore.getState().intent?.settings.brief === brief, 'complete long title description', 10000, longStudioBrief);
    assert.ok(generationRequests.at(-1).prompt.includes(`User description:\n${longStudioBrief}`));
    assert.equal(generationRequests.at(-1)._studio_title_settings.brief, longStudioBrief);
    await chooseWorkflow(page, 'Frames');
    await chooseWorkflow(page, 'Titles');
    assert.equal(await page.getByLabel('Describe your title animation').inputValue(), longStudioBrief,
      'A long title description survives leaving and reopening Titles without truncation.');
    await waitFor(page, () => {
      const saved = JSON.parse(localStorage.getItem('maestro-studio-title-draft-v1'));
      return saved.plannedKey.length > 9000 && window.titleHelpers.titlePlanIsCurrent(saved);
    }, 'long title plan signature survives draft normalization');
    await chooseWorkflow(page, 'Frames');
    assert.deepEqual(await stateInputs(page), normalBeforeModel,
      'Leaving Titles restores the ordinary model, prompt, references, voices and Advanced settings.');
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
