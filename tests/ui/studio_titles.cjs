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
];
const models = [
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
        import {TITLE_EFFECT_LABELS, titleDimensions, titleFrames, titleSequenceTimeline, newTitlePhrases, animatedTitlePrompt, animatedTitleParams} from './src/editor/editorAnimatedTitles';
        window.store = useStore;
        window.editorStore = useEditorStore;
        window.titleStore = useEditorTitleStore;
        window.titleHelpers = {TITLE_EFFECT_LABELS, titleDimensions, titleFrames, titleSequenceTimeline, newTitlePhrases, animatedTitlePrompt, animatedTitleParams};
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
    throw new Error(`Timed out waiting for ${description}: ${error.message}`);
  }
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
  assert.equal(await page.evaluate(() => window.editorStore.getState().project), null,
    'The Studio panel renders without an Editor project.');
}

(async () => {
  const css = appCss();
  const bundle = await bundleUi();
  const requestLog = [];
  const generationRequests = [];
  const uploadRequests = [];
  const receiptLookups = [];
  const statusLookups = [];
  const probes = [];
  const preferenceUpdates = [];
  const outputRefreshes = [];
  const projectApiRequests = [];
  const receipts = new Map();
  const statuses = new Map();
  const submissionModes = [];
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
      if (endpoint.startsWith('/api/v1/model-options/')) return json(modelOptions);
      if (endpoint === '/api/v1/studio-preferences' && method === 'PUT') {
        preferenceUpdates.push(request.postDataJSON());
        return json({generation_mode: 'video', studio_video_workflow: 'titles'});
      }
      if (endpoint === '/api/v1/studio-preferences') {
        return json({generation_mode: 'video', studio_video_workflow: 'frames', selected_model_per_mode: {video: TITLE_MODEL}});
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
        uploadRequests.push({search: url.search, contentType: request.headers()['content-type'] || '', body});
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
        if (mode.loseResponse) return route.abort('failed');
        return json({job_id: jobId, status: 'queued'});
      }
      if (endpoint === '/api/v1/editor/media/probe') {
        const body = request.postDataJSON();
        probes.push(body);
        const asset = body.asset || {};
        const dimensions = /(\d+)x(\d+)/.exec(asset.path || asset.name || '');
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
    await waitFor(page, text => document.querySelector('[aria-label="Animated title text"]')?.value === text,
      'gallery title settings restored into the Studio form', 10000, 'Loaded from gallery');
    await page.getByLabel('Animated title text').fill('Edited loaded gallery title');
    await waitFor(page, () => window.store.getState().params._studio_title_settings === undefined,
      'one-shot gallery title settings cleared after the first edit');
    await chooseWorkflow(page, 'Frames');
    assert.deepEqual(await stateInputs(page), before, 'Returning to Frames restores its normal prompt and references.');
    assert.equal(await page.evaluate(() => window.store.getState().params._studio_title_settings), undefined,
      'Frames does not inherit one-shot title settings from its stored prompt snapshot.');
    await chooseWorkflow(page, 'Titles');
    await waitFor(page, text => document.querySelector('[aria-label="Animated title text"]')?.value === text,
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
    assert.match(await sidebar.innerText(), new RegExp(`Save to ${WORKSPACE}`));
    const presetSummary = sidebar.getByText('Use an animation preset (optional)', {exact: true});
    await presetSummary.click();
    const preset = sidebar.getByLabel('Animation preset');
    const presetCount = await preset.locator('option').count();
    assert.equal(presetCount, 21, 'The custom option and all 20 animation presets are available.');
    await preset.selectOption('dust');
    await sidebar.getByLabel('Animated title text').fill('Maestro After Dark');
    await sidebar.getByLabel('Title background').selectOption('black');
    await sidebar.getByLabel('Title visual style').fill('Bright chrome lettering with a restrained blue rim light');
    await sidebar.getByLabel('Title animation description').fill('The chrome words fly inward, hold, then fade to black.');
    await sidebar.locator('label').filter({hasText: 'Duration (seconds)'}).locator('input[type="number"]').fill('8.4');
    await sidebar.locator('label').filter({hasText: 'Aspect ratio'}).locator('select').selectOption('9:16');
    await sidebar.locator('label').filter({hasText: 'Resolution'}).locator('select').selectOption('720p');
    assert.equal(await preset.inputValue(), 'custom',
      'Editing a preset description switches the title to its custom motion field.');
    const dimensions720 = await sidebar.getByText(/704×1280/).count();
    assert.equal(dimensions720, 1, 'Portrait 720p title dimensions are shown before submit.');
    await sidebar.locator('label').filter({hasText: 'Resolution'}).locator('select').selectOption('480p');
    assert.equal(await sidebar.getByText(/480×832/).count(), 1,
      'Portrait 480p title dimensions use the aligned output size.');
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
    nextSubmissionMode = {status: 'failed', loseResponse: false};
    await page.getByLabel('Title background').selectOption('chroma');
    await page.getByRole('button', {name: 'Generate title'}).click();
    await waitFor(page, () => window.titleStore.getState().intent?.status === 'failed',
      'failed title generation');
    assert.equal(generationRequests.length, 2);
    assert.equal(generationRequests[1]._studio_title_settings.background, 'chroma');
    assert.match(uploadRequests[1].body, /title-green-/);
    assert.match(await page.locator('[role="status"]').innerText(), /Synthetic generation failure/);
    assert.doesNotMatch(await page.locator('[role="status"]').innerText(), /Title saved to/);

    await page.evaluate(() => window.titleStore.getState().dismiss());
    nextSubmissionMode = {status: 'cancelled', loseResponse: false};
    await page.getByRole('button', {name: 'Generate title'}).click();
    await waitFor(page, () => window.titleStore.getState().intent?.status === 'failed'
      && window.titleStore.getState().intent?.error === 'Synthetic cancellation', 'cancelled title generation');
    assert.equal(generationRequests.length, 3);
    assert.match(await page.locator('[role="status"]').innerText(), /Synthetic cancellation/);
    assert.doesNotMatch(await page.locator('[role="status"]').innerText(), /Title saved to/);
    assert.deepEqual(projectApiRequests, [], 'No Studio title path calls Editor project APIs.');

    // Phone layout keeps the title form scrollable while the keyboard is open, and the
    // anchored workflow chooser remains a non-modal popover without a focus trap.
    await page.setViewportSize({width: 390, height: 844});
    await page.waitForTimeout(100);
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
    await page.getByLabel('Title animation description').focus();
    await page.getByLabel('Title animation description').fill('Mobile title motion stays in the visible sidebar.');
    await page.setViewportSize({width: 390, height: 500});
    await waitFor(page, () => document.querySelector('.maestro-sidebar')?.getAttribute('data-keyboard-open') === 'true',
      'mobile keyboard viewport handling');
    const scroll = await page.locator('[data-testid="studio-body-scroll"]').evaluate(element => ({
      scrollHeight: element.scrollHeight, clientHeight: element.clientHeight,
      width: element.scrollWidth, clientWidth: element.clientWidth,
    }));
    assert.ok(scroll.scrollHeight > scroll.clientHeight, 'The mobile Studio Titles panel remains vertically scrollable.');
    assert.ok(scroll.width <= scroll.clientWidth, 'The mobile title form does not overflow horizontally.');

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
    await page.evaluate(settings => window.store.setState(state => ({params: {...state.params, _studio_title_settings: settings}})), sequenceSettings);
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
    assert.match(sequenceBody.prompt, /Keep the camera locked/);
    assert.equal(generationRequests.length, 4, 'One sequence sends one POST, without per-phrase jobs.');
    assert.equal(receiptLookups.at(-1), sequenceBody._client_submission_id);
    const beforeSequenceReload = generationRequests.length;
    await page.reload();
    await mount(page, bundle, css, {workflow: 'titles', workspace: 'Changed While Running'});
    await waitFor(page, () => window.titleStore.getState().intent?.jobId === 'studio-title-job-4', 'sequence receipt reload');
    assert.equal(await page.getByLabel('Phrase 1 text', {exact: true}).inputValue(), 'CREATE MORE');
    assert.equal(await page.getByLabel('Phrase 3 duration', {exact: true}).inputValue(), '4');
    assert.equal(generationRequests.length, beforeSequenceReload, 'Sequence reload never resubmits.');
    assert.deepEqual(projectApiRequests, [], 'Sequence generation remains independent of Editor projects.');

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
