// Exercise imported H3 routing and completed-download refresh with isolated APIs.
// Run: node tests/ui/h3_checkpoint_import.cjs
const fs = require('node:fs'), path = require('node:path'), vm = require('node:vm');
const assert = require('node:assert/strict');
const root = path.resolve(__dirname, '../..');
const ts = require(path.join(root, 'ui/node_modules/typescript'));

const checkpoint = 'civitai_h3_2830065_3193337_3074134';
const framesId = `${checkpoint}_frames`;
const referencesId = `${checkpoint}_references`;
const existingCheckpointId = 'civitai_h3_older_import_frames';
const imageId = 'flux-dev';

function importedPair() {
  const companions = {frames: framesId, references: referencesId};
  return [
    {
      model_type: framesId, name: 'Imported H3 — Frames', family: 'minimax_h3',
      architecture: 'minimax_h3', is_t2v: true, is_i2v: true,
      supports_end_frame: true, supports_audio_input: true, generates_audio: true,
      omni_reference: false, h3_companion_models: companions,
      minimax_h3_model_id: `${checkpoint}_frames`,
      minimax_h3_lora_workflow: 'ref2va',
      minimax_h3_import_profile: {native_workflow: 'ref2va', sampling_profile: 'turbo'},
      minimax_h3_baked_turbo: true, is_downloaded: true,
      fps: 24, frames_minimum: 124, frames_maximum: 345, frames_steps: 17,
      guidance_max_phases: 0,
    },
    {
      model_type: referencesId, name: 'Imported H3 — References', family: 'minimax_h3',
      architecture: 'minimax_h3_ref2va', is_t2v: true, is_i2v: true,
      supports_end_frame: false, supports_audio_input: true, generates_audio: true,
      omni_reference: true, h3_companion_models: companions,
      minimax_h3_model_id: `${checkpoint}_references`,
      minimax_h3_lora_workflow: 'ref2va',
      minimax_h3_import_profile: {native_workflow: 'ref2va', sampling_profile: 'turbo'},
      minimax_h3_baked_turbo: true, is_downloaded: true,
      fps: 24, frames_minimum: 124, frames_maximum: 345, frames_steps: 17,
      guidance_max_phases: 0,
    },
  ];
}

const pair = importedPair();
const optionById = Object.fromEntries(pair.map(model => [model.model_type, {
  ...model, default_num_inference_steps: 8, default_guidance_scale: 1,
  inference_steps_min: 4, inference_steps_max: 8,
  minimax_h3_unaccelerated_default_steps: 8,
  minimax_h3_baked_turbo: true,
  minimax_h3_turbo_mode_default: false,
  sol_attention: true, sol_attention_status: {supported: true},
}]));

function fixture({
  catalog = () => pair,
  options = optionById,
  visibility = {configured: true, defaults_version: 999, enabled_models: [], initialized_mature_models: []},
  studioPreferences = {configured: false},
  downloads = [],
} = {}) {
  const modules = new Map(), storage = new Map();
  const calls = {fetchModels: 0, reloadModels: 0, visibilityUpdates: [], submissions: []};
  const api = new Proxy({
    submitGeneration: async params => {calls.submissions.push({...params}); return {job_id: 'fixture', status: 'held'};},
    fetchModels: async () => {
      calls.fetchModels++;
      return {models: catalog(calls.fetchModels, calls.reloadModels), families: [
        {id: 'minimax_h3', label: 'MiniMax H3'}, {id: 'flux', label: 'Flux'},
      ]};
    },
    reloadModels: async () => { calls.reloadModels++; return {status: 'ok'}; },
    fetchModelVisibility: async () => visibility,
    updateModelVisibility: async value => { calls.visibilityUpdates.push(value); },
    fetchStudioPreferences: async () => studioPreferences,
    updateStudioPreferences: async () => ({}),
    fetchH3WindowOverrides: async () => ({overrides: {}}),
    fetchModelOptions: async modelType => options[modelType] || {},
    fetchDefaults: async modelType => ({
      num_inference_steps: options[modelType]?.default_num_inference_steps ?? 8,
      guidance_scale: 1,
      flow_shift: 12,
      audio_flow_shift: 3,
    }),
    fetchLoras: async () => ({loras: []}),
    fetchInstalledLoras: async () => ({loras: []}),
    fetchCivitAIDownloads: async () => ({downloads}),
  }, {get: (target, name) => name in target ? target[name] : async () => ({})});

  function create(init) {
    let state;
    const get = () => state;
    const set = update => {state = {...state, ...(typeof update === 'function' ? update(state) : update)};};
    state = init(set, get);
    return {getState: get, setState: set, subscribe: () => () => {}};
  }
  function load(file) {
    if (modules.has(file)) return modules.get(file).exports;
    const module = {exports: {}};
    modules.set(file, module);
    const code = ts.transpileModule(fs.readFileSync(file, 'utf8').replaceAll('import.meta', '({})'),
      {compilerOptions: {module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2022}}).outputText;
    vm.runInNewContext(code, {module, exports: module.exports, console, URL, AbortController, structuredClone,
      localStorage: {getItem: key => storage.get(key) || null, setItem: (key, value) => storage.set(key, value)},
      setTimeout: () => 1, clearTimeout: () => {}, setInterval: () => 1, clearInterval: () => {},
      window: {setTimeout: () => 1, clearTimeout: () => {}},
      require: id => id === 'zustand' ? {create} : id.endsWith('/api/client') ? api
        : id.endsWith('/lib/theme') ? {getStoredPrefs: () => ({family: 'default', mode: 'dark'}), applyThemePrefs: () => {}}
        : id.startsWith('.') ? load(path.resolve(path.dirname(file), `${id}.ts`))
        : (() => {throw Error(`Unexpected import ${id}`);})(),
    }, {filename: file});
    return module.exports;
  }
  const exported = load(path.join(root, 'ui/src/stores/useStore.ts'));
  return {
    store: exported.useStore, calls, storage,
    settle: async () => {for (let i = 0; i < 50; i++) await Promise.resolve();},
  };
}

function imageModel() {
  return {
    model_type: imageId, name: 'Flux Dev', family: 'flux', architecture: 'flux',
    is_image: true, is_t2v: false, is_i2v: false, is_downloaded: true,
    guidance_max_phases: 1, fps: 0,
  };
}

function oldImportedModel() {
  return {
    model_type: existingCheckpointId, name: 'Older imported H3', family: 'minimax_h3',
    architecture: 'minimax_h3', is_t2v: true, is_i2v: true,
    supports_end_frame: true, omni_reference: false, is_downloaded: true,
    guidance_max_phases: 0, fps: 24,
  };
}

async function waitUntil(predicate, label) {
  const deadline = Date.now() + 2500;
  while (Date.now() < deadline) {
    if (predicate()) return;
    await new Promise(resolve => setTimeout(resolve, 5));
  }
  assert.fail(`Timed out waiting for ${label}`);
}

async function testDynamicPairRoutingAndTurboStepMemory() {
  const f = fixture({options: optionById});
  const store = f.store;
  store.setState({
    models: pair,
    enabledModels: new Set([framesId, referencesId]),
    generationMode: 'video', studioVideoWorkflow: 'frames',
    studioVideoEffectiveCreateRoute: 'generate',
    studioVideoModelPerCreateRoute: {generate: framesId, omni: referencesId},
    params: {...store.getState().params, model_type: '', image_mode: 0,
      minimax_h3_references: [], minimax_h3_turbo_mode: false},
  });

  store.getState().selectStudioVideoModel(framesId);
  await f.settle();
  assert.equal(store.getState().params.model_type, framesId);
  assert.equal(store.getState().params.num_inference_steps, 8, 'Baked Turbo starts at the verified eight-step recipe');
  assert.equal(store.getState().params.minimax_h3_turbo_mode, false, 'Baked Turbo does not turn on a separate adapter');
  assert.equal(store.getState().modelOptions.minimax_h3_lora_workflow, 'ref2va');

  store.getState().setParam('num_inference_steps', 4);
  store.getState().setStudioVideoWorkflow('references');
  await f.settle();
  assert.equal(store.getState().params.model_type, referencesId,
    'Dynamic h3_companion_models routes Frames to its CivitAI References ID');
  assert.equal(store.getState().modelOptions.minimax_h3_lora_workflow, 'ref2va',
    'The References architecture keeps the checkpoint’s native Ref2VA LoRA basis');
  assert.equal(pair[0].minimax_h3_lora_workflow, pair[1].minimax_h3_lora_workflow,
    'Both imported workflow definitions advertise the same native LoRA basis');

  store.getState().setStudioVideoWorkflow('frames');
  await f.settle();
  assert.equal(store.getState().params.model_type, framesId,
    'The dynamic companion mapping routes References back to the same checkpoint’s Frames ID');
  assert.equal(store.getState().params.num_inference_steps, 4,
    'The user’s four-step selection is restored after switching workflows');
  assert.equal(store.getState().modelOptions.minimax_h3_lora_workflow, 'ref2va');
}

async function testCompletedCheckpointRefreshPreservesUserStateAndOptOuts() {
  const nowSeconds = Math.floor(Date.now() / 1000);
  const completedDownloads = [
    {
      id: 'older-completed-download', filename: 'old-h3.safetensors',
      status: 'completed', progress: 1, bytes_downloaded: 100, bytes_total: 100,
      error: null, started_at: nowSeconds - 300, completed_at: nowSeconds - 120,
      model_type: existingCheckpointId, model_types: [existingCheckpointId],
    },
    {
      id: 'new-checkpoint-download', filename: 'generic-h3-turbo.safetensors',
      status: 'completed', progress: 1, bytes_downloaded: 200, bytes_total: 200,
      error: null, started_at: nowSeconds - 240, completed_at: nowSeconds - 120,
      model_type: framesId, model_types: [framesId, referencesId],
    },
  ];
  const beforeRefresh = [imageModel(), oldImportedModel()];
  const afterRefresh = [...beforeRefresh, ...pair];
  const f = fixture({
    catalog: (_fetchCount, reloadCount) => reloadCount > 0 ? afterRefresh : beforeRefresh,
    options: {
      [imageId]: {
        model_type: imageId, architecture: 'flux', family: 'flux',
        default_num_inference_steps: 28, default_guidance_scale: 1,
        inference_steps_min: 1, inference_steps_max: 50,
      },
      ...optionById,
    },
    visibility: {
      configured: true, defaults_version: 999,
      enabled_models: [imageId], initialized_mature_models: [],
    },
    studioPreferences: {
      configured: true, generation_mode: 'image', studio_image_workflow: 'generate',
      studio_video_workflow: 'frames', audio_sub_mode: 'speech',
      selected_model_per_mode: {image: imageId}, selected_model_per_audio_sub_mode: {},
      inference_steps_per_model: {}, h3_optimizations: {},
    },
    downloads: completedDownloads,
  });

  const store = f.store;
  await store.getState().loadModels();
  await f.settle();
  assert.equal(store.getState().generationMode, 'image');
  assert.equal(store.getState().params.model_type, imageId);
  assert.equal(store.getState().enabledModels.has(existingCheckpointId), false,
    'The existing imported checkpoint starts opted out');

  store.getState().setParam('prompt', 'Keep the active image prompt');
  store.getState().setParam('seed', 742901);
  store.getState().setParam('num_inference_steps', 37);
  await f.settle();
  const expected = {
    mode: store.getState().generationMode,
    model: store.getState().params.model_type,
    prompt: store.getState().params.prompt,
    seed: store.getState().params.seed,
    steps: store.getState().params.num_inference_steps,
  };

  store.getState().pollCivitAIDownloads();
  await waitUntil(() => f.calls.reloadModels === 1
    && store.getState().models.some(model => model.model_type === framesId)
    && store.getState().enabledModels.has(referencesId), 'checkpoint catalog refresh and visibility update');
  await f.settle();

  assert.equal(store.getState().generationMode, expected.mode, 'Refresh preserves the active generation mode');
  assert.equal(store.getState().params.model_type, expected.model, 'Refresh preserves the selected model');
  assert.equal(store.getState().params.prompt, expected.prompt, 'Refresh preserves the in-progress prompt');
  assert.equal(store.getState().params.seed, expected.seed, 'Refresh preserves the selected seed');
  assert.equal(store.getState().params.num_inference_steps, expected.steps, 'Refresh preserves the remembered steps');
  assert.equal(store.getState().enabledModels.has(framesId), true, 'Newly discovered Frames model is enabled');
  assert.equal(store.getState().enabledModels.has(referencesId), true, 'Newly discovered References companion is enabled');
  assert.equal(store.getState().enabledModels.has(existingCheckpointId), false,
    'An older completed download record does not undo the user’s opt-out');
}

async function testImportedAttentionRouting() {
  const f = fixture();
  const store = f.store;
  store.setState({models: pair, enabledModels: new Set([framesId, referencesId]),
    generationMode: 'video', studioVideoWorkflow: 'frames',
    studioVideoEffectiveCreateRoute: 'generate', durationSeconds: 5,
    params: {...store.getState().params, model_type: framesId, image_mode: 0, prompt: 'A traveler waves.'},
    modelOptions: optionById[framesId]});
  for (const attention of ['sol', 'sdpa', '']) {
    store.getState().setParam('override_attention', attention);
    await store.getState().loadModelOptions(framesId);
    assert.equal(store.getState().params.override_attention, attention, 'option refresh retains explicit attention');
    await store.getState().startGeneration('queue');
    assert.equal(f.calls.submissions.at(-1).override_attention, attention, 'generation submits the selected attention');
    store.setState({selectedOutputMeta: {params: {model_type: framesId, num_inference_steps: 8,
      resolution: '864x480', video_length: 124, prompt: 'A traveler waves.', override_attention: attention}},
      generating: false, metadataLoading: false});
    await store.getState().loadSettingsFromOutput();
    assert.equal(store.getState().params.override_attention, attention, 'saved settings restore the selected attention');
    assert.equal(store.getState().params.minimax_h3_turbo_mode, false);
    assert.equal(store.getState().params.skip_steps_cache_type, '');
  }
  const unsupported = {...optionById[framesId], sol_attention_status: {supported: false}};
  const g = fixture({options: {...optionById, [framesId]: unsupported}});
  g.store.setState({models: pair, modelOptions: unsupported, params: {...g.store.getState().params,
    model_type: framesId, override_attention: 'sol'}});
  await g.store.getState().loadModelOptions(framesId);
  assert.equal(g.store.getState().params.override_attention, '', 'unsupported Sol falls back to dense Auto');
}

(async () => {
  await testDynamicPairRoutingAndTurboStepMemory();
  await testImportedAttentionRouting();
  await testCompletedCheckpointRefreshPreservesUserStateAndOptOuts();
  console.log('Imported H3 dynamic pairing, native LoRA basis, baked Turbo step memory, and completed-download refresh checks passed');
})().catch(error => {console.error(error); process.exitCode = 1;});
