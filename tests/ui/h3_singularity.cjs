// Run real Store actions with isolated browser/network state; no GPU or downloads.
// node tests/ui/h3_singularity.cjs
const fs = require('node:fs'), path = require('node:path'), vm = require('node:vm');
const assert = require('node:assert/strict');
const root = path.resolve(__dirname, '../..');
const ts = require(path.join(root, 'ui/node_modules/typescript'));
const singularity = 'minimax_h3_ref2va_singularity';
const framesSingularity = 'minimax_h3_singularity';
const regular = 'minimax_h3_ref2va';
const presetId = 'lightx2v-ref2va-turbo4-v0.1-comfy-bf16';
const manifest = JSON.parse(fs.readFileSync(path.join(root, 'app/models/minimax_h3/turbo_presets.json')));
const turbo = manifest.presets.find(preset => preset.id === presetId);
assert.ok(turbo, 'The recommended Turbo preset must exist');
const presets = manifest.presets.filter(preset => preset.workflow === 'ref2va' || preset.workflow === 'all');
const pdd = presets.find(preset => preset.id === 'alibaba-pai-ref2va-pdd-8step');
const isSingularity = id => id === singularity || id === framesSingularity;
const isOmni = id => id === singularity || id === regular;
const options = id => ({
  model_type: id, architecture: isOmni(id) ? 'minimax_h3_ref2va' : 'minimax_h3',
  fps: 24, frames_minimum: 124, frames_maximum: 345,
  frames_steps: 17, guidance_max_phases: 1, ...(isOmni(id) ? {omni_reference: true} : {}),
  default_num_inference_steps: isSingularity(id) ? 4 : 20, default_guidance_scale: 1,
  minimax_h3_turbo: {
    ...(isSingularity(id) ? turbo : pdd), preset_id: isSingularity(id) ? presetId : pdd.id,
    presets, default_enabled: isSingularity(id), unaccelerated_steps: 20,
  },
});

const defaults = id => ({num_inference_steps: isSingularity(id) ? 4 : 20, guidance_scale: 1});

function fixture() {
  const modules = new Map(), pendingOptions = [], pendingDefaults = [];
  const storage = new Map(), submissions = [];
  const api = new Proxy({
    fetchModelOptions: model => new Promise(resolve => pendingOptions.push({model, resolve})),
    fetchDefaults: model => new Promise(resolve => pendingDefaults.push({model, resolve})),
    fetchLoras: async () => ({loras: presets.map(preset => preset.filename)}),
    updateStudioPreferences: async () => ({}),
    submitGeneration: async params => {
      submissions.push(JSON.parse(JSON.stringify(params)));
      return {job_id: `singularity-${submissions.length}`, status: 'held'};
    },
  }, {get: (target, name) => name in target ? target[name] : async () => {
    throw new Error(`Unexpected API call: ${String(name)}`);
  }});
  function create(init) {
    let state;
    const get = () => state;
    const set = update => { state = {...state, ...(typeof update === 'function' ? update(state) : update)}; };
    state = init(set, get);
    return {getState: get, setState: set, subscribe: () => () => {}};
  }
  function load(file) {
    file = path.resolve(file);
    if (modules.has(file)) return modules.get(file).exports;
    const module = {exports: {}};
    modules.set(file, module);
    const source = fs.readFileSync(file, 'utf8').replaceAll('import.meta', '({})');
    const code = ts.transpileModule(source, {compilerOptions: {
      module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2022,
    }}).outputText;
    vm.runInNewContext(code, {
      module, exports: module.exports, console, URL, AbortController, structuredClone,
      localStorage: {
        getItem: key => storage.get(key) || null,
        setItem: (key, value) => storage.set(key, value),
      },
      setTimeout: () => 1, clearTimeout: () => {}, setInterval: () => 1, clearInterval: () => {},
      window: {setTimeout: () => 1, clearTimeout: () => {}},
      require: id => {
        if (id === 'zustand') return {create};
        if (id.endsWith('/api/client')) return api;
        if (id.endsWith('/lib/theme')) return {
          getStoredPrefs: () => ({family: 'default', mode: 'dark'}), applyThemePrefs: () => {},
        };
        if (id.startsWith('.')) return load(path.resolve(path.dirname(file), id) + '.ts');
        throw new Error(`Unexpected import: ${id}`);
      },
    }, {filename: file});
    return module.exports;
  }
  const {useStore: store, modelSupportsStudioVideoMediaIntent, getModelsForFamily} = load(path.join(root, 'ui/src/stores/useStore.ts'));
  store.setState({generationMode: 'video', studioVideoWorkflow: 'references'});
  const settle = async () => { for (let i = 0; i < 8; i++) await Promise.resolve(); };
  return {store, settle, pendingOptions, pendingDefaults, modelSupportsStudioVideoMediaIntent, getModelsForFamily, storage, submissions};
}

const plain = value => JSON.parse(JSON.stringify(value));
async function settleModelLoads(f) {
  for (let turn = 0; turn < 4; turn++) {
    const optionsCalls = f.pendingOptions.splice(0);
    const defaultsCalls = f.pendingDefaults.splice(0);
    for (const call of optionsCalls) call.resolve(options(call.model));
    for (const call of defaultsCalls) call.resolve(defaults(call.model));
    await f.settle();
  }
}

(async () => {
  for (const defaultsFirst of [true, false]) {
    const f = fixture(), {store, settle} = f;
    assert.ok(store.getState().enabledModels.has(singularity));
    assert.ok(store.getState().enabledModels.has(framesSingularity));
    store.getState().selectModel(singularity);
    const resolveDefaults = () => f.pendingDefaults.shift().resolve(defaults(singularity));
    const modelOptions = () => f.pendingOptions.shift().resolve(options(singularity));
    (defaultsFirst ? resolveDefaults : modelOptions)();
    await settle();
    (defaultsFirst ? modelOptions : resolveDefaults)();
    await settle();
    let state = store.getState();
    assert.equal(state.params.minimax_h3_turbo_mode, true);
    assert.equal(state.params.minimax_h3_turbo_preset, presetId);
    assert.equal(state.params.num_inference_steps, 4);
    assert.deepEqual(plain(state.params.activated_loras), [turbo.filename]);
    assert.deepEqual(plain(state.loraWeights[turbo.filename]), [1]);
    state.setLoraWeight(turbo.filename, 0, 0.8);
    state.loadModelOptions(singularity);
    f.pendingOptions.shift().resolve(options(singularity));
    await settle();
    assert.deepEqual(plain(store.getState().loraWeights[turbo.filename]), [0.8]);
    store.getState().toggleLora(turbo.filename);
    store.getState().setParam('num_inference_steps', 20);
    store.getState().loadModelOptions(singularity);
    f.pendingOptions.shift().resolve(options(singularity));
    await settle();
    assert.equal(store.getState().params.minimax_h3_turbo_mode, false);
    assert.equal(store.getState().params.num_inference_steps, 20);
    assert.deepEqual(plain(store.getState().params.activated_loras), []);

    store.getState().selectModel(regular);
    f.pendingOptions.shift().resolve(options(regular));
    f.pendingDefaults.shift().resolve(defaults(regular));
    await settle();
    assert.equal(store.getState().params.minimax_h3_turbo_mode, false);
    assert.equal(store.getState().params.num_inference_steps, 20);
    assert.deepEqual(plain(store.getState().params.activated_loras), []);
    console.log(`Studio defaults ${defaultsFirst ? 'first' : 'last'}: recipe, weight, opt-out, model switch passed`);
  }

  const f = fixture(), {store, settle} = f;
  store.getState().selectModel(singularity);
  store.getState().setParam('minimax_h3_turbo_mode', false);
  f.pendingOptions.shift().resolve(options(singularity));
  f.pendingDefaults.shift().resolve(defaults(singularity));
  await settle();
  assert.equal(store.getState().params.minimax_h3_turbo_mode, false, 'Do not overwrite a choice during loading');
  assert.equal(store.getState().params.num_inference_steps, 20);
  assert.deepEqual(plain(store.getState().params.activated_loras), []);

  store.getState().selectModel(singularity);
  store.getState().selectModel(regular);
  f.pendingOptions[1].resolve(options(regular));
  f.pendingDefaults[1].resolve(defaults(regular));
  await settle();
  f.pendingOptions[0].resolve(options(singularity));
  f.pendingDefaults[0].resolve(defaults(singularity));
  await settle();
  assert.equal(store.getState().params.model_type, regular);
  assert.equal(store.getState().params.minimax_h3_turbo_mode, false);
  assert.equal(store.getState().params.num_inference_steps, 20);
  assert.deepEqual(plain(store.getState().params.activated_loras), []);
  const definition = {...options(singularity), family: 'minimax_h3', is_t2v: true, is_i2v: true};
  const framesDefinition = {
    ...options(framesSingularity), family: 'minimax_h3', architecture: 'minimax_h3',
    is_t2v: true, is_i2v: true, supports_end_frame: true,
  };
  const intent = {hasFrameGuidance: false, hasOmniReferences: false, hasAudioDrive: false};
  assert.equal(f.modelSupportsStudioVideoMediaIntent(definition, {...intent, workflow: 'references'}), true);
  assert.equal(f.modelSupportsStudioVideoMediaIntent(definition, {...intent, workflow: 'frames'}), false);
  assert.equal(f.modelSupportsStudioVideoMediaIntent(framesDefinition, {...intent, workflow: 'references'}), false);
  assert.equal(f.modelSupportsStudioVideoMediaIntent(framesDefinition, {...intent, workflow: 'frames'}), true,
    'Singularity Frames supports text-to-video without attached images');
  assert.equal(f.modelSupportsStudioVideoMediaIntent(framesDefinition, {
    ...intent, workflow: 'frames', hasFrameGuidance: true,
  }), true, 'Singularity Frames supports image-to-video anchors');
  assert.ok(f.getModelsForFamily('minimax_h3', [definition, framesDefinition], 'video')
    .some(model => model.model_type === framesSingularity), 'Frames entry stays in the H3 family group');

  const routing = fixture();
  const routingState = routing.store.getState();
  routing.store.setState({
    models: [definition, framesDefinition],
    enabledModels: new Set([singularity, framesSingularity]),
    generationMode: 'video',
    studioVideoWorkflow: 'references',
    studioVideoEffectiveCreateRoute: 'omni',
    studioVideoModelPerCreateRoute: {omni: singularity},
    selectedModelPerMode: {...routingState.selectedModelPerMode, video: singularity},
    modelOptions: options(singularity),
    params: {...routingState.params, model_type: singularity, image_mode: 0},
  });
  routing.store.getState().setStudioVideoWorkflow('frames');
  await settleModelLoads(routing);
  let routed = routing.store.getState();
  assert.equal(routed.params.model_type, framesSingularity,
    'References Singularity switches to its Frames ID when Frames opens');
  assert.equal(routed.studioVideoEffectiveCreateRoute, 'generate');
  assert.equal(routed.params.minimax_h3_turbo_mode, true);
  assert.equal(routed.params.num_inference_steps, 4);
  assert.equal(routed.studioVideoModelPerCreateRoute.omni, singularity,
    'The saved Reference model ID remains attached to the Omni route');

  routed.selectStudioVideoModel(framesSingularity);
  await settleModelLoads(routing);
  routing.store.setState({params: {...routing.store.getState().params, image_start: '/input/start.png'}});
  routing.store.getState().reconcileStudioVideoCreateRoute('Start frame added');
  routed = routing.store.getState();
  assert.equal(routed.studioVideoEffectiveCreateRoute, 'guided');
  assert.equal(routed.params.model_type, framesSingularity,
    'The Frames ID remains selectable with a first-frame anchor');
  routing.store.setState({params: {...routed.params, image_start: undefined, image_end: '/input/end.png'}});
  routing.store.getState().reconcileStudioVideoCreateRoute('End frame added');
  assert.equal(routing.store.getState().studioVideoEffectiveCreateRoute, 'guided');
  assert.equal(routing.store.getState().params.model_type, framesSingularity,
    'The Frames ID remains selectable with a last-frame anchor');

  routing.store.getState().setStudioVideoWorkflow('references');
  await settleModelLoads(routing);
  routed = routing.store.getState();
  assert.equal(routed.studioVideoEffectiveCreateRoute, 'omni');
  assert.equal(routed.params.model_type, singularity,
    'References restores the existing Ref2VA Singularity ID');
  assert.equal(routed.params.minimax_h3_turbo_mode, true);
  assert.equal(routed.params.num_inference_steps, 4);
  assert.equal(routed.studioVideoModelPerCreateRoute.guided, framesSingularity,
    'Guided Frames preference is kept independently from the Omni preference');

  routed.setParam('minimax_h3_turbo_mode', false);
  routed.setParam('num_inference_steps', 20);
  routing.store.getState().setStudioVideoWorkflow('frames');
  await settleModelLoads(routing);
  routed = routing.store.getState();
  assert.equal(routed.params.model_type, framesSingularity);
  assert.equal(routed.params.minimax_h3_turbo_mode, false,
    'Turbo opt-out remains off when switching to the paired Frames ID');
  assert.equal(routed.params.num_inference_steps, 20,
    'The unaccelerated step count remains 20 across paired model switches');
  const savedRoutes = JSON.parse(routing.storage.get('maestro_studio_video_create_route_v1'));
  assert.equal(savedRoutes.models.omni, singularity);
  assert.equal(savedRoutes.models.guided, framesSingularity);
  assert.equal(savedRoutes.models.generate, framesSingularity);

  const videoToVideo = fixture();
  const videoState = videoToVideo.store.getState();
  videoToVideo.store.setState({
    models: [framesDefinition, definition],
    enabledModels: new Set([framesSingularity, singularity]),
    generationMode: 'video',
    studioVideoWorkflow: 'frames',
    studioVideoEffectiveCreateRoute: 'generate',
    studioVideoModelPerCreateRoute: {generate: framesSingularity, omni: singularity},
    selectedModelPerMode: {...videoState.selectedModelPerMode, video: framesSingularity},
    modelOptions: {
      ...options(framesSingularity), video_to_video_inpaint: true,
      minimax_h3_media_sources: true,
      audio_prompt_type_sources: {choices: [['Control video audio', 'K']]},
    },
    durationSeconds: 5,
    params: {
      ...videoState.params, model_type: framesSingularity, image_mode: 0,
      prompt: 'Restage the control video with a softer palette.',
      video_guide: '/server-input/control.mp4', video_prompt_type: 'GV',
      minimax_h3_control_visual_mode: 'whole', audio_prompt_type: 'K',
      denoising_strength: 0.65, masking_strength: 0.8,
    },
  });
  videoToVideo.store.getState().reconcileStudioVideoCreateRoute('Control video added');
  assert.equal(videoToVideo.store.getState().studioVideoEffectiveCreateRoute, 'generate',
    'Control Video remains an input to the Frames workflow instead of becoming an Omni reference route');
  await videoToVideo.store.getState().startGeneration('queue');
  assert.equal(videoToVideo.submissions.length, 1);
  assert.equal(videoToVideo.submissions[0].model_type, framesSingularity);
  assert.equal(videoToVideo.submissions[0].video_guide, '/server-input/control.mp4',
    'V2V submits the Control Video source');
  assert.equal(videoToVideo.submissions[0].video_prompt_type, 'GV');
  assert.equal(videoToVideo.submissions[0].denoising_strength, 0.65,
    'V2V submits its selected denoising strength');

  store.setState({savedLoraPerMode: {video: {
    activated_loras: ['style.safetensors', pdd.filename], loras_multipliers: '0.35 1.00',
    loraWeights: {'style.safetensors': [0.35], [pdd.filename]: [1]}, availableLoras: [],
  }}});
  store.getState().initializeDirectorH3Turbo(singularity, options(singularity));
  let state = store.getState();
  assert.equal(state.directorH3TurboModeByModel[singularity], true);
  assert.equal(state.directorH3TurboPresetByModel[singularity], presetId);
  assert.equal(state.directorVideoInferenceStepsByModel[singularity], 4);
  assert.deepEqual(plain(state.savedLoraPerMode.video.activated_loras), ['style.safetensors', turbo.filename]);
  assert.equal(state.savedLoraPerMode.video.loras_multipliers, '0.35 1.00');
  state.setDirectorH3TurboMode(singularity, false);
  state.setDirectorVideoInferenceSteps(singularity, 20);
  state.initializeDirectorH3Turbo(singularity, options(singularity));
  state = store.getState();
  assert.equal(state.directorH3TurboModeByModel[singularity], false);
  assert.equal(state.directorVideoInferenceStepsByModel[singularity], 20);
  state.initializeDirectorH3Turbo(regular, options(regular));
  assert.equal(store.getState().directorH3TurboModeByModel[regular], undefined);
  console.log('Stale requests, paired Frames/References routing, V2V payload, and independent Director recipe/opt-out passed');
})().catch(error => {console.error(error); process.exitCode = 1;});
