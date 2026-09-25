// Run real Store actions with isolated browser/network state; no GPU or downloads.
// node tests/ui/h3_singularity.cjs
const fs = require('node:fs'), path = require('node:path'), vm = require('node:vm');
const assert = require('node:assert/strict');
const root = path.resolve(__dirname, '../..');
const ts = require(path.join(root, 'ui/node_modules/typescript'));
const singularity = 'minimax_h3_ref2va_singularity';
const regular = 'minimax_h3_ref2va';
const presetId = 'lightx2v-ref2va-turbo4-v0.1-comfy-bf16';
const manifest = JSON.parse(fs.readFileSync(path.join(root, 'app/models/minimax_h3/turbo_presets.json')));
const turbo = manifest.presets.find(preset => preset.id === presetId);
assert.ok(turbo, 'The recommended Turbo preset must exist');
const presets = manifest.presets.filter(preset => preset.workflow === 'ref2va' || preset.workflow === 'all');
const pdd = presets.find(preset => preset.id === 'alibaba-pai-ref2va-pdd-8step');
const options = id => ({
  model_type: id, architecture: regular, fps: 24, frames_minimum: 124, frames_maximum: 345,
  frames_steps: 17, guidance_max_phases: 1, omni_reference: true,
  default_num_inference_steps: id === singularity ? 4 : 20, default_guidance_scale: 1,
  minimax_h3_turbo: {
    ...(id === singularity ? turbo : pdd), preset_id: id === singularity ? presetId : pdd.id,
    presets, default_enabled: id === singularity, unaccelerated_steps: 20,
  },
});

function fixture() {
  const modules = new Map(), pendingOptions = [], pendingDefaults = [];
  const api = new Proxy({
    fetchModelOptions: model => new Promise(resolve => pendingOptions.push({model, resolve})),
    fetchDefaults: model => new Promise(resolve => pendingDefaults.push({model, resolve})),
    fetchLoras: async () => ({loras: presets.map(preset => preset.filename)}),
    updateStudioPreferences: async () => ({}),
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
      module, exports: module.exports, console, URL, AbortController,
      localStorage: {getItem: () => null, setItem: () => {}},
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
  const {useStore: store, modelSupportsStudioVideoMediaIntent} = load(path.join(root, 'ui/src/stores/useStore.ts'));
  store.setState({generationMode: 'video', studioVideoWorkflow: 'references'});
  const settle = async () => { for (let i = 0; i < 8; i++) await Promise.resolve(); };
  return {store, settle, pendingOptions, pendingDefaults, modelSupportsStudioVideoMediaIntent};
}

const plain = value => JSON.parse(JSON.stringify(value));
(async () => {
  for (const defaultsFirst of [true, false]) {
    const f = fixture(), {store, settle} = f;
    assert.ok(store.getState().enabledModels.has(singularity));
    store.getState().selectModel(singularity);
    const defaults = () => f.pendingDefaults.shift().resolve({num_inference_steps: 4, guidance_scale: 1});
    const modelOptions = () => f.pendingOptions.shift().resolve(options(singularity));
    (defaultsFirst ? defaults : modelOptions)();
    await settle();
    (defaultsFirst ? modelOptions : defaults)();
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
    f.pendingDefaults.shift().resolve({num_inference_steps: 20});
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
  f.pendingDefaults.shift().resolve({num_inference_steps: 4});
  await settle();
  assert.equal(store.getState().params.minimax_h3_turbo_mode, false, 'Do not overwrite a choice during loading');
  assert.equal(store.getState().params.num_inference_steps, 20);
  assert.deepEqual(plain(store.getState().params.activated_loras), []);

  store.getState().selectModel(singularity);
  store.getState().selectModel(regular);
  f.pendingOptions[1].resolve(options(regular));
  f.pendingDefaults[1].resolve({num_inference_steps: 20});
  await settle();
  f.pendingOptions[0].resolve(options(singularity));
  f.pendingDefaults[0].resolve({num_inference_steps: 4});
  await settle();
  assert.equal(store.getState().params.model_type, regular);
  assert.equal(store.getState().params.minimax_h3_turbo_mode, false);
  assert.equal(store.getState().params.num_inference_steps, 20);
  assert.deepEqual(plain(store.getState().params.activated_loras), []);
  const definition = {...options(singularity), family: 'minimax_h3'};
  const intent = {hasFrameGuidance: false, hasOmniReferences: false, hasAudioDrive: false};
  assert.equal(f.modelSupportsStudioVideoMediaIntent(definition, {...intent, workflow: 'references'}), true);
  assert.equal(f.modelSupportsStudioVideoMediaIntent(definition, {...intent, workflow: 'frames'}), false);

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
  console.log('Stale requests, References-only routing, and independent Director recipe/opt-out passed');
})().catch(error => {console.error(error); process.exitCode = 1;});
