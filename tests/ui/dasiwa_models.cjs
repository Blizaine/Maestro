// Retired presets stay opt-in if restored locally; preserve legacy routing.
const fs = require('node:fs'), path = require('node:path'), vm = require('node:vm');
const assert = require('node:assert/strict');
const root = path.resolve(__dirname, '../..');
const ts = require(path.join(root, 'ui/node_modules/typescript'));
const ids = ['minimax_h3_dasiwa', 'minimax_h3_ref2va_dasiwa',
  'minimax_h3_dasiwa_turbo', 'minimax_h3_ref2va_dasiwa_turbo'];
const allIds = [...ids, 'dasiwa_krea2_raw', 'dasiwa_krea2_turbo',
  'dasiwa_wan2_2_i2v_lightspeed_v9', 'dasiwa_ltx2_3_dragonleap_v4'];
const presets = new Map(allIds.map(id => [id, JSON.parse(fs.readFileSync(path.join(root, 'tests/fixtures/dasiwa', `${id}.json`)))]));
const models = ids.map(id => ({...presets.get(id).model, model_type: id,
  family: id.includes('krea2') ? 'krea2' : id.includes('wan2') ? 'wan' : id.includes('ltx2') ? 'ltx2' : 'minimax_h3',
  is_image: id.includes('krea2'), is_t2v: !id.includes('krea2') && !id.includes('wan2'),
  is_i2v: !id.includes('krea2'), supports_end_frame: !id.includes('ref2va'),
  omni_reference: id.includes('ref2va'), supports_audio_input: true, generates_audio: true,
  fps: 24, frames_steps: 17, frames_minimum: 124, frames_maximum: 345}));
const options = id => ({...models.find(model => model.model_type === id),
  default_num_inference_steps: presets.get(id).num_inference_steps, default_guidance_scale: 1,
  inference_steps_min: id.endsWith('_turbo') ? 4 : 2,
  inference_steps_max: id.endsWith('_turbo') ? 8 : 50,
  sol_attention: id.startsWith('minimax_h3'), sol_attention_status: {supported: true},
  guidance_max_phases: 0, image_prompt_types_allowed: id.includes('ref2va') ? '' : 'TSEV'});

function fixture(storage = new Map()) {
  const modules = new Map();
  const api = new Proxy({
    fetchModels: async () => ({models, families: [{id: 'minimax_h3', label: 'MiniMax H3'}]}),
    fetchModelVisibility: async () => ({configured: true, defaults_version: Number(storage.get('version') || 18),
      enabled_models: JSON.parse(storage.get('enabled') || '[]'), initialized_mature_models: []}),
    updateModelVisibility: async value => {
      storage.set('version', String(value.defaults_version));
      storage.set('enabled', JSON.stringify(value.enabled_models));
    },
    fetchH3WindowOverrides: async () => null,
    fetchStudioPreferences: async () => null,
    fetchModelOptions: async id => options(id),
    fetchDefaults: async id => ({...presets.get(id), model: undefined}),
    fetchLoras: async () => ({loras: []}),
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
  return {store: exported.useStore, storage, supported: exported.modelSupportsStudioVideoMediaIntent,
    settle: async () => {for (let i = 0; i < 25; i++) await Promise.resolve();}};
}

(async () => {
  const f = fixture();
  for (const id of allIds) assert.equal(f.store.getState().enabledModels.has(id), false, `${id} absent from fresh defaults`);
  await f.store.getState().loadModels();
  await f.settle();
  for (const id of allIds) assert.equal(f.store.getState().enabledModels.has(id), false, `${id} not added by migration`);
  f.store.getState().toggleModelEnabled(ids[2]);
  await f.settle();
  assert.equal(f.store.getState().enabledModels.has(ids[2]), true, 'explicit enable still works for a restored preset');
  f.store.getState().toggleModelEnabled(ids[2]);
  await f.settle();
  const reloaded = fixture(f.storage);
  await reloaded.store.getState().loadModels();
  assert.equal(reloaded.store.getState().enabledModels.has(ids[2]), false, 'a later user opt-out survives reload');

  for (const turbo of [false, true]) {
    const frames = ids[turbo ? 2 : 0], refs = ids[turbo ? 3 : 1];
    const g = fixture();
    g.store.setState({models, enabledModels: new Set(ids), generationMode: 'video', studioVideoWorkflow: 'frames',
      studioVideoModels: {frames, references: refs, extend: '', blend: '', avatar: ''},
      servicesConfig: {nsfw_mode: false}});
    g.store.getState().selectStudioVideoModel(frames);
    await g.settle();
    assert.equal(g.store.getState().params.model_type, frames);
    assert.equal(g.store.getState().params.num_inference_steps, turbo ? 8 : 25);
    assert.equal(g.store.getState().params.minimax_h3_turbo_mode, false, 'baked checkpoint never activates a managed adapter');
    if (turbo) {
      assert.equal(g.store.getState().params.override_attention, '', 'baked Turbo follows dense Auto by default');
      g.store.getState().setParam('override_attention', 'sol');
      await g.store.getState().loadModelOptions(frames);
      assert.equal(g.store.getState().params.override_attention, 'sol', 'explicit Sol remains enabled');
      g.store.getState().setParam('override_attention', 'sdpa');
      await g.store.getState().loadModelOptions(frames);
      assert.equal(g.store.getState().params.override_attention, 'sdpa', 'explicit SDPA is respected');
      g.store.getState().setParam('override_attention', 'sla');
      await g.store.getState().loadModelOptions(frames);
      assert.equal(g.store.getState().params.override_attention, '', 'unverified SLA is not inherited');
      g.store.getState().setParam('num_inference_steps', 4);
    }
    g.store.getState().setStudioVideoWorkflow('references');
    await g.settle();
    assert.equal(g.store.getState().params.model_type, refs, 'mode switch picks the same DaSiWa edition');
    g.store.getState().setStudioVideoWorkflow('frames');
    await g.settle();
    assert.equal(g.store.getState().params.model_type, frames);
    assert.equal(g.store.getState().params.num_inference_steps, turbo ? 4 : 25, 'per-model step choice survives paired workflow switching');
    const intent = {hasFrameGuidance: false, hasOmniReferences: false, hasAudioDrive: false};
    assert.equal(g.supported(models.find(m => m.model_type === frames), {...intent, workflow: 'frames', hasFrameGuidance: true}), true);
    assert.equal(g.supported(models.find(m => m.model_type === refs), {...intent, workflow: 'references', hasOmniReferences: true}), true);
    assert.equal(g.supported(models.find(m => m.model_type === refs), {...intent, workflow: 'frames'}), false);
  }
  console.log('DaSiWa excluded from defaults/migration; restored legacy routing, opt-out persistence and baked Turbo settings passed');
})().catch(error => {console.error(error); process.exitCode = 1;});
