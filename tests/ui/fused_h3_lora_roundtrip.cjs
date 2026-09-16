// Execute real Store actions with network and browser persistence isolated.
// Run with: node tests/ui/fused_h3_lora_roundtrip.cjs
const fs = require('fs'), path = require('path'), vm = require('vm');
const assert = require('node:assert/strict');
const root = path.resolve(__dirname, '../..');
const ts = require(path.join(root, 'ui/node_modules/typescript'));
const modules = new Map();
let submitted;
const api = new Proxy({
  submitGeneration: async params => {
    submitted = JSON.parse(JSON.stringify(params));
    return { job_id: 'fused-lora-roundtrip', status: 'held' };
  },
}, { get: (target, name) => name in target ? target[name] : async () => {
  throw new Error(`Unexpected API call: ${String(name)}`);
} });
function create(init) {
  let state;
  const get = () => state;
  const set = update => { state = { ...state, ...(typeof update === 'function' ? update(state) : update) }; };
  state = init(set, get);
  return { getState: get, setState: set, subscribe: () => () => {} };
}
function load(file) {
  file = path.resolve(file);
  if (modules.has(file)) return modules.get(file).exports;
  const module = { exports: {} };
  modules.set(file, module);
  const source = fs.readFileSync(file, 'utf8').replaceAll('import.meta', '({})');
  const code = ts.transpileModule(source, { compilerOptions: {
    module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2022,
  } }).outputText;
  vm.runInNewContext(code, {
    module, exports: module.exports, console, URL, AbortController,
    localStorage: { getItem: () => null, setItem: () => {} },
    setTimeout: () => 1, clearTimeout: () => {}, setInterval: () => 1, clearInterval: () => {},
    window: { setTimeout: () => 1, clearTimeout: () => {} },
    require: id => {
      if (id === 'zustand') return { create };
      if (id.endsWith('/api/client')) return api;
      if (id.endsWith('/lib/theme')) return {
        getStoredPrefs: () => ({ family: 'default', mode: 'dark' }), applyThemePrefs: () => {},
      };
      if (id.startsWith('.')) return load(path.resolve(path.dirname(file), id) + '.ts');
      throw new Error(`Unexpected import: ${id}`);
    },
  }, { filename: file });
  return module.exports;
}
const plain = value => JSON.parse(JSON.stringify(value));
(async () => {
  const { useStore: store } = load(path.join(root, 'ui/src/stores/useStore.ts'));
  const loras = ['character.safetensors', 'film_style.safetensors'];
  for (const modelType of ['minimax_h3_fused_turbo', 'minimax_h3_ref2va_fused_turbo']) {
    const defaults = JSON.parse(fs.readFileSync(path.join(root, 'app/defaults', modelType + '.json'), 'utf8'));
    const references = modelType.includes('ref2va');
    const options = {
      ...defaults.model, model_type: modelType, fps: 24,
      default_num_inference_steps: 4, default_guidance_scale: 1,
      omni_reference: references, supports_reference_audio: references,
    };
    const params = { ...defaults, model: undefined, model_type: modelType,
      prompt: 'A traveler looks toward the mountains.', resolution: '704x704',
      image_mode: 0, seed: 1234, activated_loras: loras, loras_multipliers: '0.35 0.00',
      minimax_h3_references: references ? [{ id: 'test-reference', type: 'image', path: 'character.png', filename: 'character.png' }] : [],
      minimax_h3_turbo_mode: true, minimax_h3_turbo_preset: 'stale-preset',
    };
    store.setState({
      params: { ...store.getState().params, activated_loras: [], loras_multipliers: '' },
      selectedOutputMeta: { params }, selectedOutput: -1, jobs: [],
      models: [{ ...options, family: 'minimax_h3', is_downloaded: true }],
      modelOptions: options, enabledModels: new Set([modelType]),
      generationMode: 'video', studioVideoWorkflow: references ? 'references' : 'frames',
      selectedModelPerMode: { video: modelType }, activeWorkspace: 'test',
      systemStats: { gpu: { vram_total_gb: 24 } },
      loadModelOptions: async () => { store.setState({ modelOptions: options }); },
      loadLoras: async () => { store.setState({ availableLoras: loras }); },
    });
    await store.getState().loadSettingsFromOutput();
    assert.deepEqual(plain(store.getState().params.activated_loras), loras);
    assert.equal(store.getState().params.loras_multipliers, '0.35 0.00');
    assert.deepEqual(plain(store.getState().loraWeights), {
      'character.safetensors': [0.35], 'film_style.safetensors': [0],
    });
    assert.equal(store.getState().params.minimax_h3_turbo_mode, false);
    submitted = undefined;
    await store.getState().startGeneration('queue');
    assert.ok(submitted, JSON.stringify({ jobs: store.getState().jobs, error: store.getState().promptEnhanceError }));
    assert.deepEqual(submitted.activated_loras, loras);
    assert.equal(submitted.loras_multipliers, '0.35 0.00');
    assert.equal(submitted.num_inference_steps, 4);
    assert.equal(submitted.minimax_h3_turbo_mode, false);
    console.log(`${modelType}: Load Settings -> strengths -> held submission passed`);
  }
})().catch(error => { console.error(error); process.exitCode = 1; });
