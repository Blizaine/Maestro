// CPU-only Studio action test. All browser globals and HTTP requests are local fakes.
const assert = require('node:assert/strict');
const path = require('node:path');
const Module = require('node:module');
const root = path.resolve(__dirname, '../..');
const esbuild = require(path.join(root, 'ui/node_modules/esbuild'));

const storage = () => {
  const values = new Map();
  return {getItem: key => values.get(key) ?? null, setItem: (key, value) => values.set(key, String(value)), removeItem: key => values.delete(key)};
};
global.localStorage = storage();
global.sessionStorage = storage();
global.window = {
  localStorage, sessionStorage, location: {origin: 'http://fixture.invalid', href: 'http://fixture.invalid/'},
  addEventListener() {}, removeEventListener() {}, dispatchEvent() {},
  matchMedia: () => ({matches: false, addEventListener() {}, removeEventListener() {}}),
};
Object.defineProperty(global, 'navigator', {value: {userAgent: 'Node CPU fixture', platform: 'fixture', maxTouchPoints: 0}, configurable: true});
const requests = [];
global.fetch = async (url, options = {}) => {
  if (url !== '/api/v1/llm/enhance-prompt') throw new Error(`Unexpected request: ${url}`);
  const body = JSON.parse(options.body);
  requests.push(body);
  return {ok: true, json: async () => ({original: body.prompt, enhanced: body.prompt})};
};

(async () => {
  const result = await esbuild.build({
    entryPoints: [path.join(root, 'ui/src/stores/useStore.ts')], bundle: true,
    platform: 'node', format: 'cjs', write: false, logLevel: 'silent',
    define: {'import.meta.env': '{}'},
  });
  const filename = path.join(root, '.codex-tmp/reference-enhancement-store.cjs');
  const compiled = new Module(filename, module);
  compiled.filename = filename;
  compiled.paths = module.paths;
  compiled._compile(result.outputFiles[0].text, filename);
  const {useStore} = compiled.exports;
  const model = {
    model_type: 'minimax_h3_ref2va_pika_flex', architecture: 'minimax_h3', family: 'minimax_h3',
    omni_reference: true, minimax_h3_pika_flex: true, fps: 24,
    frames_minimum: 124, frames_maximum: 362, frames_steps: 17,
    t2v_class: true, i2v_class: false,
  };
  const baseParams = useStore.getState().params;
  for (const intent of ['identity', 'scene', 'composition', 'style', 'object']) {
    const references = [
      {type: 'image', path: '/uploads/room.png', role: 'Room', image_intent: intent},
      {type: 'video', path: '/uploads/rhea.mp4', role: 'Rhea', video_intent: 'character', library_character_id: 'rhea', include_audio: false},
      {type: 'video', path: '/uploads/noah.mp4', role: 'Noah', video_intent: 'character', library_character_id: 'noah', include_audio: false},
      {type: 'audio', path: '/uploads/rhea.wav', role: 'Rhea', audio_intent: 'voice', library_character_id: 'rhea'},
      {type: 'audio', path: '/uploads/noah.wav', role: 'Noah', audio_intent: 'voice', library_character_id: 'noah'},
    ];
    const frozen = structuredClone(references);
    useStore.setState({
      models: [model], modelOptions: model, generationMode: 'video', sidebarMode: 'studio',
      studioVideoWorkflow: 'references', studioVideoEffectiveCreateRoute: 'omni',
      durationSeconds: 362 / 24, slidingWindowSeconds: 362 / 24, slidingWindowOverlap: 0,
      startImage: null, endImage: null, imageRefs: [], isEnhancing: false, promptEnhanceError: null,
      reconcileStudioVideoCreateRoute() {},
      params: {...baseParams, model_type: model.model_type, prompt: 'Rhea greets Noah in the supplied room.', image_mode: 0,
        video_length: 362, minimax_h3_references: references, minimax_h3_reference_sequence: false, minimax_h3_multi_window: false},
    });
    const count = requests.length;
    await useStore.getState().enhancePrompt();
    assert.equal(useStore.getState().promptEnhanceError, null, intent);
    assert.equal(requests.length, count + 1, `${intent}: exactly one explicit enhancement request`);
    assert.deepEqual(requests.at(-1).minimax_h3_references, frozen, `${intent}: selected roles, identity IDs and voice associations reach the backend intact`);
    assert.deepEqual(references, frozen, `${intent}: input references remain unchanged`);
  }
  console.log('5 Studio Enhance Now reference-manifest cases passed; no real HTTP requests or browser state used.');
})().catch(error => {console.error(error); process.exitCode = 1;});
