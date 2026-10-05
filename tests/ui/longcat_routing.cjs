// Run: node tests/ui/longcat_routing.cjs
// Exercise the real store with intercepted API calls; no models or jobs run.
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const assert = require('node:assert/strict');
const root = path.resolve(__dirname, '../..');
const ts = require(path.join(root, 'ui/node_modules/typescript'));
const submissions = [];
const persisted = new Map();
let uploadFailure = false;

function optionsFor(modelType) {
  return {
    model_type: modelType, architecture: modelType === 'longcat_avatar_multi' ? 'longcat_avatar' : modelType,
    t2v_class: true, i2v_class: true, supports_audio_input: modelType !== 'longcat_video',
    audio_prompt_type_sources: {selection: ['A', 'AB'], default: 'A'},
    image_ref_choices: {choices: [['None', ''], ['Anchor Reference Image', 'KI']]},
    fps: modelType === 'longcat_video' ? 15 : 16, frames_minimum: 5, frames_steps: 4,
    sliding_window: true,
    sliding_window_defaults: {
      window_min: 17, window_max: 93, window_step: 4, window_default: 93,
      overlap_min: 1, overlap_max: 13, overlap_step: 4, overlap_default: 13,
      discard_last_frames: 0,
    },
  };
}

function loadStore() {
  const modules = new Map();
  const api = new Proxy({
    updateStudioPreferences: async () => ({}),
    fetchDefaults: async () => ({}),
    fetchModelOptions: async modelType => optionsFor(modelType),
    fetchLoras: async () => ({loras: []}),
    getFileUrl: name => '/file/' + encodeURIComponent(name),
    uploadImage: async file => {
      if (uploadFailure) throw new Error('intercepted upload failure');
      return {path: '/uploads/' + file.name, filename: file.name};
    },
    submitGeneration: async (params, holdForQueue) => {
      submissions.push({params: structuredClone(params), holdForQueue});
      return {job_id: 'intercepted-' + submissions.length, status: 'held'};
    },
  }, {get: (target, name) => name in target ? target[name] : async () => {
    throw new Error('Unexpected API call: ' + String(name));
  }});
  function create(init) {
    let state;
    const get = () => state;
    const set = update => {
      state = {...state, ...(typeof update === 'function' ? update(state) : update)};
    };
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
      module, exports: module.exports, console, URL, AbortController, structuredClone, File,
      fetch: async () => ({ok: false}),
      localStorage: {getItem: key => persisted.get(key) ?? null, setItem: (key, value) => persisted.set(key, value)},
      setTimeout: () => 1, clearTimeout: () => {}, setInterval: () => 1, clearInterval: () => {},
      window: {setTimeout: () => 1, clearTimeout: () => {}},
      require: id => {
        if (id === 'zustand') return {create};
        if (id.endsWith('/api/client')) return api;
        if (id.endsWith('/lib/theme')) return {
          getStoredPrefs: () => ({family: 'default', mode: 'dark'}), applyThemePrefs: () => {},
        };
        if (id.startsWith('.')) return load(path.resolve(path.dirname(file), id) + '.ts');
        throw new Error('Unexpected import: ' + id);
      },
    }, {filename: file});
    return module.exports;
  }
  return load(path.join(root, 'ui/src/stores/useStore.ts'));
}

const {
  useStore, getFamilyMode, getModelMode, getFamiliesForMode, getModelsForFamily,
  modelSupportsStudioVideoMediaIntent,
} = loadStore();
const initialParams = {...useStore.getState().params};
const families = [{id: 'longcat', label: 'LongCat', order: 60}];
const models = [
  {model_type: 'longcat_video', architecture: 'longcat_video', family: 'longcat', is_i2v: true, is_t2v: true},
  {model_type: 'longcat_avatar', architecture: 'longcat_avatar', family: 'longcat', is_i2v: true, is_t2v: true, supports_audio_input: true},
  {model_type: 'longcat_avatar_multi', architecture: 'longcat_avatar', family: 'longcat', is_i2v: true, is_t2v: true, supports_audio_input: true},
];
const baseIntent = {hasFrameGuidance: false, hasOmniReferences: false, hasAudioDrive: false};

assert.equal(getFamilyMode('longcat'), 'video');
assert.equal(getModelMode('longcat_avatar', 'longcat'), 'video');
assert.ok(getFamiliesForMode('video', families).some(family => family.id === 'longcat'));
assert.deepEqual(getModelsForFamily('longcat', models, 'video').map(model => model.model_type), models.map(model => model.model_type));
assert.ok(!getFamiliesForMode('avatar', families, 'retake').some(family => family.id === 'longcat'));
assert.equal(modelSupportsStudioVideoMediaIntent(models[0], {...baseIntent, workflow: 'frames'}), true);
assert.equal(modelSupportsStudioVideoMediaIntent(models[0], {...baseIntent, workflow: 'frames', hasFrameGuidance: true}), true);
assert.equal(modelSupportsStudioVideoMediaIntent(models[0], {...baseIntent, workflow: 'avatar'}), false);
for (const model of models.slice(1)) {
  for (const intent of [baseIntent, {...baseIntent, hasFrameGuidance: true}, {...baseIntent, hasAudioDrive: true}]) {
    assert.equal(modelSupportsStudioVideoMediaIntent(model, {...intent, workflow: 'avatar'}), true,
      model.model_type + ' must be selectable before its inputs are attached');
    assert.equal(modelSupportsStudioVideoMediaIntent(model, {...intent, workflow: 'frames'}), false);
    assert.equal(modelSupportsStudioVideoMediaIntent(model, {...intent, workflow: 'references'}), false);
  }
}

function seedAvatar({modelType = 'longcat_avatar', startImage = null, endImage = null, imageRefs = [], params = {}} = {}) {
  useStore.setState({
    models, families, enabledModels: new Set(models.map(model => model.model_type)),
    modelOptions: optionsFor(modelType), modelOptionsLoading: false,
    generationMode: 'video', studioVideoWorkflow: 'avatar',
    studioVideoCreateRoute: 'auto', studioVideoEffectiveCreateRoute: 'avatar',
    studioVideoModelPerCreateRoute: {}, selectedModelPerMode: {video: modelType},
    params: {...initialParams, model_type: modelType, image_mode: 0,
      _studio_video_workflow: 'avatar', audio_prompt_type: '', image_prompt_type: '',
      audio_guide: '', audio_guide2: '', video_prompt_type: '', image_start: undefined,
      image_end: undefined, image_refs: undefined, frames_positions: '',
      minimax_h3_references: [], ...params},
    startImage, endImage, imageRefs, imageRefType: '', jobs: [], isGenerating: false,
    promptEnhanceError: null, enhanceOnGeneration: false, deferredPromptEnhancement: null,
    h3WindowPlan: null, multiWindowClipMode: 'disabled',
    durationSeconds: 5, slidingWindowSeconds: 93 / optionsFor(modelType).fps,
    slidingWindowOverlap: 13, slidingWindowLocked: false,
  });
}

async function expectBlocked(input, message) {
  seedAvatar(input);
  const count = submissions.length;
  await useStore.getState().startGeneration('queue');
  assert.equal(useStore.getState().promptEnhanceError, message);
  assert.equal(submissions.length, count, 'incomplete inputs must not reach the generation API');
}

async function expectSubmitted(input) {
  seedAvatar(input);
  const count = submissions.length;
  await useStore.getState().startGeneration('queue');
  assert.equal(useStore.getState().promptEnhanceError, null);
  assert.equal(submissions.length, count + 1);
  assert.equal(submissions.at(-1).holdForQueue, true);
  assert.equal(useStore.getState().jobs[0].status, 'held');
  return submissions.at(-1).params;
}

(async () => {
  await expectBlocked({}, 'Avatar needs an anchor image and Voice audio 1 before generation.');
  await expectBlocked({imageRefs: [{name: 'anchor.png'}]}, 'Avatar needs Voice audio 1 before generation.');
  await expectBlocked({startImage: new File([], 'anchor.png')}, 'Avatar needs Voice audio 1 before generation.');
  await expectBlocked({endImage: new File([], 'end.png'), params: {audio_guide: 'voice.wav'}}, 'Avatar needs an anchor image before generation.');
  await expectBlocked({params: {image_refs: ['saved.png'], video_prompt_type: 'K', audio_guide: 'voice.wav'}}, 'Avatar needs an anchor image before generation.');
  await expectBlocked({params: {image_refs: ['timed.png'], video_prompt_type: 'KFI', frames_positions: '1', audio_guide: 'voice.wav'}}, 'Avatar needs an anchor image before generation.');
  await expectBlocked({imageRefs: [new File([], 'timed.png')], params: {video_prompt_type: 'KFI', frames_positions: '1', audio_guide: 'voice.wav'}}, 'Avatar needs an anchor image before generation.');
  await expectBlocked({params: {image_start: [], image_refs: [''], video_prompt_type: 'KI', audio_guide: 'voice.wav'}}, 'Avatar needs an anchor image before generation.');
  await expectBlocked({modelType: 'longcat_avatar_multi', params: {image_start: 'anchor.png', audio_guide: 'one.wav'}}, 'Avatar needs Voice audio 2 before generation.');
  const multiInput = {modelType: 'longcat_avatar_multi', params: {
    image_start: 'anchor.png', audio_guide: 'one.wav', audio_guide2: 'two.wav',
  }};
  for (const regions of ['', '80:0:20:100 50:0:100:100', '96:0:100:100 0:0:50:100', 'NaN:0:50:100 50:0:100:100']) {
    await expectBlocked({...multiInput, params: {...multiInput.params, speakers_locations: regions}},
      'Choose two valid speaker regions with Left < Right and Top < Bottom.');
  }

  const single = await expectSubmitted({startImage: new File([], 'fresh-anchor.png'), params: {
    audio_guide: 'one.wav', audio_guide2: 'stale-two.wav', audio_prompt_type: 'AB',
    image_start: 'stale-anchor.png', image_end: 'stale-end.png', image_prompt_type: 'SEV',
    video_guide: 'stale-control.mp4', video_source: 'stale-source.mp4', video_mask: 'mask.png',
    speakers_locations: '0:0:50:100 50:0:100:100',
  }});
  assert.deepEqual(single.image_refs, ['/uploads/fresh-anchor.png']);
  assert.equal(single.video_prompt_type, 'KI');
  assert.equal(single.image_prompt_type, '');
  assert.equal(single.audio_prompt_type, 'A');
  assert.equal(single.audio_guide, 'one.wav');
  assert.equal(single.audio_guide2, undefined);
  assert.equal(single.speakers_locations, undefined);
  for (const field of ['image_start', 'image_end', 'video_guide', 'video_source', 'video_mask', 'frames_positions']) assert.equal(single[field], undefined, field);
  assert.equal(single._studio_video_workflow, 'avatar');
  assert.equal(useStore.getState().params.audio_guide2, 'stale-two.wav', 'single submission cleanup must not erase a prepared second voice');

  const reference = await expectSubmitted({imageRefs: [new File([], 'reference.png')], params: {audio_guide: 'one.wav'}});
  assert.deepEqual(reference.image_refs, ['/uploads/reference.png']);
  const saved = await expectSubmitted({params: {image_refs: ['saved.png'], video_prompt_type: 'KI', audio_guide: 'one.wav'}});
  assert.deepEqual(saved.image_refs, ['saved.png']);
  const multi = await expectSubmitted({...multiInput, params: {...multiInput.params, speakers_locations: '10:20:40:80 60:10:90:70'}});
  assert.equal(multi.audio_prompt_type, 'AB');
  assert.equal(multi.audio_guide, 'one.wav');
  assert.equal(multi.audio_guide2, 'two.wav');
  assert.equal(multi.speakers_locations, '10:20:40:80 60:10:90:70');
  assert.deepEqual(multi.image_refs, ['anchor.png']);
  const defaultRegions = await expectSubmitted(multiInput);
  assert.equal(defaultRegions.speakers_locations, '0:0:50:100 50:0:100:100');

  // A 40-second window inherited from another engine must never turn the
  // full audio timeline into one oversized LongCat transformer pass.
  for (const modelType of ['longcat_avatar', 'longcat_avatar_multi']) {
    seedAvatar({modelType, params: {
      image_refs: ['anchor.png'], video_prompt_type: 'KI', audio_guide: 'one.wav',
      audio_guide2: 'two.wav', video_length: 1275, sliding_window_size: 640,
      sliding_window_overlap: 18, sliding_window_discard_last_frames: 8,
    }});
    useStore.setState({durationSeconds: 1275 / 16, slidingWindowSeconds: 40, slidingWindowOverlap: 18});
    await useStore.getState().loadModelOptions(modelType);
    assert.equal(useStore.getState().slidingWindowSeconds, 93 / 16);
    assert.equal(useStore.getState().slidingWindowOverlap, 13);
    assert.equal(useStore.getState().durationSeconds, 1275 / 16, 'Loading window limits preserves the complete song');
    // Reproduce an older saved recipe after options load; submission must
    // still bound the pass even before the visible controls reconcile it.
    useStore.setState({slidingWindowSeconds: 40, slidingWindowOverlap: 18});
    const count = submissions.length;
    await useStore.getState().startGeneration('queue');
    assert.equal(submissions.length, count + 1, useStore.getState().promptEnhanceError);
    const longSong = submissions.at(-1).params;
    assert.equal(longSong.video_length, 1275, 'Window limits must not cut the requested timeline');
    assert.equal(longSong.sliding_window_size, 93);
    assert.equal(longSong.sliding_window_overlap, 13);
    assert.equal(longSong.sliding_window_discard_last_frames, 0);
  }

  uploadFailure = true;
  await expectBlocked({startImage: new File([], 'failed.png'), params: {image_start: 'stale.png', audio_guide: 'one.wav'}},
    'The Avatar anchor image could not be uploaded. Try selecting it again.');
  uploadFailure = false;

  // Model selection remembers Avatar independently of ordinary Video.
  seedAvatar();
  useStore.setState({studioVideoModelPerCreateRoute: {generate: 'longcat_video'}});
  useStore.getState().selectStudioVideoModel('longcat_avatar_multi');
  await new Promise(resolve => setImmediate(resolve));
  assert.equal(useStore.getState().studioVideoModelPerCreateRoute.avatar, 'longcat_avatar_multi');
  useStore.getState().setStudioVideoWorkflow('frames');
  await new Promise(resolve => setImmediate(resolve));
  assert.equal(useStore.getState().params.model_type, 'longcat_video');
  assert.equal(useStore.getState().studioVideoWorkflow, 'frames');
  useStore.getState().setStudioVideoWorkflow('avatar');
  await new Promise(resolve => setImmediate(resolve));
  assert.equal(useStore.getState().params.model_type, 'longcat_avatar_multi');
  assert.equal(useStore.getState().generationMode, 'video');
  assert.equal(useStore.getState().params.image_mode, 0);
  const remembered = JSON.parse(persisted.get('maestro_studio_video_create_route_v1'));
  assert.equal(remembered.models.avatar, 'longcat_avatar_multi');
  assert.equal(remembered.models.generate, 'longcat_video');

  for (const workflow of ['extend', 'blend']) {
    seedAvatar({params: {audio_prompt_type: 'AB', audio_guide2: 'prepared-second.wav'}});
    useStore.setState({studioVideoModelPerCreateRoute: {guided: 'longcat_video'}});
    useStore.getState().setStudioVideoWorkflow(workflow);
    await new Promise(resolve => setImmediate(resolve));
    assert.equal(useStore.getState().params.model_type, 'longcat_video');
    assert.equal(useStore.getState().studioVideoWorkflow, workflow);
    assert.equal(useStore.getState().params.image_mode, workflow === 'extend' ? 3 : 4);
    assert.ok(!useStore.getState().params.audio_prompt_type.includes('B'));
    assert.equal(useStore.getState().params.audio_guide2, 'prepared-second.wav');
  }
  seedAvatar({params: {image_start: 'anchor.png', audio_guide: 'one.wav'}});
  useStore.setState({studioVideoWorkflow: 'extend', enabledModels: new Set(['longcat_avatar']),
    params: {...useStore.getState().params, image_mode: 3}});
  const beforeWrongWorkflow = submissions.length;
  await useStore.getState().startGeneration('queue');
  assert.equal(submissions.length, beforeWrongWorkflow);
  assert.equal(useStore.getState().promptEnhanceError,
    'Choose a video model for this workflow. LongCat Avatar uses the Avatar workflow.');

  // Load Settings migrates older LongCat Frames recipes and preserves speaker order.
  seedAvatar();
  useStore.setState({studioVideoWorkflow: 'frames', selectedOutputMeta: {params: {
    ...multi, _studio_video_workflow: 'frames', model_type: 'longcat_avatar_multi',
    audio_guide: 'first.wav', audio_guide2: 'second.wav',
  }}});
  await useStore.getState().loadSettingsFromOutput();
  assert.equal(useStore.getState().studioVideoWorkflow, 'avatar');
  assert.equal(useStore.getState().generationMode, 'video');
  assert.equal(useStore.getState().params.audio_guide, 'first.wav');
  assert.equal(useStore.getState().params.audio_guide2, 'second.wav');
  assert.equal(useStore.getState().params.speakers_locations, multi.speakers_locations);
  assert.equal(useStore.getState().audioGuideFilename, 'first.wav');
  assert.equal(useStore.getState().audioGuide2Filename, 'second.wav');
  console.log('LongCat Avatar routing, guarded submission, short windows with full song duration, ' + submissions.length + ' intercepted queue payloads, and settings restoration passed');
})().catch(error => { console.error(error); process.exitCode = 1; });
