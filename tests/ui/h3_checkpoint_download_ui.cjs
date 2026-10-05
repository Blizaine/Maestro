// Exercise the H3 import form through the real API client and its HTTP error path.
// Run: node tests/ui/h3_checkpoint_download_ui.cjs
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const assert = require('node:assert/strict');
const ts = require(path.resolve(__dirname, '../../ui/node_modules/typescript'));

const root = path.resolve(__dirname, '../..');

function loadTypeScript(relativePath, requireModule, globals = {}) {
  const filename = path.join(root, relativePath);
  const source = fs.readFileSync(filename, 'utf8').replaceAll('import.meta', '({})');
  const code = ts.transpileModule(source, {
    compilerOptions: {
      module: ts.ModuleKind.CommonJS,
      target: ts.ScriptTarget.ES2022,
      jsx: ts.JsxEmit.ReactJSX,
      esModuleInterop: true,
    },
  }).outputText;
  const module = {exports: {}};
  vm.runInNewContext(code, {
    module,
    exports: module.exports,
    require: requireModule,
    console,
    Error,
    ...globals,
  }, {filename});
  return module.exports;
}

function createHookRuntime() {
  let cursor = 0;
  let state = [];
  let memo = [];
  let effectRecords = [];
  let pendingEffects = [];
  let dirty = false;

  function changed(previous, next) {
    return !previous || !next || previous.length !== next.length
      || previous.some((value, index) => !Object.is(value, next[index]));
  }

  function beginRender() {
    cursor = 0;
    pendingEffects = [];
    dirty = false;
  }

  function useState(initialValue) {
    const index = cursor++;
    if (!(index in state)) state[index] = typeof initialValue === 'function' ? initialValue() : initialValue;
    const setValue = nextValue => {
      const value = typeof nextValue === 'function' ? nextValue(state[index]) : nextValue;
      if (!Object.is(state[index], value)) {
        state[index] = value;
        dirty = true;
      }
    };
    return [state[index], setValue];
  }

  function useMemo(factory, deps) {
    const index = cursor++;
    const previous = memo[index];
    if (!previous || changed(previous.deps, deps)) memo[index] = {deps, value: factory()};
    return memo[index].value;
  }

  function useEffect(effect, deps) {
    const index = cursor++;
    const previous = effectRecords[index];
    if (!previous || changed(previous.deps, deps)) {
      previous?.cleanup?.();
      effectRecords[index] = {deps, cleanup: null};
      pendingEffects.push(() => {
        effectRecords[index].cleanup = effect() || null;
      });
    }
  }

  return {
    useState,
    useMemo,
    useCallback: (callback, deps) => useMemo(() => callback, deps),
    useEffect,
    beginRender,
    takeEffects: () => pendingEffects.splice(0),
    isDirty: () => dirty,
  };
}

function findAll(node, predicate, matches = []) {
  if (Array.isArray(node)) {
    for (const child of node) findAll(child, predicate, matches);
  } else if (node && typeof node === 'object') {
    if (predicate(node)) matches.push(node);
    findAll(node.props?.children, predicate, matches);
  }
  return matches;
}

function textContent(node) {
  if (node == null || typeof node === 'boolean') return '';
  if (Array.isArray(node)) return node.map(textContent).join('');
  if (typeof node === 'string' || typeof node === 'number') return String(node);
  return textContent(node.props?.children);
}

async function testH3ChoicesAreSubmittedAndHttp400IsVisible() {
  const hooks = createHookRuntime();
  const requests = [];
  const inspectionRequests = [];
  const errorText = 'Select the creator-listed H3 workflow and sampling recipe.';
  const api = loadTypeScript('ui/src/api/client.ts', () => {
    throw new Error('client.ts unexpectedly imported a runtime module');
  }, {
    fetch: async (url, init = {}) => {
      requests.push({url, init});
      return {ok: false, status: 400, json: async () => ({detail: errorText})};
    },
  });

  const architecture = {
    architecture: 'minimax_h3', name: 'MiniMax H3', family: 'minimax_h3',
    template_model_type: 'minimax_h3',
  };
  api.fetchLoraDirectories = async () => ({directories: []});
  api.inspectH3Checkpoint = async params => {
    inspectionRequests.push({...params});
    const needsSelection = [
      ...(params.h3_native_workflow === 'auto' ? ['native_workflow'] : []),
      ...(params.h3_sampling_profile === 'auto' ? ['sampling_profile'] : []),
    ];
    const choicesRemain = needsSelection.length > 0;
    return {
      supported: !choicesRemain,
      status: choicesRemain ? 'needs_selection' : 'verified',
      reason: choicesRemain ? 'Select the creator-listed workflow and sampling recipe.' : null,
      architectures: choicesRemain ? [] : [architecture],
      suggested_architecture: choicesRemain ? null : architecture.architecture,
      profile: {
        native_workflow: params.h3_native_workflow,
        sampling_profile: params.h3_sampling_profile,
        default_steps: 8,
        needs_selection: needsSelection,
      },
    };
  };

  const storeState = {
    civitDownloads: [],
    servicesConfig: {civitai_api_key_set: true},
    loraBrowserDefaultDir: '',
    setLoraBrowserOpen: () => {},
    setSettingsOpen: () => {},
    setSettingsTab: () => {},
    startCivitAIDownload: params => api.startCivitAIDownload(params),
  };
  const useStore = selector => selector(storeState);
  const jsx = (type, props, key) => ({type, props: props || {}, key});
  const reactRuntime = {...hooks};
  const mockedModules = {
    react: reactRuntime,
    'react/jsx-runtime': {jsx, jsxs: jsx, Fragment: 'fragment'},
    'lucide-react': new Proxy({}, {get: (_target, name) => `icon:${String(name)}`}),
    dompurify: {__esModule: true, default: {sanitize: value => value}},
    '../../api/client': api,
    '../../stores/useStore': {useStore},
    '../../lib/format': {formatBytes: value => String(value)},
  };
  const component = loadTypeScript('ui/src/components/LoraBrowser/ModelDetail.tsx', name => {
    if (!(name in mockedModules)) throw new Error(`Unexpected import ${name}`);
    return mockedModules[name];
  });
  const model = {
    id: 900001,
    name: 'H3 Fixture Checkpoint',
    type: 'Checkpoint',
    creator: {username: 'fixture_creator'},
    modelVersions: [{
      id: 900002,
      name: 'H3 Fixture Checkpoint',
      baseModel: 'MiniMax H3',
      files: [{
        id: 900003,
        name: 'h3-fixture.safetensors',
        downloadUrl: 'https://civitai.com/api/download/models/900003',
        sizeKB: 1024,
        metadata: {fp: 'bf16'},
      }],
      images: [],
      trainedWords: [],
    }],
  };

  async function renderUntilSettled() {
    let tree;
    for (let attempt = 0; attempt < 25; attempt++) {
      hooks.beginRender();
      tree = component.ModelDetail({model, onBack: () => {}, kind: 'checkpoint'});
      for (const runEffect of hooks.takeEffects()) runEffect();
      await new Promise(resolve => setImmediate(resolve));
      if (!hooks.isDirty()) return tree;
    }
    assert.fail('H3 checkpoint form did not settle');
  }

  let tree = await renderUntilSettled();
  assert.equal(inspectionRequests.length, 1, 'The selected H3 file is inspected before import');
  const initialImportButton = findAll(tree, element => element.type === 'button'
    && textContent(element).includes('Import h3-fixture.safetensors'))[0];
  assert.ok(initialImportButton, 'The import button is present before creator metadata is selected');
  assert.equal(initialImportButton.props.disabled, true, 'Unresolved H3 metadata keeps import disabled');

  const nativeLabel = findAll(tree, element => element.type === 'label'
    && textContent(element).includes('Native workflow (as listed by the creator)'))[0];
  const samplingLabel = findAll(tree, element => element.type === 'label'
    && textContent(element).includes('Sampling recipe (as listed by the creator)'))[0];
  const nativeSelect = findAll(nativeLabel, element => element.type === 'select')[0];
  const samplingSelect = findAll(samplingLabel, element => element.type === 'select')[0];
  assert.ok(nativeSelect, 'The backend-requested native workflow selector is shown');
  assert.ok(samplingSelect, 'The backend-requested sampling recipe selector is shown');
  nativeSelect.props.onChange({target: {value: 'ref2va'}});
  tree = await renderUntilSettled();
  assert.equal(inspectionRequests.length, 2, 'Choosing one creator option re-inspects the file');
  assert.equal(inspectionRequests.at(-1).h3_native_workflow, 'ref2va');
  assert.equal(inspectionRequests.at(-1).h3_sampling_profile, 'auto');
  const refreshedSamplingLabel = findAll(tree, element => element.type === 'label'
    && textContent(element).includes('Sampling recipe (as listed by the creator)'))[0];
  const refreshedSamplingSelect = findAll(refreshedSamplingLabel, element => element.type === 'select')[0];
  assert.ok(refreshedSamplingSelect, 'The remaining creator option stays available');
  const stillDisabledButton = findAll(tree, element => element.type === 'button'
    && textContent(element).includes('Import h3-fixture.safetensors'))[0];
  assert.equal(stillDisabledButton.props.disabled, true, 'Import stays disabled until all required metadata is selected');
  refreshedSamplingSelect.props.onChange({target: {value: 'turbo'}});

  tree = await renderUntilSettled();
  assert.deepEqual(inspectionRequests.at(-1), {
    model_id: 900001,
    version_id: 900002,
    file_id: 900003,
    h3_sampling_profile: 'turbo',
    h3_native_workflow: 'ref2va',
    h3_qkv_layout: 'auto',
  }, 'The selected H3 metadata is re-verified before enabling import');

  const importButton = findAll(tree, element => element.type === 'button'
    && textContent(element).includes('Import h3-fixture.safetensors'))[0];
  assert.ok(importButton, 'The import button is present after verification');
  assert.equal(importButton.props.disabled, false, 'Verified H3 selections enable import');
  await importButton.props.onClick();
  tree = await renderUntilSettled();

  assert.equal(requests.length, 1, 'Clicking Import sends one request');
  assert.equal(requests[0].url, '/api/v1/civitai/download');
  assert.equal(requests[0].init.method, 'POST');
  assert.equal(requests[0].init.headers['Content-Type'], 'application/json');
  const body = JSON.parse(requests[0].init.body);
  assert.equal(body.kind, 'checkpoint');
  assert.equal(body.model_id, 900001);
  assert.equal(body.version_id, 900002);
  assert.equal(body.file_id, 900003);
  assert.equal(body.target_architecture, 'minimax_h3');
  assert.equal(body.h3_native_workflow, 'ref2va');
  assert.equal(body.h3_sampling_profile, 'turbo');
  assert.equal(body.h3_qkv_layout, 'auto');

  const alert = findAll(tree, element => element.props?.role === 'alert')[0];
  assert.equal(textContent(alert), errorText, 'The HTTP 400 detail is visible inline');
}

testH3ChoicesAreSubmittedAndHttp400IsVisible().then(() => {
  console.log('H3 workflow and sampling selections reach the download API, and HTTP 400 detail is visible');
}).catch(error => {
  console.error(error);
  process.exitCode = 1;
});
