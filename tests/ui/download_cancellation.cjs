// Exercise download cancellation through the actual API client and UI components.
// Run: node tests/ui/download_cancellation.cjs
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const assert = require('node:assert/strict');
const ts = require(path.resolve(__dirname, '../../ui/node_modules/typescript'));

const root = path.resolve(__dirname, '../..');
let fetchHandler = async () => response({});

function response(body, {ok = true, status = 200} = {}) {
  return {ok, status, json: async () => body};
}

function loadTypeScript(relativePath, requireModule, globals = {}, transform = source => source) {
  const filename = path.join(root, relativePath);
  const source = transform(fs.readFileSync(filename, 'utf8').replaceAll('import.meta', '({})'));
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
    URL,
    AbortController,
    structuredClone,
    ...globals,
  }, {filename});
  return module.exports;
}

function createHookRuntime() {
  let cursor = 0;
  const state = [];
  const memo = [];
  const effectRecords = [];
  let pendingEffects = [];
  let dirty = false;
  let timerId = 0;
  const timers = new Map();

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

  function useRef(initialValue) {
    const index = cursor++;
    if (!(index in state)) state[index] = {current: initialValue};
    return state[index];
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

  const setTimer = callback => {
    const id = ++timerId;
    timers.set(id, callback);
    return id;
  };

  return {
    useState,
    useRef,
    useMemo,
    useCallback: (callback, deps) => useMemo(() => callback, deps),
    useEffect,
    beginRender,
    takeEffects: () => pendingEffects.splice(0),
    isDirty: () => dirty,
    runNextInterval: async () => {
      const entry = timers.entries().next().value;
      if (!entry) return false;
      entry[1]();
      await flush();
      return true;
    },
    timerGlobals: () => ({
      setInterval: setTimer,
      clearInterval: id => timers.delete(id),
      setTimeout: setTimer,
      clearTimeout: id => timers.delete(id),
      window: {
        setInterval: setTimer,
        clearInterval: id => timers.delete(id),
        setTimeout: setTimer,
        clearTimeout: id => timers.delete(id),
      },
    }),
  };
}

function jsx(type, props, key) {
  return {type, props: props || {}, key};
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

function containsNode(rootNode, target) {
  if (rootNode === target) return true;
  if (Array.isArray(rootNode)) return rootNode.some(child => containsNode(child, target));
  return !!rootNode && typeof rootNode === 'object' && containsNode(rootNode.props?.children, target);
}

async function flush() {
  await Promise.resolve();
  await new Promise(resolve => setImmediate(resolve));
}

async function renderUntilSettled(hooks, render) {
  let tree;
  for (let attempt = 0; attempt < 30; attempt++) {
    hooks.beginRender();
    tree = render();
    for (const runEffect of hooks.takeEffects()) runEffect();
    await flush();
    if (!hooks.isDirty()) return tree;
  }
  assert.fail('UI did not settle');
}

const api = loadTypeScript('ui/src/api/client.ts', () => {
  throw new Error('client.ts unexpectedly imported a runtime module');
}, {fetch: (url, init) => fetchHandler(url, init)});

function mockIcons() {
  return new Proxy({}, {get: (_target, name) => `icon:${String(name)}`});
}

function createSettingsHarness(initialStatus, options = {}) {
  const hooks = createHookRuntime();
  const model = {
    model_type: 'flux_model', name: 'Flux Model', architecture: 'flux2',
    is_downloaded: false, nsfw_only: false,
  };
  const family = {id: 'flux2', label: 'Flux 2'};
  let status = initialStatus;
  const calls = {download: [], cancel: [], loadModels: 0, loadModelOptions: [], toggle: 0};
  let cancelResolver;
  let cancelReject = false;
  let firstStatusPollResolver;
  let downloadStartResolver;
  let statusPollCalls = 0;
  const modelApi = {
    downloadModel: modelType => {
      calls.download.push(modelType);
      if (options.deferDownloadStart && calls.download.length === 1) {
        return new Promise(resolve => {
          downloadStartResolver = () => {
            status = {...status, status: 'downloading', error: null};
            resolve({status: 'downloading', model_type: modelType});
          };
        });
      }
      status = {...status, status: 'downloading', error: null};
      return Promise.resolve({status: 'downloading', model_type: modelType});
    },
    cancelModelDownload: modelType => {
      calls.cancel.push(modelType);
      if (cancelReject) return Promise.reject(new Error('Cancel endpoint is temporarily unavailable'));
      return new Promise(resolve => { cancelResolver = resolve; });
    },
    fetchModelDownloads: async () => {
      statusPollCalls += 1;
      if (options.deferFirstStatusPoll && statusPollCalls === 1) {
        return new Promise(resolve => { firstStatusPollResolver = resolve; });
      }
      return {downloads: {[model.model_type]: {...status}}};
    },
  };
  const store = {
    models: [model], families: [family], enabledModels: new Set(), servicesConfig: {nsfw_mode: false},
    modelVisibilityFocus: null,
    toggleModelEnabled: () => { calls.toggle += 1; },
    resetEnabledModels: () => {}, setAllModelsEnabled: () => {}, setModelsEnabled: () => {},
    loadModels: async options => { calls.loadModels += 1; calls.loadModelOptions.push(options); },
    clearModelVisibilityFocus: () => {},
  };
  const useStore = selector => selector(store);
  const modules = {
    react: hooks,
    'react/jsx-runtime': {jsx, jsxs: jsx, Fragment: 'fragment'},
    'lucide-react': mockIcons(),
    '../../types': {},
    '../../stores/useStore': {
      useStore,
      getFamiliesForMode: mode => mode === 'image' ? [family] : [],
      getModelsForFamily: (_familyId, models, mode) => mode === 'image' ? models : [],
    },
    '../../api/client': modelApi,
    '../../lib/theme': {FAMILIES: [], resolveVariant: () => null, onOsThemeChange: () => () => {}},
  };
  const component = loadTypeScript('ui/src/components/SettingsDrawer/SystemSettingsPanel.tsx', name => {
    if (!(name in modules)) throw new Error(`Unexpected settings import ${name}`);
    return modules[name];
  }, {
    ...hooks.timerGlobals(),
    localStorage: {getItem: () => null, setItem: () => {}},
    requestAnimationFrame: () => 1,
  }, source => source.replace('function ModelVisibilitySection()', 'export function ModelVisibilitySection()'));
  const render = () => component.ModelVisibilitySection();
  return {
    hooks, render, model, store, calls,
    setStatus: next => { status = next; },
    resolveCancel: result => cancelResolver(result),
    rejectCancel: () => { cancelReject = true; },
    resolveFirstStatusPoll: result => firstStatusPollResolver(result),
    resolveDownloadStart: () => downloadStartResolver(),
    statusPollCalls: () => statusPollCalls,
  };
}

async function testClientUsesOpaqueCancellationContracts() {
  const requests = [];
  fetchHandler = async (url, init = {}) => {
    requests.push({url, init});
    return response({status: 'cancelling', model_type: 'model/a'});
  };

  await api.cancelModelDownload('model/a');
  assert.equal(requests[0].url, '/api/v1/models/model%2Fa/download/cancel');
  assert.equal(requests[0].init.method, 'POST');

  await api.cancelDownload('opaque-cancel-42');
  assert.equal(requests[1].url, '/api/v1/downloads/cancel');
  assert.equal(requests[1].init.method, 'POST');
  assert.deepEqual(JSON.parse(requests[1].init.body), {download_id: 'opaque-cancel-42'});
}

async function testSettingsRetryCancelAcknowledgementAndReopen() {
  const harness = createSettingsHarness({status: 'cancelled', error: null, cancellable: false});
  let tree = await renderUntilSettled(harness.hooks, harness.render);
  const imageMode = findAll(tree, node => node.type === 'button' && textContent(node).includes('Image'))[0];
  imageMode.props.onClick();
  tree = await renderUntilSettled(harness.hooks, harness.render);

  let downloadButton = findAll(tree, node => node.type === 'button' && textContent(node).trim() === 'Retry')[0];
  assert.ok(downloadButton, 'Cancelled downloads can be retried');
  const modelLabel = findAll(tree, node => node.type === 'label' && textContent(node).includes('Flux Model'))[0];
  assert.ok(modelLabel, 'The enable checkbox keeps a separate label');
  assert.equal(containsNode(modelLabel, downloadButton), false, 'Download is outside the checkbox label');

  harness.setStatus({status: 'downloading', error: null, cancel_id: 'opaque-model-cancel', cancellable: true});
  await downloadButton.props.onClick();
  tree = await renderUntilSettled(harness.hooks, harness.render);
  let cancelButton = findAll(tree, node => node.type === 'button' && textContent(node).trim() === 'Cancel')[0];
  assert.ok(cancelButton, 'The active, server-cancellable model shows Cancel');
  assert.equal(harness.calls.toggle, 0, 'Download did not activate the model checkbox');

  const cancelPromise = cancelButton.props.onClick();
  tree = await renderUntilSettled(harness.hooks, harness.render);
  assert.match(textContent(tree), /Cancelling/);
  assert.match(textContent(tree), /Flux Model/);
  assert.equal(harness.calls.cancel[0], 'flux_model', 'Model cancellation uses its model endpoint key');
  harness.resolveCancel({status: 'cancelling', model_type: 'flux_model'});
  await cancelPromise;
  tree = await renderUntilSettled(harness.hooks, harness.render);
  assert.match(textContent(tree), /Cancelling/, 'The row stays pending while the worker stops');

  harness.setStatus({status: 'cancelled', error: null, cancellable: false});
  await harness.hooks.runNextInterval();
  tree = await renderUntilSettled(harness.hooks, harness.render);
  downloadButton = findAll(tree, node => node.type === 'button' && textContent(node).trim() === 'Retry')[0];
  assert.ok(downloadButton, 'The row becomes retryable after the backend acknowledges cancellation');
  assert.match(textContent(tree), /Cancelled/, 'Acknowledged cancellation gets a neutral status beside retry');
  assert.equal(findAll(tree, node => node.props?.role === 'alert').length, 0, 'Cancellation is not shown as a failed download');
  await downloadButton.props.onClick();
  tree = await renderUntilSettled(harness.hooks, harness.render);
  assert.deepEqual(harness.calls.download, ['flux_model', 'flux_model'], 'Retry after cancellation starts the model download again');
  assert.equal(harness.calls.toggle, 0, 'Retry still does not toggle model visibility');
  assert.equal(downloadButton.props.className.includes('min-h-[40px]'), true, 'The Settings download target is sized for touch');

  const reopened = createSettingsHarness({
    status: 'downloading', error: null, cancel_id: 'reopened-model-cancel', cancellable: true,
  });
  reopened.model.is_downloaded = true;
  let reopenedTree = await renderUntilSettled(reopened.hooks, reopened.render);
  const reopenedImageMode = findAll(reopenedTree, node => node.type === 'button' && textContent(node).includes('Image'))[0];
  reopenedImageMode.props.onClick();
  reopenedTree = await renderUntilSettled(reopened.hooks, reopened.render);
  const reopenedCancelButton = findAll(reopenedTree, node => node.type === 'button' && textContent(node).trim() === 'Cancel')[0];
  assert.ok(reopenedCancelButton,
    'An active transfer takes precedence over is_downloaded when Settings reopens');
  assert.equal(reopenedCancelButton.props.className.includes('min-h-[40px]'), true, 'The Settings cancel target is sized for touch');
  const reopenedDelete = findAll(reopenedTree, node => node.type === 'button' && node.props?.title === 'Delete model files')[0];
  assert.equal(reopenedDelete.props.disabled, true, 'Delete cannot race an active transfer');

  reopened.rejectCancel();
  const reopenedCancel = findAll(reopenedTree, node => node.type === 'button' && textContent(node).trim() === 'Cancel')[0];
  reopenedCancel.props.onClick();
  reopenedTree = await renderUntilSettled(reopened.hooks, reopened.render);
  assert.match(textContent(reopenedTree), /Cancel failed/);
  assert.ok(findAll(reopenedTree, node => node.type === 'button' && textContent(node).trim() === 'Cancel')[0],
    'A cancellation error leaves only that model row retryable');
}

async function testSettingsRefreshesOnlyCatalogAfterCompletion() {
  const harness = createSettingsHarness({status: 'completed', error: null, cancellable: false});
  let tree = await renderUntilSettled(harness.hooks, harness.render);
  findAll(tree, node => node.type === 'button' && textContent(node).includes('Image'))[0].props.onClick();
  tree = await renderUntilSettled(harness.hooks, harness.render);
  assert.match(textContent(tree), /Complete/);
  assert.equal(harness.calls.loadModelOptions.length, 1);
  assert.equal(harness.calls.loadModelOptions[0]?.catalogOnly, true,
    'Completion refresh preserves selected generation settings by refreshing only the catalog');
}

async function testDownloadedTransformerCanResumeCancelledCompanionFiles() {
  const harness = createSettingsHarness({status: 'cancelled', error: null, cancellable: false});
  harness.model.is_downloaded = true;
  let tree = await renderUntilSettled(harness.hooks, harness.render);
  findAll(tree, node => node.type === 'button' && textContent(node).includes('Image'))[0].props.onClick();
  tree = await renderUntilSettled(harness.hooks, harness.render);

  assert.match(textContent(tree), /Cancelled/);
  assert.doesNotMatch(textContent(tree), /Downloaded/,
    'A cancelled companion transfer takes precedence over the already-downloaded transformer');
  const retryButton = findAll(tree, node => node.type === 'button'
    && node.props?.['aria-label'] === 'Retry download model files for Flux Model')[0];
  assert.ok(retryButton, 'Cancelled companion assets remain retryable when transformer weights are present');
  await retryButton.props.onClick();
  tree = await renderUntilSettled(harness.hooks, harness.render);
  const deleteButton = findAll(tree, node => node.type === 'button' && node.props?.title === 'Delete model files')[0];
  assert.equal(deleteButton.props.disabled, true,
    'Delete is disabled while a companion download is active');
  assert.match(textContent(tree), /Downloading/);

  const failed = createSettingsHarness({status: 'failed', error: 'companion asset failed', cancellable: false});
  failed.model.is_downloaded = true;
  let failedTree = await renderUntilSettled(failed.hooks, failed.render);
  findAll(failedTree, node => node.type === 'button' && textContent(node).includes('Image'))[0].props.onClick();
  failedTree = await renderUntilSettled(failed.hooks, failed.render);
  assert.ok(findAll(failedTree, node => node.type === 'button'
    && node.props?.['aria-label'] === 'Retry download model files for Flux Model')[0],
  'A failed companion transfer remains retryable even when transformer weights are already downloaded');
}

async function testSettingsRearmsPollingAfterDownloadAcknowledgement() {
  const harness = createSettingsHarness(
    {status: 'cancelled', error: null, cancellable: false},
    {deferFirstStatusPoll: true, deferDownloadStart: true},
  );
  let tree = await renderUntilSettled(harness.hooks, harness.render);
  findAll(tree, node => node.type === 'button' && textContent(node).includes('Image'))[0].props.onClick();
  tree = await renderUntilSettled(harness.hooks, harness.render);
  const downloadButton = findAll(tree, node => node.type === 'button' && textContent(node).trim() === 'Download')[0];
  const startPromise = downloadButton.props.onClick();

  // The mount poll began before the POST and can still return its stale empty snapshot.
  harness.resolveFirstStatusPoll({downloads: {}});
  await flush();
  tree = await renderUntilSettled(harness.hooks, harness.render);
  assert.ok(findAll(tree, node => node.type === 'button' && textContent(node).trim() === 'Download')[0],
    'A stale empty status snapshot can temporarily clear the optimistic row');

  harness.resolveDownloadStart();
  await startPromise;
  tree = await renderUntilSettled(harness.hooks, harness.render);
  assert.match(textContent(tree), /Downloading/,
    'The POST acknowledgement restores the active row so polling cannot stop early');
  assert.ok(harness.statusPollCalls() >= 2, 'Polling resumes after the server acknowledges the download');
}

function makeActiveDownload(fileId, filename, cancelId, cancellable, status = 'downloading') {
  return {
    file_id: fileId, filename, started_at: 1, last_active_at: 1,
    downloaded_bytes: 50, total_bytes: 100, status, seconds_since_progress: 0,
    cancel_id: cancelId, cancellable,
  };
}

async function testGlobalBannerCancelsOnlyCancellableRow() {
  const hooks = createHookRuntime();
  const state = {
    downloads: [
      makeActiveDownload('file-a', 'alpha.safetensors', 'opaque-active-a', true),
      makeActiveDownload('file-b', 'beta.gguf', 'opaque-active-b', false),
      makeActiveDownload('file-c', 'gamma.safetensors', 'opaque-active-c', true),
    ],
  };
  const requests = [];
  let resolveAlpha;
  fetchHandler = (url, init = {}) => {
    requests.push({url, init});
    if (url === '/api/v1/downloads/active') return Promise.resolve(response({downloads: state.downloads}));
    const body = JSON.parse(init.body);
    if (body.download_id === 'opaque-active-a') {
      return new Promise(resolve => { resolveAlpha = resolve; });
    }
    if (body.download_id === 'opaque-active-c') {
      return Promise.resolve(response({detail: 'Already publishing'}, {ok: false, status: 409}));
    }
    return Promise.resolve(response({status: 'cancelling', download_id: body.download_id}));
  };
  const modules = {
    react: hooks,
    'react/jsx-runtime': {jsx, jsxs: jsx, Fragment: 'fragment'},
    'lucide-react': mockIcons(),
    '../api/client': api,
    '../lib/useLongPoll': {
      useLongPoll: (fetchSince, onData, onError) => {
        const callbacks = hooks.useRef({fetchSince, onData, onError});
        hooks.useEffect(() => { callbacks.current = {fetchSince, onData, onError}; });
        hooks.useEffect(() => {
          let active = true;
          const timers = hooks.timerGlobals();
          const tick = async () => {
            try {
              const current = callbacks.current;
              const result = await current.fetchSince(undefined, new AbortController().signal);
              if (active) current.onData(result);
            } catch (error) {
              if (active) callbacks.current.onError?.(error);
            }
          };
          void tick();
          const interval = timers.setInterval(() => { void tick(); }, 2000);
          return () => { active = false; timers.clearInterval(interval); };
        }, []);
      },
    },
  };
  const component = loadTypeScript('ui/src/components/DownloadStatusBanner.tsx', name => {
    if (!(name in modules)) throw new Error(`Unexpected banner import ${name}`);
    return modules[name];
  }, hooks.timerGlobals());
  const render = () => component.DownloadStatusBanner();

  let tree = await renderUntilSettled(hooks, render);
  assert.equal(findAll(tree, node => node.type === 'button' && textContent(node).trim() === 'Cancel').length, 2,
    'Only rows with server-provided cancellable=true and cancel_id expose cancellation');
  assert.ok(!textContent(tree).includes('Cancel download of beta.gguf'));

  const alphaCancel = findAll(tree, node => node.type === 'button' && node.props?.['aria-label'] === 'Cancel download of alpha.safetensors')[0];
  alphaCancel.props.onClick();
  tree = await renderUntilSettled(hooks, render);
  assert.match(textContent(tree), /Cancelling/);
  assert.match(textContent(tree), /alpha.safetensors/);
  assert.match(textContent(tree), /beta.gguf/);
  assert.deepEqual(JSON.parse(requests.find(item => item.url === '/api/v1/downloads/cancel').init.body),
    {download_id: 'opaque-active-a'});
  assert.equal(findAll(tree, node => node.type === 'button' && node.props?.['aria-label'] === 'Cancel download of beta.gguf').length, 0,
    'The unrelated non-cancellable row has no cancel action');

  resolveAlpha(response({status: 'cancelling', download_id: 'opaque-active-a'}));
  await flush();
  tree = await renderUntilSettled(hooks, render);
  assert.match(textContent(tree), /Cancelling/);
  assert.doesNotMatch(textContent(tree), /Cancelled/);

  state.downloads = [
    makeActiveDownload('file-a', 'alpha.safetensors', 'opaque-active-a', false, 'cancelled'),
    ...state.downloads.slice(1),
  ];
  await hooks.runNextInterval();
  tree = await renderUntilSettled(hooks, render);
  assert.match(textContent(tree), /Cancelled/);
  assert.doesNotMatch(textContent(tree), /interrupted/i, 'A confirmed cancellation is neutral, not an incomplete warning');
  assert.match(textContent(tree), /beta.gguf/, 'Other downloads remain visible');

  const gammaCancel = findAll(tree, node => node.type === 'button' && node.props?.['aria-label'] === 'Cancel download of gamma.safetensors')[0];
  gammaCancel.props.onClick();
  await flush();
  tree = await renderUntilSettled(hooks, render);
  assert.match(textContent(tree), /Cancel failed: Already publishing/);
  assert.ok(findAll(tree, node => node.type === 'button' && node.props?.['aria-label'] === 'Cancel download of gamma.safetensors')[0],
    'An HTTP 409 stays attached to the clicked transfer and leaves its action available');
  assert.equal(requests.filter(item => item.url === '/api/v1/downloads/cancel').length, 2,
    'No request is sent for the unrelated beta transfer');
}

function makeCivitDownload(id, filename, cancelId, cancellable, status = 'downloading') {
  return {
    id, filename, status, progress: 50, bytes_downloaded: 50, bytes_total: 100,
    error: null, started_at: 1, completed_at: null, cancel_id: cancelId, cancellable,
  };
}

async function testCivitDownloadBarShowsPendingAndNeutralTerminal() {
  const hooks = createHookRuntime();
  const store = {
    civitDownloads: [
      makeCivitDownload('civit-a', 'checkpoint-a.safetensors', 'opaque-civit-a', true),
      makeCivitDownload('civit-b', 'lora-b.safetensors', 'opaque-civit-b', true),
    ],
  };
  const requests = [];
  let resolveA;
  fetchHandler = (url, init = {}) => {
    requests.push({url, init});
    const body = JSON.parse(init.body);
    if (body.download_id === 'opaque-civit-a') return new Promise(resolve => { resolveA = resolve; });
    if (body.download_id === 'opaque-civit-b') {
      return Promise.resolve(response({detail: 'Already completed'}, {ok: false, status: 409}));
    }
    return Promise.resolve(response({status: 'cancelling', download_id: body.download_id}));
  };
  const modules = {
    react: hooks,
    'react/jsx-runtime': {jsx, jsxs: jsx, Fragment: 'fragment'},
    'lucide-react': mockIcons(),
    '../../stores/useStore': {useStore: selector => selector(store)},
    '../../api/client': api,
    '../../types': {},
  };
  const component = loadTypeScript('ui/src/components/LoraBrowser/DownloadBar.tsx', name => {
    if (!(name in modules)) throw new Error(`Unexpected DownloadBar import ${name}`);
    return modules[name];
  }, hooks.timerGlobals());
  const render = () => component.DownloadBar();

  let tree = await renderUntilSettled(hooks, render);
  const checkpointCancel = findAll(tree, node => node.type === 'button'
    && node.props?.['aria-label'] === 'Cancel download of checkpoint-a.safetensors')[0];
  assert.equal(checkpointCancel.props.className.includes('min-h-[40px]'), true, 'CivitAI Cancel target is sized for touch');
  checkpointCancel.props.onClick();
  tree = await renderUntilSettled(hooks, render);
  assert.match(textContent(tree), /Cancelling/);
  assert.match(textContent(tree), /checkpoint-a.safetensors/);
  assert.ok(findAll(tree, node => node.type === 'button'
    && node.props?.['aria-label'] === 'Cancel download of lora-b.safetensors')[0],
  'The second CivitAI transfer remains independently cancellable');
  assert.deepEqual(JSON.parse(requests[0].init.body), {download_id: 'opaque-civit-a'});

  resolveA(response({status: 'cancelling', download_id: 'opaque-civit-a'}));
  await flush();
  tree = await renderUntilSettled(hooks, render);
  assert.match(textContent(tree), /Cancelling/);
  assert.doesNotMatch(textContent(tree), /Cancelled/);

  store.civitDownloads = [
    {...makeCivitDownload('civit-a', 'checkpoint-a.safetensors', 'opaque-civit-a', false, 'cancelled'),
      completed_at: Date.now() / 1000 - 10},
    store.civitDownloads[1],
  ];
  tree = await renderUntilSettled(hooks, render);
  await hooks.runNextInterval();
  tree = await renderUntilSettled(hooks, render);
  assert.match(textContent(tree), /Cancelled/);
  assert.doesNotMatch(textContent(tree), /Download failed/);
  assert.ok(findAll(tree, node => node.type === 'button'
    && node.props?.['aria-label'] === 'Cancel download of lora-b.safetensors')[0]);

  const loraCancel = findAll(tree, node => node.type === 'button'
    && node.props?.['aria-label'] === 'Cancel download of lora-b.safetensors')[0];
  loraCancel.props.onClick();
  await flush();
  tree = await renderUntilSettled(hooks, render);
  assert.match(textContent(tree), /Cancel failed: Already completed/);
  assert.ok(findAll(tree, node => node.type === 'button'
    && node.props?.['aria-label'] === 'Cancel download of lora-b.safetensors')[0],
  'A cancellation conflict is shown on the matching transfer and can be retried');
  assert.deepEqual(requests.map(item => JSON.parse(item.init.body).download_id),
    ['opaque-civit-a', 'opaque-civit-b']);

  store.civitDownloads = [
    {...store.civitDownloads[0], completed_at: Date.now() / 1000 - 31},
    store.civitDownloads[1],
  ];
  tree = await renderUntilSettled(hooks, render);
  assert.doesNotMatch(textContent(tree), /checkpoint-a\.safetensors/,
    'Acknowledged cancellation history expires based on completed_at');
}

function createStoreHarness() {
  const modules = new Map();
  const storage = new Map();
  let nextTimer = 0;
  const timers = new Map();
  const snapshots = [
    {downloads: [{id: 'civit-cancelling', filename: 'stopping.safetensors', status: 'cancelling', completed_at: null}]},
    {downloads: [{id: 'civit-cancelling', filename: 'stopping.safetensors', status: 'cancelled', completed_at: null}]},
  ];
  let fetchCount = 0;
  const apiMock = new Proxy({
    fetchCivitAIDownloads: async () => snapshots[Math.min(fetchCount++, snapshots.length - 1)],
  }, {get: (target, name) => name in target ? target[name] : async () => ({})});

  function create(init) {
    let state;
    const get = () => state;
    const set = update => { state = {...state, ...(typeof update === 'function' ? update(state) : update)}; };
    state = init(set, get);
    return {getState: get, setState: set, subscribe: () => () => {}};
  }

  function load(file) {
    if (modules.has(file)) return modules.get(file).exports;
    const module = {exports: {}};
    modules.set(file, module);
    const source = fs.readFileSync(file, 'utf8').replaceAll('import.meta', '({})');
    const code = ts.transpileModule(source, {
      compilerOptions: {module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2022},
    }).outputText;
    vm.runInNewContext(code, {
      module, exports: module.exports, console, URL, AbortController, structuredClone,
      localStorage: {
        getItem: key => storage.get(key) || null,
        setItem: (key, value) => storage.set(key, value),
      },
      setTimeout: () => 1, clearTimeout: () => {}, setInterval: () => 1, clearInterval: () => {},
      window: {
        setTimeout: callback => { const id = ++nextTimer; timers.set(id, callback); return id; },
        clearTimeout: id => timers.delete(id),
      },
      require: name => name === 'zustand' ? {create}
        : name.endsWith('/api/client') ? apiMock
          : name.endsWith('/lib/theme') ? {getStoredPrefs: () => ({family: 'default', mode: 'dark'}), applyThemePrefs: () => {}}
            : name.startsWith('.') ? load(path.resolve(path.dirname(file), `${name}.ts`))
              : (() => { throw new Error(`Unexpected store import ${name}`); })(),
    }, {filename: file});
    return module.exports;
  }

  return {
    store: load(path.join(root, 'ui/src/stores/useStore.ts')).useStore,
    fetchCount: () => fetchCount,
    advanceTimer: async () => {
      const entry = timers.entries().next().value;
      assert.ok(entry, 'The poller schedules another snapshot while cancellation is pending');
      entry[1]();
      await flush();
    },
  };
}

async function testCivitPollerWaitsForCancellationAcknowledgement() {
  const harness = createStoreHarness();
  harness.store.getState().pollCivitAIDownloads();
  await flush();
  assert.equal(harness.fetchCount(), 1, 'The poller sees the pending cancellation');
  assert.equal(harness.store.getState().civitDownloads[0].status, 'cancelling');
  await harness.advanceTimer();
  assert.equal(harness.fetchCount(), 2, 'The poller continues until the worker acknowledges cancellation');
  assert.equal(harness.store.getState().civitDownloads[0].status, 'cancelled');
}

(async () => {
  await testClientUsesOpaqueCancellationContracts();
  await testSettingsRetryCancelAcknowledgementAndReopen();
  await testSettingsRefreshesOnlyCatalogAfterCompletion();
  await testDownloadedTransformerCanResumeCancelledCompanionFiles();
  await testSettingsRearmsPollingAfterDownloadAcknowledgement();
  await testGlobalBannerCancelsOnlyCancellableRow();
  await testCivitDownloadBarShowsPendingAndNeutralTerminal();
  await testCivitPollerWaitsForCancellationAcknowledgement();
  console.log('Model and generic downloads use opaque cancellation IDs, wait for server acknowledgement, preserve retry and unrelated rows, and show neutral cancellation states');
})().catch(error => {
  console.error(error);
  process.exitCode = 1;
});
