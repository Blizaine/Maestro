// Run with: node tests/ui/remote_model_memory.cjs
// Exercise the component and real API client with controlled server responses.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const root = path.resolve(__dirname, '../..');
const ts = require(path.join(root, 'ui/node_modules/typescript'));
const states = [];
let cursor = 0;
const react = {
  useId: () => 'loaded-model',
  useState(initial) {
    const index = cursor++;
    if (!(index in states)) states[index] = initial;
    return [states[index], value => { states[index] = value; }];
  },
};
const jsx = (type, props) => ({ type, props });
const calls = [];
let reply;
function load(relativePath, dependencies = {}) {
  const filename = path.join(root, relativePath);
  const module = { exports: {} };
  vm.runInNewContext(ts.transpileModule(fs.readFileSync(filename, 'utf8'), {
    compilerOptions: { module: ts.ModuleKind.CommonJS, jsx: ts.JsxEmit.ReactJSX },
  }).outputText, {
    module, exports: module.exports, Error,
    require: name => dependencies[name] || { jsx, jsxs: jsx },
    fetch: async (url, options) => { calls.push({ url, options }); return reply(); },
  }, { filename });
  return module.exports;
}
const api = load('ui/src/api/client.ts');
const { RemoteModelMemoryControls } = load(
  'ui/src/components/SettingsDrawer/RemoteModelMemoryControls.tsx',
  { react, '../../api/client': api },
);
let tree;
function render() { cursor = 0; tree = RemoteModelMemoryControls({ modelId: 'writer' }); }
function nodes(node = tree) {
  if (!node || typeof node !== 'object') return [];
  return [node, ...[].concat(node.props?.children || []).flatMap(child => nodes(child))];
}
const button = text => nodes().find(node => node.type === 'button' && node.props.children === text);
const select = () => nodes().find(node => node.type === 'select');
const textFor = role => nodes().find(node => node.props?.role === role)?.props.children;
const instance = id => ({ instance_id: id, model_key: 'writer', display_name: 'Writer' });
const ok = payload => ({ ok: true, json: async () => payload });

(async () => {
  render();
  assert.equal(calls.length, 0, 'Settings must not contact an external server until Refresh is clicked');
  let resolveList;
  reply = () => new Promise(resolve => { resolveList = resolve; });
  const refreshing = button('Refresh loaded models').props.onClick();
  render();
  assert.equal(button('Working…').props.disabled, true);
  assert.equal(select(), undefined);
  resolveList(ok({ server_url: 'http://localhost:1234', instances: [instance('first'), instance('second')] }));
  await refreshing; render();
  assert.equal(select().props.value, '', 'Multiple matching instances require an explicit choice');
  assert.equal(button('Unload selected model').props.disabled, true);
  select().props.onChange({ target: { value: 'second' } }); render();
  assert.equal(button('Unload selected model').props.disabled, false);

  let resolveUnload;
  reply = () => new Promise(resolve => { resolveUnload = resolve; });
  const unloading = button('Unload selected model').props.onClick(); render();
  assert.equal(select().props.disabled, true);
  assert.equal(button('Unload selected model').props.disabled, true);
  const request = calls.at(-1);
  assert.equal(request.url, '/api/v1/llm/remote/unload');
  assert.deepEqual(JSON.parse(request.options.body), {
    instance_id: 'second', server_url: 'http://localhost:1234',
  });
  assert.equal(request.options.headers.Authorization, undefined, 'Credentials stay in Maestro server configuration');
  resolveUnload({ ok: false, json: async () => ({ detail: 'The remote server changed. Refresh loaded models before unloading.' }) });
  await unloading; render();
  assert.match(textFor('alert'), /remote server changed/);
  assert.equal(select(), undefined, 'An uncertain selection requires a fresh list');
  assert.equal(textFor('status'), undefined, 'Failure cannot be presented as memory release');

  reply = async () => ok({ server_url: 'http://localhost:1234', instances: [instance('first')] });
  await button('Refresh loaded models').props.onClick(); render();
  assert.equal(select().props.value, 'first', 'A unique configured model can be preselected');
  reply = async () => ok({ status: 'unloaded' });
  await button('Unload selected model').props.onClick(); render();
  assert.match(textFor('status'), /confirmed.*unloaded/);
  assert.equal(textFor('alert'), undefined);
  assert.equal(select(), undefined);

  reply = async () => ok({ server_url: 'http://localhost:1234', instances: [] });
  await button('Refresh loaded models').props.onClick(); render();
  assert.match(textFor('status'), /no language models loaded/);
  assert.equal(calls.filter(call => call.options?.method === 'POST').length, 2);
  console.log('LM Studio memory controls: explicit refresh, instance selection, busy state, stale server rejection and verified success passed');
})().catch(error => { console.error(error); process.exitCode = 1; });
