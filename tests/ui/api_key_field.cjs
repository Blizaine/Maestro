// Run with: node tests/ui/api_key_field.cjs
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const root = path.resolve(__dirname, '../..');
const ts = require(path.join(root, 'ui/node_modules/typescript'));
const filename = path.join(root, 'ui/src/components/shared/ApiKeyField.tsx');
const states = [];
let cursor = 0;
const react = {
  useId: () => 'key-field',
  useState(initial) {
    const i = cursor++;
    if (!(i in states)) states[i] = initial;
    return [states[i], value => { states[i] = value; }];
  },
};
const jsx = (type, props) => ({ type, props });
const mod = { exports: {} };
vm.runInNewContext(ts.transpileModule(fs.readFileSync(filename, 'utf8'), {
  compilerOptions: { module: ts.ModuleKind.CommonJS, jsx: ts.JsxEmit.ReactJSX },
}).outputText, {
  module: mod, exports: mod.exports, Error,
  require: name => name === 'react' ? react : { jsx, jsxs: jsx },
}, { filename });
const props = { label: 'CivitAI API Key', maskedValue: '', isSet: false };
let tree;
function render() { cursor = 0; tree = mod.exports.ApiKeyField(props); }
function nodes(node = tree) {
  if (!node || typeof node !== 'object') return [];
  return [node, ...[].concat(node.props?.children || []).flatMap(child => nodes(child))];
}
const find = (type, label) => nodes().find(n => n.type === type && (!label || n.props.children === label));
const flush = () => new Promise(resolve => setImmediate(resolve));

(async () => {
  let rejectSave;
  const attempts = [];
  props.onSave = value => { attempts.push(value); return new Promise((_, reject) => { rejectSave = reject; }); };
  render(); find('button', 'Set').props.onClick(); render();
  find('input').props.onChange({ target: { value: 'real...key-with-ellipsis' } }); render();
  find('button', 'Save').props.onClick(); render();
  assert.equal(find('input').props.value, 'real...key-with-ellipsis');
  assert.equal(find('button', 'Saving…').props.disabled, true);
  assert.equal(find('button', 'Cancel').props.disabled, true);
  rejectSave(new Error('Server unavailable')); await flush(); render();
  assert.equal(find('input').props.value, 'real...key-with-ellipsis', 'Failed save retains entered key');
  assert.equal(nodes().find(n => n.props?.role === 'alert').props.children, 'Server unavailable');
  assert.equal(find('button', 'Save').props.disabled, false);
  props.onSave = async value => { attempts.push(value); props.isSet = true; props.maskedValue = 'real...psis'; };
  render();
  find('button', 'Save').props.onClick(); await flush(); render();
  assert.equal(find('input'), undefined, 'Editor closes only after successful save');
  assert.ok(find('button', 'Change'));
  assert.ok(nodes().some(n => n.props?.children === 'real...psis'));
  assert.deepEqual(attempts, ['real...key-with-ellipsis', 'real...key-with-ellipsis']);
  console.log('API key field: pending, failure, retained value, and successful retry passed');
})().catch(error => { console.error(error); process.exitCode = 1; });
