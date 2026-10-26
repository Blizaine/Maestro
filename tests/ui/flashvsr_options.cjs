'use strict';
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const root = path.resolve(__dirname, '../..');
const ts = require(path.join(root, 'ui/node_modules/typescript'));
const source = fs.readFileSync(path.join(root, 'ui/src/lib/flashvsr.ts'), 'utf8');
const result = ts.transpileModule(source, {
  compilerOptions: { module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2020 },
  reportDiagnostics: true,
});
assert.equal((result.diagnostics || []).filter(d => d.category === ts.DiagnosticCategory.Error).length, 0);
const exportsObject = {};
vm.runInNewContext(result.outputText, { exports: exportsObject });
const { flashvsrOptions, flashvsrScale } = exportsObject;
assert.equal(flashvsrOptions.length, 10);
assert.equal(new Set(flashvsrOptions.map(option => option.value)).size, 10);
for (const prefix of ['flashvsr', 'flashvsrplus']) {
  for (const [suffix, scale] of [['2', 2], ['3', 3], ['4', 4], ['2pass2', 2], ['2pass4', 4]]) {
    const method = prefix + suffix;
    const option = flashvsrOptions.find(option => option.value === method);
    assert.ok(option);
    assert.equal(flashvsrScale(method), scale);
    assert.match(option.label, prefix.endsWith('plus') ? /Turbo/ : /v1\.1/);
  }
}
assert.equal(flashvsrScale(''), 1);
console.log('FlashVSR versioned selections and export dimensions passed');
