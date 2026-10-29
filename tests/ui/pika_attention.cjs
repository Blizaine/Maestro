const assert = require('node:assert/strict')
const fs = require('node:fs')
const path = require('node:path')
const vm = require('node:vm')
const ts = require('../../ui/node_modules/typescript')
const source = fs.readFileSync(path.resolve(__dirname, '../../ui/src/lib/h3Attention.ts'), 'utf8')
const compiled = ts.transpileModule(source, { compilerOptions: { module: ts.ModuleKind.CommonJS } }).outputText
const context = { exports: {} }
vm.runInNewContext(compiled, context)
const normalize = context.exports.normalizeH3Attention
const pika = { minimax_h3_pika_flex: true, sol_attention: true, sla_attention: true, sla_attention_default: false, sol_attention_status: { supported: true } }
for (const mode of ['sage2', 'sdpa', 'sol', 'sla']) assert.equal(normalize(pika, mode), mode)
assert.equal(normalize(pika, undefined), 'sage2')
assert.equal(normalize({ ...pika, sol_attention_status: { supported: false } }, 'sol'), 'sage2')
assert.equal(normalize({ ...pika, sla_attention_status: { supported: false } }, 'sla'), 'sla')
assert.equal(normalize({}, 'sage2'), undefined)
assert.equal(normalize({ sla_attention: true, sla_attention_default: true }, undefined), 'sla')
assert.equal(normalize({ sol_attention: true, sol_attention_status: { supported: true } }, 'sol'), 'sol')
const ordinaryH3 = { sol_attention: true, sla_attention: true, sla_attention_default: false, sol_attention_status: { supported: true } }
assert.equal(normalize(ordinaryH3, 'sage2'), '')
assert.equal(normalize({ ...ordinaryH3, sla_attention_default: true }, 'sage2'), 'sla')
console.log('Pika attention choices, dense default, sparse fallback and ordinary H3 normalization passed.')
