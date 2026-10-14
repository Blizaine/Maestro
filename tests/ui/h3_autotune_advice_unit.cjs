// Run: node tests/ui/h3_autotune_advice_unit.cjs
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const root = path.resolve(__dirname, '../..');
const esbuild = require(path.join(root, 'ui/node_modules/esbuild'));
const source = fs.readFileSync(path.join(root,'ui/src/lib/h3PerformanceAdvice.ts'),'utf8');
const transformed = esbuild.transformSync(source,{loader:'ts',format:'cjs',target:'node22'});
const policy = JSON.parse(fs.readFileSync(path.join(root,'app/shared/h3_performance_policy.json'),'utf8'));
const loaded = {exports:{}};
new Function('module','exports','require',transformed.code)(loaded,loaded.exports,dependency=>
  dependency==='../../../app/shared/h3_performance_policy.json'?policy:require(dependency));
const {h3PerformanceAdvice} = loaded.exports;
const options = {model_type:'minimax_h3_ref2va_fused_turbo',architecture:'minimax_h3_ref2va',
  minimax_h3_fused_turbo:true,sla_attention_default:true,omni_reference:true,
  omni_sequence_memory_policy:{auto_resolution_pixels:{auto_1080p:1920*1088}}};
const input = (overrides={}) => ({options,resolution:'1920x1088',frames:243,
  totalVramGb:12,totalRamGb:31.76,steps:6,attention:'sla',...overrides});
for (const [resolution,frames,expected] of [
  ['1280x704',124,0],['1280x704',243,0],['1280x704',362,0],
  ['1920x1088',124,0],['1920x1088',243,1],['1920x1088',362,2],['1088x1920',362,2]]) {
  assert.equal(h3PerformanceAdvice(input({resolution,frames})).split,expected,`${resolution} / ${frames}f`);
}
assert.equal(h3PerformanceAdvice(input({resolution:'auto_1080p'})).split,1);
assert.equal(h3PerformanceAdvice(input({resolution:'unknown'})).split,null);
for (const totalVramGb of [6,8,10,16,24]) assert.equal(h3PerformanceAdvice(input({totalVramGb})),null);
for (const totalRamGb of [0,16,24,NaN]) assert.equal(h3PerformanceAdvice(input({totalRamGb})),null);
assert.equal(h3PerformanceAdvice(input({options:{...options,minimax_h3_full_checkpoint:true}})),null);
assert.equal(h3PerformanceAdvice(input({options:{...options,audio_only:true}})),null);
assert.equal(h3PerformanceAdvice(input({options:{...options,model_type:'viggle_animate'}})),null);
const densePrunedOptions={...options,minimax_h3_fused_turbo:false,sla_attention_default:false};
const pruned=h3PerformanceAdvice(input({options:densePrunedOptions,steps:20,attention:'sage2'}));
assert.equal(pruned.slow,true);
assert.equal(h3PerformanceAdvice(input({options:densePrunedOptions,steps:20,frames:226,attention:'sage2'})).slow,false);
assert.equal(h3PerformanceAdvice(input({options:densePrunedOptions,steps:20,frames:345,attention:'sage2'})).slow,true);
assert.ok(pruned.notes.some(text=>text.includes('116 minutes')&&text.includes('not an SLA or Veda sparse-attention ETA')));
for (const attention of ['sla','veda']) {
  const sparsePruned=h3PerformanceAdvice(input({options:{...options,minimax_h3_fused_turbo:false,sla_attention_default:false},
    steps:20,attention}));
  assert.equal(sparsePruned.slow,false,`${attention} must not inherit the dense/Sage2 duration warning`);
  assert.ok(!sparsePruned.notes.some(text=>text.includes('116 minutes')));
}
const fused=h3PerformanceAdvice(input());
assert.equal(fused.slow,false);
assert.ok(fused.notes.some(text=>text.includes('individual test results')));
assert.equal(h3PerformanceAdvice(input({frames:719})).experimental,true);
assert.equal(h3PerformanceAdvice(input({frames:719,resolution:'832x480'})).split,2);
assert.equal(h3PerformanceAdvice(input({frames:124,resolution:'2560x1440'})).split,2);
assert.equal(h3PerformanceAdvice(input({loraCount:1})).experimental,true);
assert.ok(h3PerformanceAdvice(input({referenceCount:2})).notes.some(text=>text.includes('conditioning')));
assert.ok(h3PerformanceAdvice(input({attention:'sage2'})).notes.some(text=>text.includes('tested recipe')));
assert.ok(h3PerformanceAdvice(input({steps:12})).notes.some(text=>text.includes('tested recipe')));
assert.equal(h3PerformanceAdvice(input({frames:Infinity})).split,null);
console.log('H3 Auto-tune UI: benchmark bands, scope, presets, warnings and unknown inputs passed');
