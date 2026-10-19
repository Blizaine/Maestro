const assert = require('node:assert/strict');
const path = require('node:path');
const esbuild = require(path.resolve('ui/node_modules/esbuild'));
(async () => {
 const bundle = await esbuild.build({entryPoints:['ui/src/lib/flux3Layout.ts'],bundle:true,platform:'node',format:'cjs',write:false});
 const module={exports:{}};new Function('module','exports',bundle.outputFiles[0].text)(module,module.exports);
 const {parseFlux3Layout:parse,serializeFlux3Layout:serialize,validBox}=module.exports;
 const original='# Keep this comment\nA kitten <kitten>.\n[{"id":"kitten","desc":"same kitten","from":"ref_image_0","src_bbox":[100,200,900,800],"tgt_bbox":null,"extra":"preserve me"}]';
 const parsed=parse(original);assert.equal(parsed.elements[0].extra,'preserve me');
 const out=serialize(parsed.caption,parsed.elements);assert.equal(parse(out).elements[0].tgt_bbox,null);assert.match(out,/# Keep this comment/);
 assert.deepEqual(parse('A red bicycle.').elements,[]);
 assert.throws(()=>parse('Broken [{"id":'),/incomplete/);
 assert.throws(()=>serialize('A tree <tree>',[{id:'tree',desc:'oak',bbox:[700,100,200,900]}]),/output box/);
 assert.throws(()=>serialize('A kitten',[{id:'kitten',desc:'kitten',bbox:[0,0,1000,1000]}]),/Add <kitten>/);
 assert.equal(validBox([0,0,1000,1000]),true);assert.equal(validBox([0,0,1001,1000]),false);
 const mixed=serialize('A cat <cat> under <title>',[{id:'cat',desc:'cat',from:'ref_image_0',src_bbox:[0,0,1000,1000],tgt_bbox:[100,100,900,900]},{id:'title',desc:'TITLE',bbox:[0,100,90,900]}]);
 assert.equal(parse(mixed).elements[1].from,'target');assert.equal(parse(mixed).elements[1].src_bbox,null);
 console.log('Flux3 layout round-trip, preserved extras, removal, Y/X, validation and mixed edit schema passed.');
})();
