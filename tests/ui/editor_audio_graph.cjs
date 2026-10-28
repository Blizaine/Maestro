const assert=require('node:assert/strict');
const path=require('node:path');
const esbuild=require(path.resolve(__dirname,'../../ui/node_modules/esbuild'));
const built=esbuild.buildSync({entryPoints:[path.resolve(__dirname,'../../ui/src/editor/editorAudioGraph.ts')],bundle:true,write:false,platform:'node',format:'cjs',logLevel:'silent'});
let sources=0, contexts=0;
class Node {connect(){} disconnect(){} }
class Context {
 constructor(){contexts++;this.currentTime=0;this.state='running';this.destination={};}
 resume(){return Promise.resolve();}
 createGain(){const node=new Node();node.gain={setTargetAtTime(value){node.level=value;}};return node;}
 createAnalyser(){const node=new Node();node.getFloatTimeDomainData=values=>values.fill(.25);return node;}
 createMediaElementSource(){sources++;return new Node();}
}
global.window={AudioContext:Context};
const moduleResult={exports:{}};new Function('module','exports',built.outputFiles[0].text)(moduleResult,moduleResult.exports);
const graph=moduleResult.exports;
const media={dataset:{},volume:1};
graph.setEditorMediaGain(media,2);assert.equal(media.volume,1,'Native-only preview safely caps its volume');
graph.resumeEditorAudio([media]);assert.equal(contexts,1);assert.equal(sources,1);assert.equal(media.dataset.editorGain,'2','The mixer retains authored boost instead of clipping it to native volume');
graph.setEditorMediaGain(media,.5);assert.equal(media.volume,1,'Mixer gain owns attenuation');assert.equal(media.dataset.editorGain,'0.5');
graph.releaseEditorMediaAudio(media);graph.resumeEditorAudio([media]);assert.equal(sources,1,'Reconnect reuses its source node after StrictMode-style cleanup');
assert.equal(graph.editorMixPeak(),.25,'Meter samples the real master analyser');
console.log('Preview gain, attenuation, source reuse, gesture unlock and master peak sampling passed.');
