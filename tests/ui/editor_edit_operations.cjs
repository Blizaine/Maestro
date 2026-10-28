// Deterministic source-boundary and linked-edit checks; no app or media jobs.
const assert = require('node:assert/strict');
const path = require('node:path');
const esbuild = require(path.resolve(__dirname, '../../ui/node_modules/esbuild'));
const bundle = esbuild.buildSync({stdin:{contents:"export {applyEditorEditOperation} from './src/editor/editorEditOperations'; export {evaluateEditorProperty} from './src/editor/editorAnimation'",resolveDir:path.resolve(__dirname,'../../ui'),loader:'ts'},bundle:true,write:false,platform:'node',format:'cjs',logLevel:'silent'});
const built={exports:{}};new Function('module','exports',bundle.outputFiles[0].text)(built,built.exports);
const apply=built.exports.applyEditorEditOperation;
const evaluate=built.exports.evaluateEditorProperty;
const clone=value=>JSON.parse(JSON.stringify(value));
const item=(id,start,duration,source=2)=>({id,asset_id:'video',name:id,start,duration,source_in:source,speed:1,volume:1,opacity:1,fit:'contain',transform:{x:0,y:0,scale:1,rotation:0}});
const project=()=>({id:'ops',canvas:{fps:24,width:1920,height:1080},assets:{video:{type:'video',duration:30}},tracks:[{id:'v',type:'video',locked:false,items:[item('a',0,4),item('b',4,4),item('c',8,4)]}]});
{
 const p=project();apply(p,{type:'slip',itemId:'b',seconds:.5});
 assert.equal(p.tracks[0].items[1].source_in,2.5);assert.equal(p.tracks[0].items[1].start,4);assert.equal(p.tracks[0].items[1].duration,4);
 const before=clone(p);assert.throws(()=>apply(p,{type:'slip',itemId:'b',seconds:40}),/enough source/);assert.deepEqual(p,before);
}
{
 const p=project();apply(p,{type:'roll',itemId:'a',seconds:1});
 assert.equal(p.tracks[0].items[0].duration,5);assert.equal(p.tracks[0].items[1].start,5);assert.equal(p.tracks[0].items[1].duration,3);assert.equal(p.tracks[0].items[1].source_in,3);
 assert.equal(p.tracks[0].items[1].start+p.tracks[0].items[1].duration,8);
 const before=clone(p);assert.throws(()=>apply(p,{type:'roll',itemId:'a',seconds:10}),/enough source/);assert.deepEqual(p,before);
}
{
 const p=project();p.tracks[0].items[1].start=8;p.tracks[0].items[2].start=12;
 p.tracks[0].items[1].link_group_id='av';p.tracks.push({id:'sound',type:'audio',locked:false,items:[{...item('voice',8,4),link_group_id:'av'}]});
 apply(p,{type:'close-gap',itemId:'b'});assert.equal(p.tracks[0].items[1].start,4);assert.equal(p.tracks[0].items[2].start,8);assert.equal(p.tracks[1].items[0].start,4);
 assert.throws(()=>apply(p,{type:'close-gap',itemId:'b'}),/no gap/);
}
{
 const p=project();apply(p,{type:'crossfade',itemId:'b',seconds:.5});
 assert.equal(p.tracks[0].items[1].start,3.5);assert.equal(p.tracks[0].items[2].start,7.5);assert.equal(p.tracks[0].items[1].transition_overlap,.5);assert.equal(p.tracks[0].items[0].fade_out,0);
 apply(p,{type:'crossfade',itemId:'b',seconds:0});assert.equal(p.tracks[0].items[1].start,4);assert.equal(p.tracks[0].items[2].start,8);
 const before=clone(p);assert.throws(()=>apply(p,{type:'crossfade',itemId:'b',seconds:10}),/fit inside/);assert.deepEqual(p,before);
}
{
 const p=project();p.tracks[0].locked=true;const before=clone(p);
 assert.throws(()=>apply(p,{type:'slip',itemId:'a',seconds:1}),/Unlock/);assert.deepEqual(p,before);
}
{
 const p=project();apply(p,{type:'import-captions',name:'Captions',offset:1,cues:[{start:0,end:2,text:'First line\nsecond line'},{start:3,end:4,text:'Later'}]});
 const captions=p.tracks.find(track=>track.name==='Captions');assert.equal(captions.type,'text');assert.equal(captions.items.length,2);
 assert.equal(captions.items[0].caption,true);assert.equal(captions.items[0].start,1);assert.equal(captions.items[0].duration,2);
 assert.equal(captions.items[0].text,'First line\nsecond line');assert.equal(captions.items[0].style.y,346);assert.equal(captions.items[0].style.background_opacity,.5);
 const before=clone(p);assert.throws(()=>apply(p,{type:'import-captions',name:'Bad',cues:[{start:-2,end:1,text:'Invalid'}]}),/before the project/);assert.deepEqual(p,before);
 captions.locked=true;const locked=clone(p);assert.throws(()=>apply(p,{type:'import-captions',name:'Captions',cues:[{start:0,end:1,text:'Blocked'}]}),/Unlock/);assert.deepEqual(p,locked);
}
{
 const music=item('music',0,10);music.volume=.8;
 const narration={...item('voice-a',2,3),asset_id:'voice',name:'Voice A'};
 const later={...item('voice-b',4.5,2.5),asset_id:'voice',name:'Voice B'};
 const p={id:'audio',canvas:{fps:24,width:1920,height:1080},assets:{music:{type:'audio',duration:12,has_audio:true},voice:{type:'audio',duration:10,has_audio:true}},tracks:[
  {id:'music-track',type:'audio',locked:false,muted:false,items:[music]},
  {id:'voice-track',type:'audio',locked:false,muted:false,items:[narration,later]},
 ]};
 apply(p,{type:'duck-audio',itemId:'music',narrationTrackId:'voice-track',duckDb:12,attack:1,release:1});
 const edited=p.tracks[0].items[0];const ducked=.8*Math.pow(10,-12/20);
 assert.equal(edited.volume,.8);assert.deepEqual(edited.keyframes.volume.map(frame=>frame.time),[0,2,3,6,7,10]);
 assert.ok(Math.abs(evaluate(edited,'volume',0)-.8)<1e-6);assert.ok(Math.abs(evaluate(edited,'volume',2.5)-((.8+ducked)/2))<1e-6);
 assert.ok(Math.abs(evaluate(edited,'volume',3)-ducked)<1e-6);assert.ok(Math.abs(evaluate(edited,'volume',6)-ducked)<1e-6);
 assert.ok(Math.abs(evaluate(edited,'volume',6.5)-((ducked+.8)/2))<1e-6);assert.ok(Math.abs(evaluate(edited,'volume',7)-.8)<1e-6);
}
{
 const music=item('music',0,8);music.volume=1;const p={id:'audio',canvas:{fps:24,width:1920,height:1080},assets:{music:{type:'audio',duration:8,has_audio:true},voice:{type:'audio',duration:4,has_audio:true}},tracks:[
  {id:'music-track',type:'audio',locked:false,items:[music]},
  {id:'voice-track',type:'audio',locked:false,items:[{...item('voice',2,2),asset_id:'voice'}]},
 ]};
 apply(p,{type:'duck-audio',itemId:'music',narrationTrackId:'voice-track',duckDb:6,attack:0,release:0});
 const edited=p.tracks[0].items[0];const gain=Math.pow(10,-6/20);
 assert.equal(edited.keyframes.volume[0].easing,'hold');assert.equal(edited.keyframes.volume[1].value, gain);
 assert.ok(Math.abs(evaluate(edited,'volume',1.99)-1)<1e-6);assert.ok(Math.abs(evaluate(edited,'volume',2)-gain)<1e-6);
 assert.ok(Math.abs(evaluate(edited,'volume',3.99)-gain)<1e-6);assert.ok(Math.abs(evaluate(edited,'volume',4)-1)<1e-6);
}
{
 const music=item('music',0,5);const p={id:'audio',canvas:{fps:24,width:1,height:1},assets:{music:{type:'audio',duration:5,has_audio:true},voice:{type:'audio',duration:1,has_audio:true}},tracks:[
  {id:'music-track',type:'audio',locked:false,items:[music]},
  {id:'voice-track',type:'audio',locked:true,items:[{...item('voice',1,1),asset_id:'voice'}]},
 ]};
 const before=clone(p);assert.throws(()=>apply(p,{type:'duck-audio',itemId:'music',narrationTrackId:'voice-track',duckDb:12,attack:.2,release:.2}),/Unlock/);assert.deepEqual(p,before);
 p.tracks[1].locked=false;p.tracks[1].items[0].start=8;const noOverlap=clone(p);
 assert.throws(()=>apply(p,{type:'duck-audio',itemId:'music',narrationTrackId:'voice-track',duckDb:12,attack:.2,release:.2}),/no narration/);assert.deepEqual(p,noOverlap);
}
{
 const p=project();const clip=p.tracks[0].items[0];clip.keyframes={volume:[{time:0,value:2,easing:'linear'}]};
 apply(p,{type:'set-volume',itemId:'a',volume:1,envelopeGain:.5});assert.equal(clip.volume,1);assert.equal(clip.keyframes.volume[0].value,1);
 const before=clone(p);assert.throws(()=>apply(p,{type:'set-volume',itemId:'a',volume:5}),/between 0 and 4/);assert.deepEqual(p,before);
}
console.log('Slip, roll, linked gap closure, crossfade, caption import, duck envelopes, gain normalization, source boundaries and locked-track protection passed.');
