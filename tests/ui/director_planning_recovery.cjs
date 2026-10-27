const assert = require('node:assert/strict');
const path = require('node:path');
const esbuild = require('../../ui/node_modules/esbuild');
const vm = require('node:vm');
const root = path.resolve(__dirname, '../..');
const built = esbuild.buildSync({stdin: {contents: `
  export {recommendAutoDuration} from './ui/src/lib/durationPlanning';
  export {useStore} from './ui/src/stores/useStore';
`, resolveDir: root}, bundle:true, platform:'node', format:'cjs', write:false, logLevel:'silent'});
const context = {module:{exports:{}}, exports:{}, require, process, console, setTimeout, clearTimeout,
  setInterval, clearInterval, URL, AbortController, fetch: async()=>({ok:true,json:async()=>({})})};
context.exports = context.module.exports;
vm.runInNewContext(built.outputFiles[0].text, context);
const {recommendAutoDuration, useStore} = context.module.exports;
for (const noun of ['commercial', 'advertisement', 'ad', 'promo']) {
  const plan = recommendAutoDuration(14.375, 0, 0, {prompt:`Create a 60-second ${noun}. ` + 'A new visual beat. '.repeat(100)});
  assert.equal(plan.requestedSeconds, 60, noun);
  assert.equal(plan.source, 'explicit_prompt');
}
const prompts = [{video_prompt:'Nora answers the after-hours call.'}];
useStore.setState({sidebarMode:'director', directorAudioFile:null, directorStep:'review',
  directorClipPlans:prompts, directorError:'Retained validation detail', loadDirectorQueue:async()=>{}});
useStore.getState().setSidebarMode('studio');
useStore.getState().setSidebarMode('director');
assert.equal(useStore.getState().directorStep, 'review');
assert.equal(useStore.getState().directorError, 'Retained validation detail');
assert.equal(useStore.getState().directorClipPlans, prompts);
console.log('PASS: 4 explicit ad durations and text-only Director navigation preserves work.');
