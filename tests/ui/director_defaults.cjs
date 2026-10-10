// Real store actions with isolated API responses; no live jobs or settings.
const assert=require('node:assert/strict'),path=require('node:path');
const root=path.resolve(__dirname,'../..'),esbuild=require(path.join(root,'ui/node_modules/esbuild'));
const {chromium}=require(process.env.MAESTRO_PLAYWRIGHT||'playwright');
const fused='minimax_h3_ref2va_fused_turbo',pruned='minimax_h3_ref2va',studio='ltx2_25';
const compatible={compatible:true};
const models=[fused,pruned,studio].map(model_type=>({model_type,name:model_type,family:model_type===studio?'ltx2':'minimax_h3',architecture:model_type,omni_reference:model_type!==studio,is_t2v:true,is_i2v:true,fps:24,guidance_max_phases:1,director:{video:{music_video:compatible,short_film_story:compatible,short_film_audio:compatible,seamless:compatible},shot_image_support:model_type===studio?'required':'direct_references'}}));
(async()=>{
 const bundle=await esbuild.build({stdin:{contents:"import {useStore,DEFAULT_DIRECTOR_VIDEO_MODEL} from './src/stores/useStore';window.store=useStore;window.directorDefault=DEFAULT_DIRECTOR_VIDEO_MODEL;",resolveDir:path.join(root,'ui'),loader:'ts'},bundle:true,write:false,define:{'process.env.NODE_ENV':'\"development\"'},logLevel:'silent'});
 const browser=await chromium.launch({headless:true,...(process.platform==='win32'?{executablePath:process.env.MAESTRO_CHROME||'C:/Program Files/Google/Chrome/Application/chrome.exe'}:{})});
 try{
  const page=await browser.newPage();const errors=[],requests=[];page.on('pageerror',e=>errors.push(e.message));
  await page.route('**/*',route=>{
   const request=route.request(),key=new URL(request.url()).pathname;requests.push({method:request.method(),key,body:request.method()==='POST'?request.postDataJSON():undefined});
   if(key==='/')return route.fulfill({contentType:'text/html',body:'<div></div>'});
   let body={entries:[],running:false,clips:[]};
   if(key.includes('/model-options/')){const model=decodeURIComponent(key.split('/').pop());body={...models.find(m=>m.model_type===model),frames_minimum:124,frames_maximum:345,frames_steps:17,default_num_inference_steps:model===fused?4:8};}
   return route.fulfill({json:body});
  });
  await page.goto('http://director-defaults.test');await page.addScriptTag({content:bundle.outputFiles[0].text});
  const seed=async(extra={})=>page.evaluate(({models,studio,extra})=>{
   const state=window.store.getState();state.directorReset();window.store.setState({models,modelsLoaded:true,enabledModels:new Set(models.map(m=>m.model_type)),modelVisibilityOverrides:{},selectedModelPerMode:{video:studio},params:{...state.params,model_type:studio},...extra});
  },{models,studio,extra});
  await seed();assert.equal(await page.evaluate(()=>window.directorDefault),fused);
  await page.evaluate(()=>{const s=window.store.getState();s.setSidebarMode('director');s.setDirectorSkill('music_video');});
  await page.waitForFunction(fused=>window.store.getState().selectedModelPerMode.video===fused,fused);
  assert.equal(await page.evaluate(()=>window.store.getState().params.model_type),studio,'Initializing Director does not select a different Studio model');
  await page.evaluate(pruned=>{const s=window.store.getState();s.selectDirectorVideoModel(pruned);s.setSidebarMode('studio');s.setSidebarMode('director');s.setDirectorSkill('short_film');},pruned);
  assert.equal(await page.evaluate(()=>window.store.getState().selectedModelPerMode.video),pruned,'Returning to a draft and switching project type preserve an explicit model choice');
  await seed();await page.evaluate(()=>window.store.getState().setDirectorSkill('short_film'));
  assert.equal(await page.evaluate(()=>window.store.getState().selectedModelPerMode.video),fused,'New short films also use Fused References');
  await page.evaluate(()=>{const s=window.store.getState();s.directorReset();s.setDirectorSkill('music_video');});
  assert.equal(await page.evaluate(()=>window.store.getState().selectedModelPerMode.video),fused,'A new project after Reset uses the default');
  await seed({modelVisibilityOverrides:{'video:references':{[fused]:false}}});await page.evaluate(()=>window.store.getState().setDirectorSkill('music_video'));
  assert.equal(await page.evaluate(()=>window.store.getState().selectedModelPerMode.video),studio,'A deliberately hidden Fused model is not forced back on');
  assert.equal(await page.evaluate(fused=>window.store.getState().modelVisibilityOverrides['video:references'][fused],fused),false);
  await seed({models:models.filter(m=>m.model_type!==fused)});await page.evaluate(()=>window.store.getState().setDirectorSkill('music_video'));
  assert.equal(await page.evaluate(()=>window.store.getState().selectedModelPerMode.video),studio,'Unavailable default does not replace an available choice');
  for(const extra of [{directorProjectId:'saved-project'},{directorSourcePipelineId:'saved-pipeline'},{directorQueueEditingEntryId:'queued-project'},{directorAudioPath:'/uploaded/song.wav'},{directorStep:'review_video',directorClipPlans:[{video_prompt:'Saved scene'}]}]){
   await seed({...extra,selectedModelPerMode:{video:pruned}});await page.evaluate(()=>window.store.getState().setDirectorSkill('music_video'));
   assert.equal(await page.evaluate(()=>window.store.getState().selectedModelPerMode.video),pruned,'Loaded work keeps its model: '+Object.keys(extra)[0]);
  }
  // The soundtrack planner must use the same new default if no selection exists.
  await seed({selectedModelPerMode:{},directorSkill:'music_video',directorAnalysis:{duration:30,bpm:120,sections:[],lyrics:[]},directorStep:'style'});
  await page.evaluate(()=>window.store.getState().directorSetEnergyBias(0));
  assert.equal(requests.filter(r=>r.key==='/api/v1/audio/plan-structure').at(-1)?.body.video_model,fused,'Timeline fallback agrees with the Director default');
  assert.deepEqual(errors,[]);assert.ok(!requests.some(r=>r.method==='DELETE'||r.key==='/api/v1/generate'||/\/download(?:\/|$)/.test(r.key)));
  console.log('Director defaults: new music/short-film projects, explicit choices, reset, hidden/unavailable models, restored work and timeline fallback passed');
 }finally{await browser.close();}
})().catch(error=>{console.error(error);process.exitCode=1});
