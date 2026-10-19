// Real React/Store coverage of measured defaults, late hardware detection and manual choices.
// Run: node tests/ui/h3_duration_defaults.cjs
const assert=require('node:assert/strict');
const fs=require('node:fs'); const path=require('node:path');
const root=path.resolve(__dirname,'../..');
const esbuild=require(path.join(root,'ui/node_modules/esbuild'));
const {chromium}=require(process.env.MAESTRO_PLAYWRIGHT || 'playwright');
const measured=JSON.parse(fs.readFileSync(path.join(root,'app/shared/h3_duration_recommendations.json'),'utf8'));
const memory=checkpoint=>({checkpoint,resource_recommendations:measured,
  reference_margin_steps:checkpoint.startsWith('fused')?0:1,
  auto_resolution_pixels:{auto_720p:1280*704,auto_1080p:1920*1088},
  resolution_bands:[{min_pixels:1800000,vram_tiers:[{max_vram_gb:12,frames:null},{frames:158}]},
    {min_pixels:800000,vram_tiers:[{max_vram_gb:12,frames:124},{frames:243}]},
    {min_pixels:0,vram_tiers:[{frames:345}]}]});
const model=(reference=false,pruned=false)=>({model_type:reference?(pruned?'minimax_h3_ref2va':'minimax_h3_ref2va_fused_turbo'):pruned?'minimax_h3_fl2va':'minimax_h3_fused_turbo',
  architecture:reference?'minimax_h3_ref2va':'minimax_h3',omni_reference:reference,minimax_h3_fused_turbo:!pruned,
  guidance_max_phases:1,fps:24,frames_minimum:124,frames_maximum:345,frames_steps:17,sliding_window:true,
  sliding_window_defaults:{window_min:124,window_max:345,window_step:17,window_default:243,overlap_default:18,discard_last_frames:0},
  resolution_preset_order:['720p','1080p'],resolution_presets:{'720p':{values:{'16:9':'1280x704'}},'1080p':{values:{'16:9':'1920x1088'}}},
  sliding_window_memory_policy:memory(pruned?'pruned':'fused_4step'),
  omni_sequence_memory_policy:memory(pruned?'pruned':'fused_4step_references'),
  director_memory_policy:memory(pruned?'pruned':reference?'fused_4step_references':'fused_4step')});
(async()=>{
 const bundle=await esbuild.build({stdin:{contents:`
 import React from 'react';import {createRoot} from 'react-dom/client';
 import {useStore} from './src/stores/useStore';
 import * as h3 from './src/lib/h3Memory';
 import {DurationSlider} from './src/components/Sidebar/DurationSlider';
 import {DirectorPanel} from './src/components/Sidebar/DirectorPanel';
 import {DirectorMusicClipLength} from './src/components/Sidebar/DirectorMusicClipLength';
 import {DirectorChat} from './src/components/Sidebar/DirectorChat';
 window.store=useStore;window.h3=h3;window.root=createRoot(document.getElementById('root'));
 window.mount=kind=>window.root.render(kind==='studio'?<DurationSlider includeWindowSettings/>:kind==='director-chat'?<DirectorChat key={useStore.getState().selectedModelPerMode.video}/>:<><DirectorMusicClipLength/><DirectorPanel/></>);
 `,resolveDir:path.join(root,'ui'),loader:'tsx'},bundle:true,write:false,jsx:'automatic',define:{'process.env.NODE_ENV':'"development"'},logLevel:'silent'});
 const browser=await chromium.launch({headless:true,args:['--disable-gpu'],...(process.platform==='win32'?{executablePath:process.env.MAESTRO_CHROME||'C:/Program Files/Google/Chrome/Application/chrome.exe'}:{})});
 try{
  const page=await browser.newPage({viewport:{width:390,height:844}});const errors=[],requests=[];
  page.on('pageerror',e=>errors.push(e.message));let selected=model();
  await page.route('**/*',async route=>{
   const url=new URL(route.request().url());let body={};
   if(url.pathname==='/')return route.fulfill({contentType:'text/html',body:'<div id="root"></div>'});
   if(url.pathname.includes('model-options'))body=selected;
   if(url.pathname.includes('music-clip-limits')){
    const requested=route.request().postDataJSON();
    const profile=await page.evaluate(({requested,selected})=>{const s=window.store.getState();return window.h3.recommendedH3PassProfile(selected.director_memory_policy,requested.video_params.resolution,s.systemStats.gpu.vram_total_gb,s.systemStats.ram.total_gb)},{requested,selected});
    const frames=profile?.frames||124;
    body={fps:24,frames_minimum:124,frame_step:17,hard_max_frames:345,recommended_frames:frames,recommended_seconds:frames/24,max_frames:requested.director_max_shot_frames||frames,max_seconds:(requested.director_max_shot_frames||frames)/24,auto:true,
     recommendation_basis:'reported',recommendation_note:'Director test fixture: reported 720p runs provide a starting point.',recommendation_version:'director-fixture-v1'};
   }
   if(url.pathname.includes('audio/plan-structure')){requests.push(route.request().postDataJSON());body={clips:[]};}
   return route.fulfill({contentType:'application/json',body:JSON.stringify(body)});
  });
  await page.goto('http://duration-defaults.test');await page.addScriptTag({content:bundle.outputFiles[0].text});
  const unit=await page.evaluate(({memory,measured})=>{
   const p={...memory,resource_recommendations:measured};const fn=window.h3.recommendedH3PassProfile;
   return [fn(p,'1280x704',12,31.76).frames,fn(p,'1280x704',12,16).frames,fn(p,'1280x704',12,0).frames,
    fn(p,'1280x704',8,32).frames,fn(p,'1920x1088',12,31.76).frames,
    fn(p,'1280x704',24,128).frames,fn(p,'auto_720p',12,31.76).frames];
  },{memory:memory('fused_4step_references'),measured});
  assert.deepEqual(unit,[345,124,124,243,124,345,345]);
  await page.evaluate(selected=>{const s=window.store.getState();window.store.setState({generationMode:'video',studioVideoWorkflow:'frames',
   modelOptions:selected,systemStats:{gpu:{vram_total_gb:12},ram:{total_gb:0}},durationSeconds:124/24,slidingWindowSeconds:124/24,
   slidingWindowLocked:false,h3WindowOverrides:{},resolutionPreset:'720p',aspectRatio:'16:9',
   params:{...s.params,model_type:selected.model_type,resolution:'1280x704',prompt:'A kitten flies on a dragon.',video_length:124,
   sliding_window_size:124,_duration_planning_mode:'auto',minimax_h3_multi_window:false,minimax_h3_extended_duration:false}});window.mount('studio');},selected);
  await page.waitForTimeout(100);
  await page.evaluate(()=>window.store.setState({systemStats:{gpu:{vram_total_gb:12},ram:{total_gb:31.76}}}));
  await page.waitForFunction(()=>window.store.getState().params.video_length===345);
  assert.equal(await page.evaluate(()=>window.store.getState().params.sliding_window_size),345,'Auto changes stale short windows when RAM detection arrives');
  await page.evaluate(()=>{const s=window.store.getState();s.setParam('_duration_planning_mode','duration');s.setDurationSeconds(124/24);window.store.setState({systemStats:{gpu:{vram_total_gb:24},ram:{total_gb:128}}});});
  await page.waitForTimeout(100);assert.equal(await page.evaluate(()=>window.store.getState().params.video_length),124,'Manual Time stays short');
  await page.evaluate(()=>{const s=window.store.getState();s.setSlidingWindowLocked(true);s.setSlidingWindowSeconds(243/24);s.setParam('_duration_planning_mode','windows');s.setDurationSeconds(243/24);});
  await page.waitForTimeout(100);assert.equal(await page.evaluate(()=>window.store.getState().params.sliding_window_size),243,'Manual window survives resources');
  await page.evaluate(()=>{const s=window.store.getState();s.setSlidingWindowLocked(false);s.setParam('_duration_planning_mode','auto');window.store.setState({systemStats:{gpu:{vram_total_gb:12},ram:{total_gb:31.76}}});s.setResolutionPreset('1080p');});
  await page.waitForFunction(()=>window.store.getState().params.video_length===124);
  assert.equal(await page.evaluate(()=>window.store.getState().params.sliding_window_size),124,'1080p Auto is conservative');
  selected=model(true);
  await page.evaluate(async selected=>{const s=window.store.getState();window.store.setState({studioVideoWorkflow:'references',params:{...s.params,model_type:selected.model_type},modelOptions:null,resolutionPreset:'720p'});await window.store.getState().loadModelOptions(selected.model_type);},selected);
  await page.waitForFunction(()=>window.store.getState().params.video_length===345);
  assert.equal(await page.evaluate(()=>window.store.getState().slidingWindowSeconds*24),345,'References gets the new longer cap');
  selected=model(false,true);
  await page.evaluate(async selected=>{const s=window.store.getState();window.store.setState({studioVideoWorkflow:'frames',params:{...s.params,model_type:selected.model_type},modelOptions:null});await window.store.getState().loadModelOptions(selected.model_type);},selected);
  await page.waitForFunction(()=>window.store.getState().params.video_length===345);
  assert.equal(await page.evaluate(()=>window.store.getState().slidingWindowSeconds*24),345,'Pruned Frames uses the repeatedly tested 14.4s cap');
  selected=model(true,true);
  await page.evaluate(async selected=>{const s=window.store.getState();window.store.setState({studioVideoWorkflow:'references',params:{...s.params,model_type:selected.model_type},modelOptions:null});await window.store.getState().loadModelOptions(selected.model_type);},selected);
  await page.waitForFunction(()=>window.store.getState().params.video_length===345);
  assert.equal(await page.evaluate(()=>window.store.getState().slidingWindowSeconds*24),345,'Pruned References includes tested reference headroom');
  const referenceProfiles=await page.evaluate(policy=>[
   window.h3.recommendedH3OmniSequenceProfile(policy,'1280x704',12,124,345,17,31.76),
   window.h3.recommendedH3OmniSequenceProfile(policy,'1280x704',12,124,345,17,16),
  ],memory('pruned'));
  assert.equal(referenceProfiles[0].frames,345);
  assert.equal(referenceProfiles[0].referenceMarginFrames,0);
  assert.equal(referenceProfiles[1].frames,124,'Low RAM stays on the conservative fallback');
  // The 8 GB laptop completed 10.125s Fused Frames and References clips.
  // Exercise the real Studio Auto controls, not just a direct matrix lookup.
  for (const reference of [false,true]) {
   selected=model(reference);
   await page.evaluate(async ({selected,reference})=>{const s=window.store.getState();window.store.setState({
    generationMode:'video',studioVideoWorkflow:reference?'references':'frames',modelOptions:null,
    systemStats:{gpu:{vram_total_gb:7.96},ram:{total_gb:31.02}},
    durationSeconds:124/24,slidingWindowSeconds:124/24,slidingWindowLocked:false,h3WindowOverrides:{},
    resolutionPreset:'720p',aspectRatio:'16:9',params:{...s.params,model_type:selected.model_type,
     resolution:'1280x704',video_length:124,sliding_window_size:124,_duration_planning_mode:'auto',
     minimax_h3_multi_window:false,minimax_h3_extended_duration:false},
   });await window.store.getState().loadModelOptions(selected.model_type);}, {selected,reference});
   await page.waitForFunction(()=>window.store.getState().params.video_length===243);
   assert.equal(await page.evaluate(()=>window.store.getState().params.sliding_window_size),243,
    `8 GB / 32 GB Fused ${reference?'References':'Frames'} Auto uses the verified 10.1s window`);
  }
  await page.evaluate(()=>window.store.setState({systemStats:{gpu:{vram_total_gb:12},ram:{total_gb:31.76}}}));
  selected=model(true);
  await page.evaluate(selected=>{const s=window.store.getState();s.directorReset();window.store.setState({selectedModelPerMode:{video:selected.model_type},
   directorSkill:'music_video',directorResolution:'720p',directorAspectRatio:'16:9',directorSeamless:false,directorStep:'structure',
   directorAnalysis:{duration:255,bpm:120,sections:[],lyrics:[]},directorSongDuration:255,directorPlannedClips:[],directorVideoMaxShotFramesByModel:{},
   directorMusicClipSeconds:null,directorLoading:false,directorError:null});window.mount('director');},selected);
  const music=page.getByRole('group',{name:'Music video clip length'});await music.getByRole('slider').waitFor();
  assert.equal(Number(await music.getByRole('slider').inputValue()),345);
  const musicRecommendation=music.getByTestId('director-music-recommendation');
  assert.match(await musicRecommendation.locator('summary').innerText(),/Reported recommendation/);
  await musicRecommendation.locator('summary').click();
  assert.match(await musicRecommendation.innerText(),/reported 720p runs provide a starting point/);
  assert.match(await musicRecommendation.innerText(),/Guidance updated director-fixture-v1/);
  const laptopLimits=page.waitForResponse(response=>response.url().includes('music-clip-limits')&&response.request().method()==='POST');
  await page.evaluate(()=>window.store.setState({systemStats:{gpu:{vram_total_gb:7.96},ram:{total_gb:31.02}}}));
  await laptopLimits;
  await page.waitForFunction(()=>document.querySelector('[aria-label="Music video clip length"] input[type=range]')?.value==='243');
  assert.equal(Number(await music.getByRole('slider').inputValue()),243,
   'Director music Auto uses the same verified 10.1s Fused recommendation on the 8 GB laptop');
  const musicSlider=music.getByRole('slider');await musicSlider.focus();await musicSlider.press('End');
  const manualMusicSeconds=await page.evaluate(()=>window.store.getState().directorMusicClipSeconds);
  assert.ok(manualMusicSeconds>0,'moving the music clip slider stores a manual choice');
  const refreshedLimits=page.waitForResponse(response=>response.url().includes('music-clip-limits')&&response.request().method()==='POST');
  await page.evaluate(()=>window.store.setState({systemStats:{gpu:{vram_total_gb:8},ram:{total_gb:16}}}));
  await refreshedLimits;
  await musicRecommendation.waitFor();
  assert.equal(await page.evaluate(()=>window.store.getState().directorMusicClipSeconds),manualMusicSeconds,
   'resource updates refresh recommendation details without replacing a manual clip length');
  await music.getByRole('button',{name:'Auto'}).click();
  const auto=page.getByRole('button',{name:'Automatic cut speed'});assert.equal(await auto.getAttribute('aria-pressed'),'true');
  const cut=page.locator('input[type=range]').last();assert.equal(await cut.inputValue(),'-2');await cut.focus();await cut.press('ArrowRight');
  await page.waitForFunction(()=>window.store.getState().directorEnergyBias===-1 && !window.store.getState().directorLoading);assert.equal(requests.at(-1).energy_bias,-1);
  await auto.click();await page.waitForFunction(()=>window.store.getState().directorEnergyBias===null && !window.store.getState().directorLoading);assert.equal(requests.at(-1).energy_bias,null);

  // Real Director setup controls used to omit RAM, bypassing measured tiers.
  for (const [reference,pruned] of [[false,false],[true,false],[false,true],[true,true]]) {
   selected=model(reference,pruned);
   await page.evaluate(selected=>{const s=window.store.getState();s.directorReset();window.store.setState({
    selectedModelPerMode:{video:selected.model_type},directorSkill:'short_film',shortFilmPath:'story',directorStep:'style',
    directorResolution:'720p',directorAspectRatio:'16:9',directorSceneDescription:'',directorLoading:false,directorError:null,
    directorVideoMaxShotFramesByModel:{},systemStats:{gpu:{vram_total_gb:12},ram:{total_gb:0}},
   });window.mount('director-chat');},selected);
   await page.getByText(/Auto max shot 5\.2s on 12 GB/).waitFor();
   await page.getByRole('button',{name:'Advanced',exact:true}).click();
   const gpuRecommendation=page.getByTestId('director-gpu-recommendation');
   assert.match(await gpuRecommendation.locator('summary').innerText(),/Reported recommendation/);
   await gpuRecommendation.locator('summary').click();
   assert.match(await gpuRecommendation.innerText(),/reported 720p runs provide a starting point/);
   assert.match(await gpuRecommendation.innerText(),/Guidance updated director-fixture-v1/);
   await page.getByRole('option',{name:'Auto · 5.2s',exact:true}).waitFor({state:'attached'});
   await page.evaluate(()=>window.store.setState({systemStats:{gpu:{vram_total_gb:12},ram:{total_gb:31.76}}}));
   await page.getByText(/Auto max shot 14\.4s on 12 GB/).waitFor();
   await page.getByRole('option',{name:'Auto · 14.4s',exact:true}).waitFor({state:'attached'});
   await page.getByRole('button',{name:'Window',exact:true}).click();
   await page.getByRole('spinbutton',{name:'Window count',exact:true}).fill('1');
   await page.waitForFunction(()=>window.store.getState().shortFilmTargetDuration===345/24);
   await page.getByText(/current automatic shot target is 14\.4s/).waitFor();
  }
  assert.deepEqual(errors,[]);
  console.log('H3 duration defaults: measured 14.4s Pruned/Fused Frames/References, reference headroom, RAM/VRAM scopes, late detection in Studio/Director, manual Time/windows, conservative1080p and Director Auto/manual pacing passed');
 }finally{await browser.close();}
})().catch(error=>{console.error(error);process.exitCode=1});
