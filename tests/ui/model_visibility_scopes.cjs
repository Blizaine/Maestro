// Actual React settings and Zustand persistence across browser origins; no real app requests.
const assert=require('node:assert/strict'), fs=require('node:fs'), path=require('node:path');
const root=path.resolve(__dirname,'../..');
const esbuild=require(path.join(root,'ui/node_modules/esbuild'));
const {chromium}=require(process.env.MAESTRO_PLAYWRIGHT || 'playwright');
const make=(model_type,name,extra={})=>({model_type,name,family:'minimax_h3',architecture:model_type,is_t2v:true,is_i2v:true,is_downloaded:true,guidance_max_phases:1,fps:24,...extra});
const models=[make('minimax_h3','H3 Pruned 20B — Frames'), make('minimax_h3_full','H3 Full 33B — First/Last'),
 make('minimax_h3_fused_turbo','H3 Fused 4-Step — Frames'),
 make('ltx2_25','LTX-2.5 — Distilled',{family:'ltx2'}),
 make('longcat_avatar','LongCat Avatar Single',{family:'longcat',supports_audio_input:true}),
 make('longcat_avatar_multi','LongCat Avatar Multi',{family:'longcat',supports_audio_input:true}),
 make('minimax_h3_ref2va','H3 Omni Pruned 20B — References',{omni_reference:true}),
 make('minimax_h3_ref2va_fused_turbo','H3 Fused 4-Step — References',{omni_reference:true}),
 make('minimax_h3_ref2va_imported','H3 Custom INT8 — References',{omni_reference:true}),
 make('minimax_h3_ref2va_mature','H3 Mature — References',{omni_reference:true,nsfw_only:true}),
 make('flux_dev','Flux Dev',{family:'flux',architecture:'flux',is_downloaded:true,is_t2v:false,is_i2v:false}),
 make('flux_schnell','Flux Schnell',{family:'flux',architecture:'flux',is_downloaded:false,is_t2v:false,is_i2v:false}),
 make('yue2','YuE2',{family:'tts',audio_only:true}), make('minimax_music3','MiniMax Music3',{family:'tts',audio_only:true})];
models[models.length-1].is_downloaded=false;
const families=[{id:'minimax_h3',label:'MiniMax H3'},{id:'ltx2',label:'LTX-2'},{id:'longcat',label:'LongCat'},{id:'flux',label:'FLUX'},{id:'tts',label:'Audio'}];
(async()=>{
 const bundle=await esbuild.build({stdin:{contents:`import React from 'react';import {createRoot} from 'react-dom/client';
 import {useStore,getModelGroupsForVisibilityScope,enabledModelsForContext} from './src/stores/useStore';
 import {ModelVisibilitySection} from './src/components/SettingsDrawer/SystemSettingsPanel';
 import {GenerateButton} from './src/components/Sidebar/GenerateButton';
 import {ModelSelector} from './src/components/Sidebar/ModelSelector';
 window.store=useStore;window.groups=getModelGroupsForVisibilityScope;
 window.visibleIds=scope=>{const s=useStore.getState(),enabled=enabledModelsForContext(s,scope);return getModelGroupsForVisibilityScope(scope,s.families,s.models).flatMap(g=>g.models).filter(m=>enabled.has(m.model_type)&&(!m.nsfw_only||s.servicesConfig?.nsfw_mode)).map(m=>m.model_type).sort()};window.root=createRoot(document.getElementById('root'));
 window.mount=()=>window.root.render(<><ModelSelector placement="below"/><GenerateButton/><ModelVisibilitySection/></>);`,resolveDir:path.join(root,'ui'),loader:'tsx'},
 plugins:[{name:'expose-test-section',setup(build){build.onLoad({filter:/SystemSettingsPanel\.tsx$/},async args=>({contents:fs.readFileSync(args.path,'utf8').replace('function ModelVisibilitySection()', 'export function ModelVisibilitySection()'),loader:'tsx',resolveDir:path.dirname(args.path)}));}}],
 bundle:true,write:false,jsx:'automatic',define:{'process.env.NODE_ENV':'"development"'},logLevel:'silent'});
 const cssPath=path.join(root,'ui/dist/assets',fs.readdirSync(path.join(root,'ui/dist/assets')).find(file=>file.endsWith('.css')));
 const builtCss=fs.readFileSync(cssPath,'utf8');
 const browser=await chromium.launch({headless:true,...(process.platform==='win32'?{executablePath:process.env.MAESTRO_CHROME||'C:/Program Files/Google/Chrome/Application/chrome.exe'}:{})});
 let visibility={configured:true,enabled_models:models.filter(m=>!m.model_type.endsWith('imported')).map(m=>m.model_type),defaults_version:19,initialized_mature_models:[]};
 let prefs={configured:true,generation_mode:'video',studio_video_workflow:'frames',selected_model_per_mode:{video:'minimax_h3'},music_defaults_version:1};
 const errors=[],puts=[],submissions=[],destructiveRequests=[];
 const open=async(width=390)=>{
  const context=await browser.newContext({viewport:{width,height:900}}),page=await context.newPage();page.on('pageerror',e=>errors.push(e.message));
  await page.route('**/*',async route=>{
   const request=route.request(),key=new URL(request.url()).pathname;
   if(request.method()==='DELETE'||/\/download(?:\/|$)/.test(key)&&request.method()==='POST')destructiveRequests.push(key);
   if(key==='/')return route.fulfill({contentType:'text/html',body:`<style>${builtCss}</style><div id="root"></div>`});
   let body={loras:[],items:[],overrides:{},downloads:{},styles:[]};
   if(key==='/api/v1/models')body={models,families};
   if(key==='/api/v1/model-visibility'){if(request.method()==='PUT'){visibility={configured:true,...request.postDataJSON()};puts.push(visibility);}body=visibility;}
   if(key==='/api/v1/studio-preferences'){if(request.method()==='PUT')prefs={configured:true,...prefs,...request.postDataJSON()};body=prefs;}
   if(key.includes('/model-options/')){const id=decodeURIComponent(key.split('/').pop());body={...models.find(m=>m.model_type===id),frames_minimum:124,frames_maximum:345,frames_steps:17};}
   if(key.includes('/defaults/'))body={num_inference_steps:4,guidance_scale:1};
   if(key==='/api/v1/generate')submissions.push(request.postDataJSON());
   return route.fulfill({json:body});
  });
  await page.goto('http://visibility-scopes.test');await page.addScriptTag({content:bundle.outputFiles[0].text});
  await page.evaluate(async()=>{await window.store.getState().loadModels();window.mount();});
  return {page,context};
 };
 const assertDownloadRowHeights=async page=>{
  const image=page.locator('[data-model-visibility-media="image"]');
  await image.getByRole('button',{name:'Image',exact:true}).click();
  const group=page.locator('[data-model-visibility-scope="image:generate"]');
  const workflow=group.getByRole('button',{name:'Image · Generate',exact:true});
  await workflow.click();
  const undownloaded=page.getByRole('button',{name:'Download model files for Flux Schnell',exact:true});
  const downloaded=page.getByRole('checkbox',{name:'Flux Dev in Image · Generate',exact:true});
  const rowHeight=control=>control.evaluate(el=>el.closest('div.group').getBoundingClientRect().height);
  const [downloadHeight,downloadedHeight]=await Promise.all([rowHeight(undownloaded),rowHeight(downloaded)]);
  assert.equal(downloadHeight,28,`Download row is 28px at viewport width ${await page.evaluate(()=>window.innerWidth)}`);
  assert.equal(downloadedHeight,28,`Downloaded row is 28px at viewport width ${await page.evaluate(()=>window.innerWidth)}`);
  assert.equal(downloadHeight,downloadedHeight,'Download and Downloaded rows have equal height');
 };
 try{
  let {page,context}=await open();
  const choice=(scope,label)=>page.locator(`[data-model-visibility-scope="${scope}"]`).getByRole('checkbox',{name:label,exact:true});
  for(const media of ['video','image','audio']){
   const wrapper=page.locator(`[data-model-visibility-media="${media}"]`);
   assert.equal(await wrapper.getByRole('button').getAttribute('aria-expanded'),'false',`${media} media starts collapsed`);
  }
  const video=page.locator('[data-model-visibility-media="video"]');
  await video.getByRole('button',{name:'Video',exact:true}).click();
  assert.equal(await video.getByRole('button',{name:'Video',exact:true}).getAttribute('aria-expanded'),'true','Video media opens its workflows');
  const frames=page.locator('[data-model-visibility-scope="video:frames"]');
  const framesButton=frames.getByRole('button',{name:'Video · Frames',exact:true});
  assert.equal(await framesButton.getAttribute('aria-expanded'),'false','Workflows start collapsed inside expanded media');
  assert.equal(await frames.getByRole('checkbox').count(),0,'Workflow checkboxes stay hidden while the workflow is collapsed');
  assert.equal(await framesButton.getByText('Frames',{exact:true}).count(),1,'The workflow button shows its short label while keeping its full accessible name');
  await framesButton.click();
  await choice('video:frames','H3 Pruned 20B in Video · Frames').uncheck();
  await page.waitForFunction(()=>window.store.getState().params.model_type==='minimax_h3_full');
  assert.equal(await page.evaluate(()=>window.store.getState().enabledModels.has('minimax_h3')),true,'Scoped hiding preserves legacy baseline');
  await page.getByRole('button',{name:'Video · Extend',exact:true}).click();
  assert.equal(await choice('video:extend','H3 Pruned 20B — Frames in Video · Extend').isChecked(),true,'Same model remains enabled in Extend');
  await page.evaluate(()=>window.store.getState().setStudioVideoWorkflow('references'));
  await video.getByRole('button',{name:'Video',exact:true}).click();
  assert.equal(await video.getByRole('button',{name:'Video',exact:true}).getAttribute('aria-expanded'),'false','A collapsed media parent hides its workflows');
  await page.getByRole('button',{name:'Choose model'}).click();
  await page.getByRole('button',{name:/Enable more models/}).click();
  await page.waitForFunction(()=>{
   const media=document.querySelector('[data-model-visibility-media="video"] > button');
   const workflow=document.querySelector('[data-model-visibility-scope="video:references"] > button');
   return media?.getAttribute('aria-expanded')==='true'&&workflow?.getAttribute('aria-expanded')==='true';
  });
  const refGroup=page.locator('[data-model-visibility-scope="video:references"]');
  await refGroup.getByRole('checkbox',{name:'H3 Custom INT8 in Video · References',exact:true}).check();
  await page.waitForFunction(()=>window.store.getState().modelVisibilityOverrides['video:references']?.minimax_h3_ref2va_imported===true);
  assert.equal(await page.evaluate(()=>window.store.getState().enabledModels.has('minimax_h3_ref2va_imported')),false,'Scoped enabling does not change other workflows');
  assert.equal(await refGroup.getByText('H3 Mature',{exact:true}).count(),0,'Mature gate remains independent');
  // Bulk controls affect only the family in its current workflow.
  await page.evaluate(()=>{const s=window.store.getState();s.setModelsEnabled(s.models.filter(m=>m.omni_reference).map(m=>m.model_type),false,'video:references');});
  await page.getByRole('button',{name:'Need model',exact:true}).waitFor();
  await page.evaluate(()=>window.store.getState().startGeneration());
  assert.equal(submissions.length,0,'Hidden current model cannot bypass the button via shortcuts');
  assert.match(await page.evaluate(()=>window.store.getState().promptEnhanceError),/Enable or select/);
  await page.evaluate(()=>window.store.getState().setModelsEnabled(['minimax_h3_ref2va_imported'],true,'video:references'));
  await page.waitForFunction(()=>window.store.getState().params.model_type==='minimax_h3_ref2va_imported');
  await page.waitForFunction(()=>!window.store.getState().modelOptionsLoading);await page.waitForTimeout(200);
  await page.evaluate(()=>{const s=window.store.getState();window.store.setState({servicesConfig:{...s.servicesConfig,nsfw_mode:false},
    params:{...s.params,model_type:'minimax_h3_ref2va_mature',minimax_h3_references:[{id:'r',type:'image',path:'fixture.png'}]},
    modelVisibilityOverrides:{...s.modelVisibilityOverrides,'video:references':{...s.modelVisibilityOverrides['video:references'],minimax_h3_ref2va_mature:true}}});});
  await page.getByRole('button',{name:'Need model',exact:true}).waitFor();
  await page.evaluate(()=>{const s=window.store.getState();s.setModelsEnabled(['minimax_h3_ref2va_mature'],false,'video:references');});
  await page.waitForFunction(()=>window.store.getState().params.model_type==='minimax_h3_ref2va_imported');
  await page.waitForFunction(()=>!window.store.getState().modelOptionsLoading);await page.waitForTimeout(100);
  const finalOverrides=JSON.parse(JSON.stringify(visibility.mode_overrides));assert.ok(puts.length>0);
  await assertDownloadRowHeights(page);
  await context.close();
  // New origin/context has no localStorage; saved server choices remain authoritative.
  ({page,context}=await open(1366));
  await assertDownloadRowHeights(page);
  assert.deepEqual(await page.evaluate(()=>window.store.getState().modelVisibilityOverrides),finalOverrides);
  assert.equal(await page.evaluate(()=>window.store.getState().params.model_type),'minimax_h3_ref2va_imported','Boot restores a scoped addition absent from global baseline');
  await page.evaluate(()=>window.store.getState().setStudioVideoWorkflow('frames'));
  await page.waitForFunction(()=>window.store.getState().params.model_type==='minimax_h3_full');
  await page.evaluate(()=>window.store.getState().setStudioVideoWorkflow('extend'));
  assert.equal(await page.evaluate(()=>window.store.getState().modelVisibilityOverrides['video:extend']),undefined);
  await page.evaluate(()=>{const s=window.store.getState();s.setGenerationMode('audio');s.setAudioSubMode('music');s.setModelsEnabled(['yue2'],false,'audio:music');});
  await page.waitForFunction(()=>window.store.getState().params.model_type==='minimax_music3');
  await page.evaluate(()=>window.store.getState().setAllModelsEnabled(false));
  assert.deepEqual(await page.evaluate(()=>window.store.getState().modelVisibilityOverrides),{});
  await page.evaluate(()=>window.store.getState().resetEnabledModels());
  assert.deepEqual(await page.evaluate(()=>window.visibleIds('video:frames')),['ltx2_25','minimax_h3','minimax_h3_fused_turbo'],'Global reset restores the new Frames defaults');
  await context.close();

  // A fresh install receives scoped defaults; existing preferences above were preserved.
  visibility={configured:false,enabled_models:[],defaults_version:0,initialized_mature_models:[]};
  prefs={configured:false};
  ({page,context}=await open());
  const expected={
   'video:frames':['ltx2_25','minimax_h3','minimax_h3_fused_turbo'],
   'video:references':['minimax_h3_ref2va','minimax_h3_ref2va_fused_turbo'],
   'video:avatar':['longcat_avatar','longcat_avatar_multi'],
   'video:extend':['ltx2_25','minimax_h3','minimax_h3_fused_turbo','minimax_h3_ref2va','minimax_h3_ref2va_fused_turbo'],
   'video:blend':['ltx2_25','minimax_h3','minimax_h3_fused_turbo','minimax_h3_ref2va','minimax_h3_ref2va_fused_turbo'],
  };
  for(const [scope,ids] of Object.entries(expected))assert.deepEqual(await page.evaluate(scope=>window.visibleIds(scope),scope),ids,`Fresh ${scope} defaults`);
  assert.equal(await page.evaluate(()=>window.store.getState().params.model_type),'ltx2_25','Fresh video starts with a visible LTX-2.5 model');
  await page.evaluate(()=>{const s=window.store.getState();s.setModelsEnabled(['minimax_h3_ref2va_imported'],true,'video:references');s.setModelsEnabled(['longcat_avatar_multi'],false,'video:avatar');s.setModelsEnabled(['minimax_h3_full'],true,'video:frames');s.selectStudioVideoModel('minimax_h3_full');});
  await page.waitForFunction(()=>window.store.getState().params.model_type==='minimax_h3_full');
  const beforeReset=await page.evaluate(()=>({baseline:[...window.store.getState().enabledModels],overrides:window.store.getState().modelVisibilityOverrides}));
  await page.getByRole('button',{name:'Choose model'}).click();
  await page.getByRole('button',{name:'Reset model visibility for current workflow'}).click();
  await page.getByRole('alertdialog').getByRole('button',{name:'Confirm reset'}).click();
  await page.waitForFunction(()=>window.store.getState().params.model_type!=='minimax_h3_full');
  assert.deepEqual(await page.evaluate(()=>window.visibleIds('video:frames')),expected['video:frames']);
  assert.deepEqual(await page.evaluate(()=>[...window.store.getState().enabledModels]),beforeReset.baseline,'Scoped reset preserves the global baseline');
  const afterReset=await page.evaluate(()=>window.store.getState().modelVisibilityOverrides);
  for(const scope of Object.keys(beforeReset.overrides).filter(scope=>scope!=='video:frames'))assert.deepEqual(afterReset[scope],beforeReset.overrides[scope],`Reset preserves ${scope}`);
  assert.equal(await page.evaluate(()=>window.store.getState().models.find(m=>m.model_type==='minimax_h3_full').is_downloaded),true,'Hidden model stays downloaded');
  await page.waitForFunction(()=>window.store.getState().modelOptionsLoading===false);await page.waitForTimeout(100);
  const savedOverrides=JSON.parse(JSON.stringify(visibility.mode_overrides));
  await context.close();({page,context}=await open(1366));
  assert.deepEqual(await page.evaluate(()=>window.store.getState().modelVisibilityOverrides),savedOverrides,'Scoped reset persists across a new browser context');
  await page.evaluate(()=>{const s=window.store.getState();s.setAllModelsEnabled(false);s.resetEnabledModels('video:avatar');});
  assert.deepEqual(await page.evaluate(()=>window.visibleIds('video:avatar')),expected['video:avatar'],'Scoped reset works after None');
  assert.deepEqual(await page.evaluate(()=>window.visibleIds('video:frames')),[],'Reset after None does not enable other workflows');
  assert.deepEqual(destructiveRequests,[],'Reset never deletes or downloads model files');
  await context.close();assert.deepEqual(errors,[]);
  console.log('Workflow visibility: real settings, isolated overrides, management focus, mature gate, selection reconciliation, submission guard, server persistence, fresh curated defaults, scoped reset, selection reconciliation and no model deletion passed');
 }finally{await browser.close();}
})().catch(error=>{console.error(error);process.exitCode=1});
