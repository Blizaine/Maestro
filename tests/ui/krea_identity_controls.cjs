// Krea identity controls, durable per-model settings, and request serialization.
// All backend calls are mocked; generation requests are recorded and never executed.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const root = path.resolve(__dirname, '../..');
const esbuild = require(path.join(root, 'ui/node_modules/esbuild'));
const {chromium} = require(process.env.MAESTRO_PLAYWRIGHT || 'playwright');

const raw = 'krea2_raw_edit';
const turbo = 'krea2_turbo_edit';
const other = 'flux2_klein_9b';
const keys = ['krea2_ref_boost', 'krea2_ref_boost_a', 'krea2_grounding_px'];

(async () => {
  const bundle = await esbuild.build({stdin:{contents:`
    import React from 'react'; import {createRoot} from 'react-dom/client';
    import {useStore} from './src/stores/useStore';
    import {AdvancedSettings} from './src/components/Sidebar/AdvancedSettings';
    import * as controls from './src/lib/kreaIdentityControls';
    window.store=useStore; window.controls=controls;
    const root=createRoot(document.getElementById('root'));
    root.render(<AdvancedSettings compact/>);
    window.boot=()=>useStore.getState().loadModels();
    window.bootPromise=window.boot();`,resolveDir:path.join(root,'ui'),loader:'tsx'},
    bundle:true,write:false,jsx:'automatic',logLevel:'silent',define:{'process.env.NODE_ENV':'"development"'}});

  const outDir = path.join(root, '.codex-tmp/krea-identity-controls-20260926');
  fs.mkdirSync(outDir,{recursive:true});
  let css='';
  try {
    const assets=path.join(root,'ui/dist/assets');
    css=fs.readFileSync(path.join(assets,fs.readdirSync(assets).find(name=>name.endsWith('.css'))),'utf8');
  } catch { /* UI behavior assertions do not depend on compiled CSS. */ }

  const models=[
    {model_type:raw,name:'Krea 2 Raw Edit',family:'krea2',architecture:raw},
    {model_type:turbo,name:'Krea 2 Turbo Edit',family:'krea2',architecture:turbo},
    {model_type:other,name:'Flux Klein',family:'flux2',architecture:other},
  ];
  const kreaDefs=keys.map((id,index)=>({id,name:`Backend ${index+1}`,label:`Backend value ${index+1}`,type:'number'}));
  const options=id=>({model_type:id,architecture:id,image_outputs:true,image_ref_inpaint:false,
    max_image_refs:4,guidance_max_phases:id===turbo?0:1,lock_guidance_scale:id===turbo,
    default_num_inference_steps:id===turbo?8:30,default_guidance_scale:id===turbo?0:4,
    custom_settings_def:kreaDefs});
  let preferences={configured:true,generation_mode:'image',studio_image_workflow:'generate',
    selected_model_per_mode:{image:raw},selected_model_per_audio_sub_mode:{},
    krea_identity_settings_per_model:{}};
  const submissions=[];
  const uploads=[];
  const browser=await chromium.launch({headless:true,...(process.platform==='win32'?{
    executablePath:process.env.MAESTRO_CHROME||'C:/Program Files/Google/Chrome/Application/chrome.exe'}:{})});

  try {
    const context=await browser.newContext({viewport:{width:1100,height:900}});
    const page=await context.newPage();
    const pageErrors=[];
    page.on('pageerror',error=>pageErrors.push(error.message));
    await page.route('**/*',async route=>{
      const request=route.request(),url=new URL(request.url()),key=url.pathname;
      const json=(body,status=200)=>route.fulfill({status,json:body});
      if(key==='/') return route.fulfill({contentType:'text/html',body:`<!doctype html><meta name="viewport" content="width=device-width, initial-scale=1"><style>${css}</style><div id="root"></div><script src="/bundle.js"></script>`});
      if(key==='/bundle.js') return route.fulfill({contentType:'application/javascript',body:bundle.outputFiles[0].text});
      if(key==='/api/v1/models') return json({models,families:[{id:'krea2',label:'Krea',order:80},{id:'flux2',label:'Flux',order:70}]});
      if(key==='/api/v1/model-visibility') return json({configured:true,enabled_models:models.map(model=>model.model_type),initialized_mature_models:[]});
      if(key==='/api/v1/h3-window-overrides') return json({overrides:{}});
      if(key==='/api/v1/studio-preferences') {
        if(request.method()==='PUT') preferences={...preferences,...request.postDataJSON()};
        return json(preferences);
      }
      if(key==='/api/v1/loras/installed') return json({loras:[]});
      if(key.startsWith('/api/v1/loras/')) return json({loras:[],guidance_max_phases:1});
      if(key.startsWith('/api/v1/model-options/')) return json(options(key.split('/').pop()));
      if(key.startsWith('/api/v1/defaults/')) return json({num_inference_steps:30,guidance_scale:4,custom_settings:{}});
      if(key==='/api/v1/presets') return json({presets:[]});
      if(key==='/api/v1/jobs') return json({jobs:[]});
      if(key==='/api/v1/upload') {
        uploads.push(request.postDataBuffer()?.length || 0);
        return json({path:'/uploads/reference.png',filename:'reference.png',url:'/api/v1/file/reference.png'});
      }
      if(key==='/api/v1/generate') {
        submissions.push(request.postDataJSON());
        return json({job_id:`mock-${submissions.length}`,status:'held'});
      }
      if(key.startsWith('/api/v1/status/')) return json({job_id:key.split('/').pop(),status:'completed',progress:100,step:1,total_steps:1,output_files:[]});
      if(key==='/api/v1/outputs') return json({outputs:[],total:0});
      if(key==='/api/v1/installed-loras') return json({loras:[]});
      return json({items:[],models:[],loras:[],presets:[],jobs:[],outputs:[],downloads:[],configured:true,status:'available'});
    });

    await page.goto('http://krea-controls.test');
    await page.waitForFunction(()=>window.store && window.bootPromise);
    await page.evaluate(()=>window.bootPromise);
    await page.waitForFunction(()=>!window.store.getState().modelOptionsLoading
      && window.store.getState().modelOptions?.model_type==='krea2_raw_edit');

    const helper=await page.evaluate(()=>({
      models:['krea2_raw_edit','krea2_turbo_edit','krea2_edit','flux2_klein_9b'].map(value=>window.controls.isKreaIdentityEdit(value)),
      normalized:window.controls.normalizeKreaIdentitySettings({krea2_ref_boost:-2,krea2_ref_boost_a:14,krea2_grounding_px:2000,other:true}),
      onlyKeys:Object.keys(window.controls.pickKreaIdentitySettings({krea2_ref_boost:2,other:true})).sort(),
      twoInputSupportedEditCount:window.controls.countKreaIdentityReferences({generationMode:'image',workflow:'inpaint',imageMode:2,
        sourceImagePresent:true,referenceCount:1,editReferencesSupported:true}),
      twoInputUnsupportedEditCount:window.controls.countKreaIdentityReferences({generationMode:'image',workflow:'inpaint',imageMode:2,
        sourceImagePresent:true,referenceCount:1,editReferencesSupported:false}),
      upscaleCount:window.controls.countKreaIdentityReferences({generationMode:'image',workflow:'upscale',imageMode:0,
        sourceImagePresent:true,referenceCount:1,editReferencesSupported:true}),
    }));
    assert.deepEqual(helper.models,[true,true,false,false]);
    assert.deepEqual(helper.normalized,{krea2_ref_boost:0,krea2_ref_boost_a:10,krea2_grounding_px:1536});
    assert.deepEqual(helper.onlyKeys,[...keys].sort());
    assert.equal(helper.twoInputSupportedEditCount,2,'Mode 2 counts the source plus an optional reference when the model supports it');
    assert.equal(helper.twoInputUnsupportedEditCount,1,'Mode 2 ignores stale references the current model cannot use');
    assert.equal(helper.upscaleCount,0,'Upscale does not treat Krea generation controls as active');

    await page.evaluate(()=>{
      const s=window.store.getState();
      const reference=new File(['fixture'],'ref.png',{type:'image/png'});
      window.store.setState({sidebarMode:'studio',generationMode:'image',studioImageWorkflow:'generate',
        params:{...s.params,model_type:'krea2_raw_edit',prompt:'A portrait with a blue coat.',custom_settings:{other_setting:'keep'},image_mode:1},
        models:[{model_type:'krea2_raw_edit',architecture:'krea2_raw_edit'},
          {model_type:'krea2_turbo_edit',architecture:'krea2_turbo_edit'},
          {model_type:'flux2_klein_9b',architecture:'flux2_klein_9b'}],
        modelOptions:{...s.modelOptions,...window.controls.normalizeKreaIdentitySettings({}),model_type:'krea2_raw_edit',architecture:'krea2_raw_edit',image_ref_inpaint:false,custom_settings_def:[
          {id:'krea2_ref_boost',name:'Generic boost',label:'Generic boost',type:'number'},
          {id:'krea2_ref_boost_a',name:'Generic scene',label:'Generic scene',type:'number'},
          {id:'krea2_grounding_px',name:'Generic grounding',label:'Generic grounding',type:'number'},
        ]},
        modelOptionsLoading:false,selectedModelPerMode:{...s.selectedModelPerMode,image:'krea2_raw_edit'},
        imageRefs:[reference],imageRefType:'I',imageWorkflowSourcePath:'',imageWorkflowSourceFile:null,
        savedParamsPerMode:{},jobs:[],isGenerating:false,isEnhancing:false});
    });
    const advanced=page.getByRole('button',{name:/Advanced settings/});
    await advanced.click();
    const dialog=page.getByRole('dialog',{name:'Advanced settings'});
    const generation=dialog.getByTestId('advanced-generation');
    await generation.locator('summary').click();
    const panel=dialog.getByRole('region',{name:'Krea Identity Edit'});
    await panel.waitFor();
    assert.equal(await generation.getByText('Guidance Scale',{exact:true}).count(),1,'RAW keeps real CFG adjustable');
    assert.equal(await panel.getByRole('spinbutton',{name:'Subject likeness value'}).inputValue(),'1');
    assert.equal(await panel.getByRole('spinbutton',{name:'Scene/reference likeness · Image 1 value'}).count(),0,'Scene setting is hidden with one participating reference');
    assert.equal(await panel.getByRole('spinbutton',{name:'Grounding size · longer edge value'}).inputValue(),'768');
    assert.equal(await panel.getByText(/384–768px is the trained range/).count(),1,'Grounding help explains the trained and experimental ranges');
    assert.equal(await panel.getByText('Generic boost',{exact:true}).count(),0,'Dedicated Krea settings do not duplicate generic definitions');

    await page.evaluate(()=>window.store.setState({studioImageWorkflow:'inpaint',params:{...window.store.getState().params,image_mode:2},
      imageWorkflowSourcePath:'/uploads/source.png'}));
    await panel.getByRole('spinbutton',{name:'Subject likeness value'}).waitFor();
    assert.equal(await panel.getByRole('spinbutton',{name:'Subject likeness · Image 2 value'}).count(),0,
      'The real Krea inpaint option does not expose extra references, so stale files do not activate a second role');
    assert.equal(await panel.getByRole('spinbutton',{name:'Scene/reference likeness · Image 1 value'}).count(),0);
    await page.evaluate(()=>window.store.setState({studioImageWorkflow:'generate',params:{...window.store.getState().params,image_mode:1},
      imageWorkflowSourcePath:''}));
    await panel.getByRole('spinbutton',{name:'Subject likeness value'}).waitFor();

    // Typing a multi-digit grounding value remains a local draft until commit.
    const grounding=panel.getByRole('spinbutton',{name:'Grounding size · longer edge value'});
    await grounding.focus(); await page.keyboard.press('Control+A'); await page.keyboard.type('1024');
    assert.equal(await grounding.inputValue(),'1024','Multi-digit number entry is not clamped on each key');
    assert.equal(await page.evaluate(()=>window.store.getState().params.custom_settings.krea2_grounding_px),undefined,'A numeric draft commits only on blur or Enter');
    await grounding.press('Enter');
    await page.waitForFunction(()=>window.store.getState().params.custom_settings.krea2_grounding_px===1024);
    assert.equal(await page.evaluate(()=>window.store.getState().params.custom_settings.other_setting),'keep','Editing a Krea value retains unrelated custom settings');

    await panel.getByRole('button',{name:'Strong likeness (4)'}).click();
    await page.waitForFunction(()=>window.store.getState().params.custom_settings.krea2_ref_boost===4);
    await page.evaluate(()=>window.store.setState({imageRefs:[
      new File(['scene'],'scene.png',{type:'image/png'}),new File(['subject'],'subject.png',{type:'image/png'})]}));
    await panel.getByRole('spinbutton',{name:'Subject likeness · Image 2 value'}).waitFor();
    assert.equal(await panel.getByRole('spinbutton',{name:'Scene/reference likeness · Image 1 value'}).inputValue(),'1');
    await panel.getByRole('spinbutton',{name:'Scene/reference likeness · Image 1 value'}).fill('6.5');
    await panel.getByRole('spinbutton',{name:'Scene/reference likeness · Image 1 value'}).press('Enter');
    await page.waitForFunction(()=>window.store.getState().params.custom_settings.krea2_ref_boost_a===6.5);

    // The active badge lists only changed controls supported by active refs.
    const generationBadge=generation.locator('summary [aria-label$="active"]');
    const badgeLabels=await generationBadge.getAttribute('title');
    assert.match(badgeLabels,/Krea subject likeness/);
    assert.match(badgeLabels,/Krea scene\/reference likeness/);
    assert.match(badgeLabels,/Krea grounding 1024px/);
    await page.screenshot({path:path.join(outDir,'krea-controls-desktop.png'),fullPage:true});
    await page.setViewportSize({width:390,height:844});
    await page.screenshot({path:path.join(outDir,'krea-controls-mobile.png'),fullPage:true});
    await page.setViewportSize({width:1100,height:900});

    // Queue-submit through the real store with a mocked endpoint, confirming the exact per-model payload.
    await page.evaluate(async()=>{await window.store.getState().startGeneration('queue');window.store.setState({jobs:[],isGenerating:false});});
    await page.waitForTimeout(100);
    assert.equal(submissions.length,1,'Queued request reaches only the mocked endpoint');
    assert.deepEqual(Object.fromEntries(keys.map(key=>[key,submissions[0].custom_settings[key]])),
      {krea2_ref_boost:4,krea2_ref_boost_a:6.5,krea2_grounding_px:1024});
    assert.equal(submissions[0]._queue_mode,'held');
    assert.equal(uploads.length,2,'Reference uploads were intercepted by the fixture');

    // Switch to Turbo and verify its independent defaults, then save separate values.
    await page.evaluate(()=>window.store.getState().selectModel('krea2_turbo_edit'));
    await page.waitForFunction(()=>!window.store.getState().modelOptionsLoading
      && window.store.getState().modelOptions?.model_type==='krea2_turbo_edit'
      && window.store.getState().params.model_type==='krea2_turbo_edit');
    const turboGrounding=panel.getByRole('spinbutton',{name:'Grounding size · longer edge value'});
    assert.equal(await turboGrounding.inputValue(),'768','Turbo starts at its own defaults');
    assert.equal(await generation.getByText('Guidance Scale',{exact:true}).count(),0,'Turbo hides CFG because its runtime always disables it');
    await panel.getByRole('spinbutton',{name:'Subject likeness · Image 2 value'}).fill('7.5');
    await panel.getByRole('spinbutton',{name:'Subject likeness · Image 2 value'}).press('Enter');
    await panel.getByRole('spinbutton',{name:'Scene/reference likeness · Image 1 value'}).fill('8.25');
    await panel.getByRole('spinbutton',{name:'Scene/reference likeness · Image 1 value'}).press('Enter');
    await turboGrounding.fill('640'); await turboGrounding.press('Enter');
    await page.waitForFunction(()=>window.store.getState().params.custom_settings.krea2_grounding_px===640);
    await page.evaluate(async()=>{await window.store.getState().startGeneration('queue');window.store.setState({jobs:[],isGenerating:false});});
    await page.waitForTimeout(100);
    assert.equal(submissions.length,2);
    assert.equal(submissions[1].model_type,turbo);
    assert.deepEqual(Object.fromEntries(keys.map(key=>[key,submissions[1].custom_settings[key]])),
      {krea2_ref_boost:7.5,krea2_ref_boost_a:8.25,krea2_grounding_px:640});
    const storedMap=await page.evaluate(()=>window.store.getState().kreaIdentitySettingsPerModel);
    assert.deepEqual(storedMap[raw],{krea2_ref_boost:4,krea2_ref_boost_a:6.5,krea2_grounding_px:1024});
    assert.deepEqual(storedMap[turbo],{krea2_ref_boost:7.5,krea2_ref_boost_a:8.25,krea2_grounding_px:640});

    await page.evaluate(()=>window.store.getState().selectModel('krea2_raw_edit'));
    await page.waitForFunction(()=>!window.store.getState().modelOptionsLoading
      && window.store.getState().modelOptions?.model_type==='krea2_raw_edit'
      && window.store.getState().params.custom_settings.krea2_grounding_px===1024);
    assert.deepEqual(await page.evaluate(keys=>Object.fromEntries(keys.map(key=>[key,window.store.getState().params.custom_settings[key]])),keys),
      {krea2_ref_boost:4,krea2_ref_boost_a:6.5,krea2_grounding_px:1024});
    const waitForHost=async(predicate,label)=>{
      const deadline=Date.now()+5000;
      while(!predicate(preferences)) {
        if(Date.now()>deadline) throw new Error(`Timed out waiting for ${label}: ${JSON.stringify(preferences.krea_identity_settings_per_model)}`);
        await new Promise(resolve=>setTimeout(resolve,25));
      }
    };
    await waitForHost(value=>value.krea_identity_settings_per_model?.[turbo]?.krea2_grounding_px===640,'Turbo durable preference');

    // Reload the browser runtime and hydrate both model records from durable server preferences.
    await page.reload();
    await page.waitForFunction(()=>window.store && window.bootPromise);
    await page.evaluate(()=>window.bootPromise);
    await page.waitForFunction(modelId=>!window.store.getState().modelOptionsLoading
      && window.store.getState().modelOptions?.model_type==='krea2_raw_edit'
      && window.store.getState().kreaIdentitySettingsPerModel?.[modelId]?.krea2_grounding_px===640,turbo);
    assert.deepEqual(await page.evaluate(()=>window.store.getState().params.custom_settings),
      {krea2_ref_boost:4,krea2_ref_boost_a:6.5,krea2_grounding_px:1024},'Reload restores the selected model settings');
    assert.deepEqual(await page.evaluate(modelId=>window.store.getState().kreaIdentitySettingsPerModel[modelId],turbo),
      {krea2_ref_boost:7.5,krea2_ref_boost_a:8.25,krea2_grounding_px:640},'Reload retains Turbo settings separately');
    await page.getByRole('button',{name:/Advanced settings/}).click();
    const reloadedDialog=page.getByRole('dialog',{name:'Advanced settings'});
    const reloadedGeneration=reloadedDialog.getByTestId('advanced-generation');
    await reloadedGeneration.locator('summary').click();

    // A sidecar restore uses that output's settings rather than borrowing later preferences.
    await page.evaluate(()=>window.store.setState({selectedOutputMeta:{params:{model_type:'krea2_raw_edit',prompt:'Restored image',
      custom_settings:{krea2_ref_boost:0,krea2_ref_boost_a:9.5,krea2_grounding_px:1536,ignored:'drop'},image_mode:1}}}));
    await page.evaluate(()=>window.store.getState().loadSettingsFromOutput());
    await page.waitForFunction(()=>window.store.getState().params.prompt==='Restored image'
      && window.store.getState().params.custom_settings.krea2_grounding_px===1536);
    assert.deepEqual(await page.evaluate(()=>window.store.getState().params.custom_settings),
      {krea2_ref_boost:0,krea2_ref_boost_a:9.5,krea2_grounding_px:1536},'Sidecar restore uses exact output values and supported keys');

    // Older sidecars without settings use neutral defaults, not this model's remembered values.
    await page.evaluate(()=>window.store.setState({selectedOutputMeta:{params:{model_type:'krea2_raw_edit',prompt:'Legacy image',image_mode:1}}}));
    await page.evaluate(()=>window.store.getState().loadSettingsFromOutput());
    await page.waitForFunction(()=>window.store.getState().params.prompt==='Legacy image'
      && window.store.getState().params.custom_settings.krea2_grounding_px===768);
    assert.deepEqual(await page.evaluate(()=>window.store.getState().params.custom_settings),
      {krea2_ref_boost:1,krea2_ref_boost_a:1,krea2_grounding_px:768});

    // The range is immediate, numeric input can be cancelled, and reset preserves other custom keys.
    await page.evaluate(()=>{
      const s=window.store.getState();
      window.store.setState({generationMode:'image',sidebarMode:'studio',studioImageWorkflow:'generate',
        modelOptions:{...s.modelOptions,model_type:'krea2_raw_edit',architecture:'krea2_raw_edit',image_ref_inpaint:true},
        params:{...s.params,model_type:'krea2_raw_edit',custom_settings:{krea2_ref_boost:1,krea2_ref_boost_a:1,krea2_grounding_px:768,other_setting:'keep'}},
        selectedModelPerMode:{...s.selectedModelPerMode,image:'krea2_raw_edit'},imageRefs:[new File(['subject'],'subject.png',{type:'image/png'})]});
    });
    const advancedButton=page.getByRole('button',{name:'Advanced settings',exact:true});
    if(await advancedButton.getAttribute('aria-expanded')!=='true') await advancedButton.click();
    const finalDialog=page.getByRole('dialog',{name:'Advanced settings'});
    const finalGeneration=finalDialog.getByTestId('advanced-generation');
    if(!await finalGeneration.locator('summary').evaluate(node=>node.parentElement.open)) await finalGeneration.locator('summary').click();
    const finalPanel=finalDialog.getByRole('region',{name:'Krea Identity Edit'});
    await finalPanel.waitFor();
    await finalPanel.getByRole('spinbutton',{name:'Grounding size · longer edge value'}).waitFor();
    const finalGrounding=finalPanel.getByRole('spinbutton',{name:'Grounding size · longer edge value'});
    await finalGrounding.fill('1200'); await finalGrounding.press('Escape');
    assert.equal(await finalGrounding.inputValue(),'768','Escape discards an uncommitted draft');
    const slider=panel.getByRole('slider',{name:'Grounding size · longer edge slider'});
    await slider.focus(); await slider.press('ArrowRight');
    await page.waitForFunction(()=>window.store.getState().params.custom_settings.krea2_grounding_px===832);
    await finalPanel.getByRole('button',{name:'Reset defaults'}).click();
    await page.waitForFunction(()=>window.store.getState().params.custom_settings.krea2_grounding_px===768);
    assert.equal(await page.evaluate(()=>window.store.getState().params.custom_settings.other_setting),'keep');

    // A non-Krea model with identical custom keys never gets Krea controls or badges.
    await page.evaluate(()=>{
      const s=window.store.getState();
      window.store.setState({params:{...s.params,model_type:'flux2_klein_9b',custom_settings:{krea2_ref_boost:5,krea2_grounding_px:1408}},
        selectedModelPerMode:{...s.selectedModelPerMode,image:'flux2_klein_9b'},
        modelOptions:{...s.modelOptions,model_type:'flux2_klein_9b',architecture:'flux2_klein_9b'}});
    });
    assert.equal(await reloadedDialog.getByRole('region',{name:'Krea Identity Edit'}).count(),0);
    assert.equal(await reloadedGeneration.locator('summary [aria-label$="active"]').count(),0,
      'A non-Krea model does not show a Krea-only active badge');
    assert.equal(pageErrors.length,0,pageErrors.join('\n'));
    await context.close();
    console.log('Krea Identity UI: exact model gating, active-reference roles, draft/slider/reset behavior, per-model request values, server-preference reload, sidecar restore, and non-Krea badge isolation passed.');
    console.log(`Screenshots: ${path.join(outDir,'krea-controls-desktop.png')} and ${path.join(outDir,'krea-controls-mobile.png')}`);
  } finally { await browser.close(); }
})().catch(error=>{console.error(error);process.exitCode=1;});
