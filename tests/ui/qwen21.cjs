// Real model selection/store and references. All API traffic is isolated.
const assert = require('node:assert/strict');
const path = require('node:path');
const fs = require('node:fs');
const root = path.resolve(__dirname, '../..');
const esbuild = require(path.join(root, 'ui/node_modules/esbuild'));
const {chromium} = require(process.env.MAESTRO_PLAYWRIGHT || 'playwright');
const id = 'qwen_image_21_7B', previous = 'flux2_klein_9b';
const definition = JSON.parse(fs.readFileSync(path.join(root, 'app/defaults', `${id}.json`), 'utf8'));

(async () => {
  const bundle = await esbuild.build({stdin:{contents:`
    import React from 'react'; import {createRoot} from 'react-dom/client';
    import {useStore, modelSupportsImageWorkflow} from './src/stores/useStore';
    import {ModelSelector} from './src/components/Sidebar/ModelSelector';
    import {ImageRefSection} from './src/components/Sidebar/ImageRefSection';
    window.store=useStore; window.supports=modelSupportsImageWorkflow;
    createRoot(document.getElementById('root')).render(<><ModelSelector/><ImageRefSection/></>);
    window.boot=useStore.getState().loadModels();`,resolveDir:path.join(root,'ui'),loader:'tsx'},
    bundle:true,write:false,jsx:'automatic',logLevel:'silent',define:{'process.env.NODE_ENV':'"development"'}});
  const browser = await chromium.launch({headless:true,...(process.platform==='win32' ? {
    executablePath:process.env.MAESTRO_CHROME || 'C:/Program Files/Google/Chrome/Application/chrome.exe'} : {})});
  try {
    const errors=[];
    let prefs={configured:true,generation_mode:'image',studio_image_workflow:'generate',selected_model_per_mode:{image:previous}};
    let visibility={configured:true,defaults_version:15,enabled_models:[previous],initialized_mature_models:[]};
    const models=[{model_type:previous,name:'Flux 2 Klein 9B',family:'flux2',architecture:previous,supports_image_edit:true},
      {...definition.model,model_type:id,family:'qwen',is_downloaded:false,supports_image_edit:true,supports_ref_images:true}];
    async function open(width) {
      const context=await browser.newContext({viewport:{width,height:850}});
      const page=await context.newPage();
      page.on('pageerror',error=>errors.push(error.message));
      await page.route('**/*',async route=>{
        const key=new URL(route.request().url()).pathname;
        let result={items:[],loras:[],characters:[],presets:[],styles:[],overrides:{}};
        if(key==='/') return route.fulfill({contentType:'text/html',body:'<div id="root" style="max-width:360px;margin-top:400px"></div>'});
        if(key==='/api/v1/models') result={models,families:[{id:'qwen',label:'Qwen',order:110},{id:'flux2',label:'Flux',order:100}]};
        if(key==='/api/v1/model-visibility') {
          if(route.request().method()==='PUT') visibility={configured:true,...route.request().postDataJSON()};
          result=visibility;
        }
        if(key==='/api/v1/studio-preferences') {
          if(route.request().method()==='PUT') prefs={...prefs,configured:true,...route.request().postDataJSON()};
          result=prefs;
        }
        if(key.includes('/defaults/')) result={...definition,video_prompt_type:'I'};
        if(key.includes('/model-options/')) result={model_type:key.split('/').pop(),image_outputs:true,max_image_refs:10,
          image_ref_choices:{choices:[['None',''],['Reference images','I']],default:'I'},no_background_removal:true,
          guidance_max_phases:1,vae_block_size:32};
        await route.fulfill({json:result});
      });
      await page.goto('http://qwen21.test');
      await page.addScriptTag({content:bundle.outputFiles[0].text});
      await page.evaluate(()=>window.boot);
      await page.waitForFunction(()=>!window.store.getState().modelOptionsLoading);
      return {page,context};
    }
    let {page,context}=await open(1100);
    assert.equal(await page.evaluate(()=>window.store.getState().params.model_type),previous,'Do not change the selected model');
    assert.ok(await page.evaluate(id=>window.store.getState().enabledModels.has(id),id),'Update exposes the new model once');
    await page.getByRole('button',{name:'Choose model',exact:true}).click();
    await page.getByRole('button',{name:/Qwen Image 2.1 7B/}).click();
    await page.waitForFunction(id=>window.store.getState().params.model_type===id && !window.store.getState().modelOptionsLoading,id);
    const state=await page.evaluate(()=>({steps:window.store.getState().params.num_inference_steps,cfg:window.store.getState().params.guidance_scale}));
    assert.deepEqual(state,{steps:40,cfg:1});
    await page.getByText('Up to 10 reference images.',{exact:true}).waitFor();
    assert.deepEqual(await page.evaluate(id=>{
      const model=window.store.getState().models.find(m=>m.model_type===id);
      return ['generate','inpaint','outpaint'].map(w=>window.supports(model,w,true));
    },id),[true,false,false]);
    await page.evaluate(id=>window.store.getState().toggleModelEnabled(id),id);
    await page.waitForTimeout(200);
    await context.close();
    ({page,context}=await open(390));
    assert.equal(await page.evaluate(id=>window.store.getState().enabledModels.has(id),id),false,'The user can hide it permanently');
    assert.equal(errors.length,0,errors.join('\n'));
    await context.close();
    console.log('Qwen 2.1 visibility migration, selection, defaults, reference capability and persisted opt-out passed.');
  } finally {await browser.close();}
})().catch(error=>{console.error(error);process.exitCode=1;});
