// Native FLUX 3 selection, adaptive image references and visibility persistence.
const assert = require('node:assert/strict');
const path = require('node:path');
const fs = require('node:fs');
const root = path.resolve(__dirname, '../..');
const esbuild = require(path.join(root, 'ui/node_modules/esbuild'));
const {chromium} = require(process.env.MAESTRO_PLAYWRIGHT || 'playwright');
const id = 'flux3_klein', previous = 'flux2_klein_9b';
const qwenId = 'qwen_image_21_7B';
const definition = JSON.parse(fs.readFileSync(path.join(root, 'app/defaults', `${id}.json`), 'utf8'));

(async () => {
  const bundle = await esbuild.build({absWorkingDir:path.join(root,'ui'),stdin:{contents:`
    import React from 'react'; import {createRoot} from 'react-dom/client';
    import {useStore, modelSupportsImageWorkflow} from './src/stores/useStore';
    import {ModelSelector} from './src/components/Sidebar/ModelSelector';
    import {ImageRefSection} from './src/components/Sidebar/ImageRefSection';
    import {AdvancedSettings} from './src/components/Sidebar/AdvancedSettings';
    window.store=useStore; window.supports=modelSupportsImageWorkflow;
    createRoot(document.getElementById('root')).render(<><ModelSelector/><ImageRefSection/><AdvancedSettings compact/></>);
    window.boot=useStore.getState().loadModels();`,resolveDir:path.join(root,'ui'),loader:'tsx'},
    bundle:true,write:false,jsx:'automatic',logLevel:'silent',define:{'process.env.NODE_ENV':'"development"'}});
  const browser = await chromium.launch({headless:true,...(process.platform==='win32' ? {
    executablePath:process.env.MAESTRO_CHROME || 'C:/Program Files/Google/Chrome/Application/chrome.exe'} : {})});
  try {
    const errors=[];
    const uploadedPaths=[];
    const submittedGenerations=[];
    let prefs={configured:true,generation_mode:'image',studio_image_workflow:'generate',selected_model_per_mode:{image:previous}};
    let visibility={configured:true,defaults_version:21,enabled_models:[previous],initialized_mature_models:[]};
    const models=[{model_type:previous,name:'Flux 2 Klein 9B',family:'flux2',architecture:previous,supports_image_edit:true},
      {model_type:qwenId,name:'Qwen Image 2.1 7B',family:'qwen',architecture:qwenId,
        supports_image_edit:true,supports_image_inpaint:true,supports_image_outpaint:true},
      {...definition.model,model_type:id,family:'flux3',is_downloaded:false,supports_image_edit:true,
        supports_ref_images:true,supports_image_inpaint:false,supports_image_outpaint:false}];
    async function open(width) {
      const context=await browser.newContext({viewport:{width,height:850}});
      const page=await context.newPage();
      page.on('pageerror',error=>errors.push(error.message));
      await page.route('**/*',async route=>{
        const key=new URL(route.request().url()).pathname;
        const method=route.request().method();
        let result={items:[],loras:[],characters:[],presets:[],styles:[],overrides:{}};
        if(key==='/') return route.fulfill({contentType:'text/html',body:'<div id="root" style="max-width:360px;margin-top:350px"></div>'});
        if(key==='/api/v1/models') result={models,families:[
          {id:'flux3',label:'Flux 3',order:102},{id:'flux2',label:'Flux 2',order:101},{id:'qwen',label:'Qwen',order:103},
        ]};
        if(key==='/api/v1/model-visibility') {
          if(method==='PUT') visibility={configured:true,...route.request().postDataJSON()};
          result=visibility;
        }
        if(key==='/api/v1/studio-preferences') {
          if(method==='PUT') prefs={...prefs,configured:true,...route.request().postDataJSON()};
          result=prefs;
        }
        if(key.includes('/defaults/')) result=key.endsWith('/'+id) ? definition : {num_inference_steps:4,model:models[0]};
        if(key.includes('/model-options/')) {
          const modelType=key.split('/').pop();
          result={model_type:modelType,architecture:modelType,image_outputs:true,
          max_image_refs:10,image_ref_choices:modelType===id
            ? {choices:[['None',''],['Reference images / image editing','I']],default:'I'}
            : modelType===qwenId
              ? {choices:[['None',''],['Reference images','I']],default:'I'}
              : {choices:[['None',''],['Subject / landscape and people / objects','KI']],default:'KI'},
          image_ref_inpaint:modelType===qwenId,inpaint_video_prompt_type:modelType===qwenId?'VAG':undefined,
          no_background_removal:true,preserve_image_ref_native_size:true,no_negative_prompt:true,
          loras_disabled:modelType!==id,guidance_max_phases:0,lock_guidance_scale:true,vae_block_size:16};
        }
        if(key===`/api/v1/loras/${id}`) result={loras:['flux3-style.safetensors'],guidance_max_phases:0};
        if(key==='/api/v1/jobs') result={jobs:[]};
        if(key==='/api/v1/upload' && method==='POST') {
          const path=`uploads/ref-${uploadedPaths.length+1}.png`;
          uploadedPaths.push(path);
          return route.fulfill({json:{filename:path.split('/').pop(),path,url:`/api/v1/file/${path.split('/').pop()}`}});
        }
        if(key==='/api/v1/generate' && method==='POST') {
          submittedGenerations.push(route.request().postDataJSON());
          return route.fulfill({json:{job_id:`mock-job-${submittedGenerations.length}`,status:'queued'}});
        }
        if(/^\/api\/v1\/jobs\/mock-job-\d+$/.test(key)) {
          result={status:'completed',progress:100,step:1,total_steps:1,output_files:[]};
        }
        await route.fulfill({json:result});
      });
      await page.goto('http://flux3.test');
      await page.addScriptTag({content:bundle.outputFiles[0].text});
      await page.evaluate(()=>window.boot);
      await page.waitForFunction(()=>!window.store.getState().modelOptionsLoading);
      return {page,context};
    }
    let {page,context}=await open(1100);
    assert.equal(await page.evaluate(()=>window.store.getState().params.model_type),previous,'Retain selected model');
    assert.ok(await page.evaluate(id=>window.store.getState().enabledModels.has(id),id),'Expose once on update');
    await page.getByRole('button',{name:'Choose model',exact:true}).click();
    await page.getByRole('button',{name:/^Flux 3 Klein 5.6B/}).click();
    await page.waitForFunction(id=>window.store.getState().params.model_type===id && !window.store.getState().modelOptionsLoading,id);
    assert.equal(await page.evaluate(()=>window.store.getState().params.num_inference_steps),16);
    await page.getByRole('button',{name:'Advanced settings'}).click();
    const loraSection=page.getByTestId('advanced-loras');
    assert.equal(await loraSection.locator('summary').innerText(),'LoRAs & presets',
      'Enable the FLUX 3 adapter section when the backend confirms LoRA support');
    await loraSection.locator('summary').click();
    await page.getByText('flux3-style',{exact:true}).waitFor();
    const flux3Lora=page.getByRole('button',{name:'flux3-style',exact:true});
    await flux3Lora.click();
    assert.deepEqual(await page.evaluate(()=>window.store.getState().params.activated_loras),['flux3-style.safetensors'],
      'Allow a compatible FLUX 3 adapter to be selected');
    await flux3Lora.click();
    assert.deepEqual(await page.evaluate(()=>window.store.getState().params.activated_loras),[],
      'Allow the selected FLUX 3 adapter to be cleared');
    assert.deepEqual(await page.evaluate(id=>{
      const model=window.store.getState().models.find(m=>m.model_type===id);
      return ['generate','inpaint','outpaint'].map(w=>window.supports(model,w,true));
    },id),[true,false,false]);
    await page.getByText('Up to 10 reference images.',{exact:true}).waitFor();

    // Exercise the real Zustand generation path. The stale selector mirrors a
    // Flux 2/KI setting restored immediately before the ImageRefSection effect
    // has a chance to normalize its local selector.
    await page.evaluate(id=>{
      const refs=Array.from({length:10},(_,index)=>new File([`image-${index+1}`],`reference-${index+1}.png`,{type:'image/png'}));
      window.store.setState(state=>({
        generationMode:'image',studioImageWorkflow:'generate',imageRefs:refs,imageRefType:'KI',
        params:{...state.params,model_type:id,prompt:'Combine these reference images into one photo.',
          image_mode:1,image_refs:['stale/legacy-reference.png'],video_prompt_type:'KI'},
      }));
    },id);
    await page.evaluate(async()=>window.store.getState().startGeneration());
    assert.equal(submittedGenerations.length,1,'Submit through useStore.startGeneration');
    assert.equal(submittedGenerations[0].video_prompt_type,'I','Normalize FLUX 3 references to its only supported selector');
    assert.deepEqual(submittedGenerations[0].image_refs,
      Array.from({length:10},(_,index)=>`uploads/ref-${index+1}.png`),
      'Upload and submit all ten references in their original order');

    await page.evaluate(id=>{
      window.store.setState(state=>({
        isGenerating:false,jobs:[],imageRefs:[],imageRefType:'KI',
        params:{...state.params,model_type:id,prompt:'A simple text-only scene.',image_refs:undefined,video_prompt_type:'KI'},
      }));
    },id);
    await page.evaluate(async()=>window.store.getState().startGeneration());
    assert.equal(submittedGenerations.length,2,'Submit the no-reference case');
    assert.equal(submittedGenerations[1].video_prompt_type,'','Clear stale reference selectors when no references remain');
    assert.equal(Object.hasOwn(submittedGenerations[1],'image_refs'),false,'Do not submit stale reference paths');

    await page.getByRole('button',{name:'Choose model',exact:true}).click();
    await page.getByRole('button',{name:/^Flux 2 Klein 9B/}).click();
    await page.waitForFunction(modelType=>window.store.getState().params.model_type===modelType
      && !window.store.getState().modelOptionsLoading,previous);
    await page.evaluate(modelType=>{
      window.store.setState(state=>({
        isGenerating:false,jobs:[],generationMode:'image',studioImageWorkflow:'generate',
        imageRefs:[new File(['flux2-reference'],'flux2-reference.png',{type:'image/png'})],imageRefType:'KI',
        params:{...state.params,model_type:modelType,prompt:'Preserve Flux 2 reference behavior.',
          image_mode:1,image_refs:undefined,video_prompt_type:'K'},
      }));
    },previous);
    await page.evaluate(async()=>window.store.getState().startGeneration());
    assert.equal(submittedGenerations[2].video_prompt_type,'KI','Preserve Flux 2 KI selector behavior');

    await page.evaluate(modelType=>window.store.setState(state=>({
      enabledModels:new Set([...state.enabledModels,modelType]),
    })),qwenId);
    await page.getByRole('button',{name:'Choose model',exact:true}).click();
    await page.getByRole('button',{name:/^Qwen Image 2.1 7B/}).click();
    await page.waitForFunction(modelType=>window.store.getState().params.model_type===modelType
      && !window.store.getState().modelOptionsLoading,qwenId);
    await page.evaluate(modelType=>{
      window.store.setState(state=>({
        isGenerating:false,jobs:[],generationMode:'image',studioImageWorkflow:'inpaint',
        imageWorkflowSourcePath:'uploads/qwen-source.png',imageWorkflowMaskPath:'uploads/qwen-mask.png',
        imageRefs:[new File(['qwen-reference'],'qwen-reference.png',{type:'image/png'})],imageRefType:'I',
        params:{...state.params,model_type:modelType,prompt:'Edit the masked source image.',
          image_mode:2,image_guide:'uploads/qwen-source.png',image_mask:'uploads/qwen-mask.png',
          image_refs:undefined,video_prompt_type:'KI'},
      }));
    },qwenId);
    await page.evaluate(async()=>window.store.getState().startGeneration());
    assert.equal(submittedGenerations[3].video_prompt_type,'VAGI','Keep Qwen inpaint and reference controls');
    assert.equal(submittedGenerations[3].image_mode,2);
    assert.equal(submittedGenerations[3]._studio_image_workflow,'inpaint');
    assert.equal(submittedGenerations[3].image_guide,'uploads/qwen-source.png');
    assert.equal(submittedGenerations[3].image_mask,'uploads/qwen-mask.png');
    assert.deepEqual(submittedGenerations[3].image_refs,[uploadedPaths[11]]);

    await page.evaluate(id=>window.store.getState().toggleModelEnabled(id),id);
    await page.waitForTimeout(250);
    await context.close();
    ({page,context}=await open(390));
    assert.equal(await page.evaluate(id=>window.store.getState().enabledModels.has(id),id),false,'Keep explicit opt-out');
    assert.equal(errors.length,0,errors.join('\n'));
    await context.close();
    console.log('FLUX 3 LoRA picker visibility and selection, ten-reference submission, stale-selector cleanup, Flux 2 KI and Qwen inpaint preservation, workflow filtering and opt-out passed.');
  } finally {await browser.close();}
})().catch(error=>{console.error(error);process.exitCode=1;});
