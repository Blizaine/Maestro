// Real Studio/Director controls and store; every HTTP call is intercepted.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const root = path.resolve(process.env.MAESTRO_TEST_ROOT || path.join(__dirname, '../..'));
const esbuild = require(path.join(root, 'ui/node_modules/esbuild'));
const {chromium} = require(process.env.MAESTRO_PLAYWRIGHT || 'playwright');
const STYLE = 'FilmStyle.safetensors';
const optionsFor = model => ({
  model_type: model, architecture: 'minimax_h3', minimax_h3_pika_flex: true,
  experimental_loras_available: true, loras_disabled: false,
  omni_reference: model.includes('ref2va'), fps: 24, frames_minimum: 124,
  frames_maximum: 362, frames_steps: 17, guidance_max_phases: 1,
  default_num_inference_steps: 6, lock_inference_steps: true,
  lock_guidance_scale: true, lock_guidance_phases: true,
  t2v_class: true, i2v_class: !model.includes('ref2va'),
});

(async () => {
  const bundle = await esbuild.build({stdin: {contents: `
    import React from 'react'; import {createRoot} from 'react-dom/client';
    import {useStore} from './src/stores/useStore';
    import {AdvancedSettings} from './src/components/Sidebar/AdvancedSettings';
    import {DirectorH3Optimizations} from './src/components/Sidebar/DirectorH3Optimizations';
    window.store=useStore;
    window.mount=kind=>{window.reactRoot?.unmount(); window.reactRoot=createRoot(document.getElementById('root'));
      window.reactRoot.render(kind==='director'?<DirectorH3Optimizations/>:<AdvancedSettings/>);};
  `, resolveDir:path.join(root,'ui'), loader:'tsx'}, bundle:true, write:false,
    jsx:'automatic', define:{'process.env.NODE_ENV':'"development"'}, logLevel:'silent'});
  const browser = await chromium.launch({headless:true, ...(process.platform === 'win32'
    ? {executablePath:process.env.MAESTRO_CHROME || 'C:/Program Files/Google/Chrome/Application/chrome.exe'} : {})});
  try {
    const page = await browser.newPage();
    const errors = [], submissions = [], pipelines = [];
    page.on('pageerror', error => {errors.push(error.message); console.error(error.stack || error.message);});
    await page.route('**/*', route => {
      const request = route.request(), url = new URL(request.url());
      const json = body => route.fulfill({json:body});
      if(url.pathname === '/') return route.fulfill({contentType:'text/html', body:'<div id="root"></div>'});
      if(url.pathname.includes('/media-flow/capabilities')) return json({
        neural_rendering:{available:false,reason:'Isolated test',scales:[]},
        frame_generation:{available:false,reason:'Isolated test',factors:[]},
      });
      if(url.pathname.includes('/model-options/')) return json(optionsFor(decodeURIComponent(url.pathname.split('/').pop())));
      if(url.pathname.includes('/defaults/')) return json({num_inference_steps:6, guidance_scale:1});
      if(url.pathname.endsWith('/details')) return json({loras:[{filename:STYLE}]});
      if(url.pathname.startsWith('/api/v1/loras/')) return json({loras:[STYLE], guidance_max_phases:1});
      if(url.pathname === '/api/v1/generate') {
        submissions.push(request.postDataJSON());
        return route.fulfill({status:400,json:{detail:'Isolated request captured; no generation'}});
      }
      if(url.pathname === '/api/v1/director/queue') {
        if(request.method() === 'POST') pipelines.push(request.postDataJSON().params);
        return json({version:1,paused:true,entries:[],running:false});
      }
      return json({presets:[], jobs:[], queue:[], pipelines:[], loras:[], checks:[], models:[]});
    });
    await page.goto('http://pika-loras.test');
    const html = fs.readFileSync(path.join(root,'ui/dist/index.html'),'utf8');
    const css = /href="(\/assets\/[^" ]+\.css)"/.exec(html)?.[1];
    if(css) await page.addStyleTag({content:fs.readFileSync(path.join(root,'ui/dist',css.slice(1)),'utf8')});
    await page.addScriptTag({content:bundle.outputFiles[0].text});
    for(const model of ['minimax_h3_pika_flex','minimax_h3_ref2va_pika_flex']) {
      const options = optionsFor(model);
      await page.evaluate(({model,options,style}) => {
        const state = window.store.getState();
        const references = options.omni_reference ? [{type:'image',path:'/uploads/ref.png',filename:'ref.png',role:'Traveler'}] : [];
        window.store.setState({models:[{model_type:model,name:model,family:'minimax_h3',architecture:'minimax_h3',
            is_t2v:true,is_i2v:!options.omni_reference,is_downloaded:true,omni_reference:options.omni_reference,fps:24}],
          modelsLoaded:true, modelsLoading:false, modelOptions:options, modelOptionsLoading:false,
          enabledModels:new Set([model]), generationMode:'video', sidebarMode:'studio',
          studioVideoWorkflow:options.omni_reference?'references':'frames',
          studioVideoCreateRoute:'auto',studioVideoEffectiveCreateRoute:options.omni_reference?'omni':'generate',
          selectedModelPerMode:{video:model}, startImage:null,endImage:null,imageRefs:[],jobs:[],isGenerating:false,
          enhanceOnGeneration:false,enhanceOnGenerationDefault:false,
          durationSeconds:124/24,slidingWindowLocked:false,slidingWindowSeconds:124/24,
          resolutionPreset:'480p',aspectRatio:'16:9', availableLoras:[style], loraWeights:{[style]:[0.65]},
          params:{...state.params,model_type:model,prompt:'A small toy robot crosses a garden.',resolution:'832x480',
            image_mode:0,video_length:124,num_inference_steps:6,guidance_scale:1,flow_shift:12,
            audio_flow_shift:3,sample_solver:'euler',activated_loras:[style],loras_multipliers:'0.65',
            minimax_h3_experimental_loras:false,minimax_h3_turbo_mode:false,
            minimax_h3_multi_window:false,minimax_h3_reference_sequence:false,
            minimax_h3_extended_duration:false,minimax_h3_references:references,
            minimax_h3_window_storyboard:false,_duration_planning_mode:'duration'},
        });
        window.mount('studio');
      }, {model,options,style:STYLE});
      await page.getByRole('button',{name:/Advanced settings/}).click();
      const section = page.getByTestId('advanced-loras');
      if(await section.getAttribute('open') === null) await section.locator('summary').click();
      const toggle = page.getByRole('checkbox',{name:'Enable LoRAs (experimental)',exact:true});
      await toggle.waitFor();
      assert.equal(await toggle.isChecked(),false);
      assert.match(await page.getByRole('dialog').innerText(), /required Pika adapter/);
      await page.evaluate(() => window.store.getState().startGeneration());
      assert.deepEqual(submissions.at(-1)?.activated_loras,[], `${model}: off means no hidden extra adapter`);
      assert.equal(await page.evaluate(() => window.store.getState().params.activated_loras.length),1,
        'Turning experiments off retains saved selections in the form');
      await toggle.check();
      await page.evaluate(() => window.store.setState({jobs:[],isGenerating:false}));
      await page.evaluate(() => window.store.getState().startGeneration());
      assert.deepEqual(submissions.at(-1)?.activated_loras,[STYLE]);
      assert.equal(submissions.at(-1).minimax_h3_experimental_loras,true);
      assert.equal(submissions.at(-1).loras_multipliers,'0.65');
      assert.equal(submissions.at(-1).num_inference_steps,6);
      const saved = submissions.at(-1);
      await page.evaluate(async params => {
        window.store.setState({jobs:[],isGenerating:false,
          selectedOutputMeta:{source:'sidecar',params},selectedOutput:0,
          outputs:[{name:'saved.mp4',type:'video',mode:'video',url:'/saved.mp4',
            path:'/outputs/saved.mp4',size:1,created_at:1}],
        });
        await window.store.getState().loadSettingsFromOutput();
      }, saved);
      assert.equal(await page.evaluate(() => window.store.getState().params.minimax_h3_experimental_loras),true,
        'Pencil restoration retains explicit opt-in and selected weights');
      await page.evaluate(style => window.store.setState({
        sidebarMode:'director', directorSkill:'short_film', shortFilmPath:'story',
        shortFilmTargetDuration:12,directorStep:'style',directorLoading:false,directorError:null,
        directorAnalysis:null,directorPlannedClips:[],directorClipPlans:[],directorClipImages:[],
        directorH3References:[],directorReferenceImage:null,directorReferenceImagePath:'',
        directorCharacterRefPaths:[],directorLocationRefPaths:[],directorSpeakerMappings:[],
        directorAudioPath:'',directorQueueEditingEntryId:null,
        savedLoraPerMode:{video:{activated_loras:[style],loras_multipliers:'0.65',
          loraWeights:{[style]:[0.65]},availableLoras:[style]}},
      }),STYLE);
      await page.evaluate(() => window.mount('director'));
      const directorToggle = page.getByRole('checkbox',{name:'Enable LoRAs (experimental)',exact:true});
      await directorToggle.waitFor();
      assert.equal(await directorToggle.isChecked(),false, 'Studio opt-in does not enable Director');
      await directorToggle.check();
      assert.equal(await page.evaluate(model => window.store.getState().directorH3ExperimentalLorasByModel[model],model),true);
      await page.evaluate(() => window.store.getState().startDirectorPipeline('queue'));
      assert.equal(pipelines.at(-1)?.video_params?.minimax_h3_experimental_loras,true);
      assert.deepEqual(pipelines.at(-1)?.video_loras?.activated_loras,[STYLE]);
      assert.equal(pipelines.at(-1).video_loras.loras_multipliers,'0.65');
      await directorToggle.uncheck();
      await page.evaluate(() => window.store.getState().startDirectorPipeline('queue'));
      assert.equal(pipelines.at(-1)?.video_params?.minimax_h3_experimental_loras,false);
      assert.deepEqual(pipelines.at(-1)?.video_loras?.activated_loras,[]);
      assert.equal(await page.evaluate(() => window.store.getState().params.minimax_h3_experimental_loras),true,
        'Director opt-out does not alter Studio');
    }
    assert.deepEqual(errors,[]);
    console.log('Pika Frames/References experimental toggle, real submit weights, inactive saved choices, six steps, and Director isolation passed.');
  } finally {await browser.close();}
})().catch(error => {console.error(error);process.exitCode=1;});
