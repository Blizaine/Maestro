// Real Fish controls and store actions; all network requests are intercepted.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const root = path.resolve(__dirname, '../..');
const esbuild = require(path.join(root, 'ui/node_modules/esbuild'));
const playwright = require(process.env.MAESTRO_PLAYWRIGHT || 'playwright');
const origin = 'http://fish-sampling.test';

(async () => {
  const bundle = await esbuild.build({stdin:{resolveDir:path.join(root,'ui'),loader:'tsx',contents:`
    import React from 'react'; import {createRoot} from 'react-dom/client';
    import {useStore} from './src/stores/useStore';
    import {FishSpeechSampling} from './src/components/Sidebar/FishSpeechSampling';
    window.store=useStore;
    const defaults=${JSON.stringify(JSON.parse(fs.readFileSync(path.join(root,'app/defaults/fish_drama3_int4.json'),'utf8')))};
    const options={model_type:'fish_drama3_int4',architecture:'fish_drama3',audio_only:true,
      max_voice_count:3,audio_mode_from_voice_count:true,guidance_max_phases:0,
      duration_slider:{min:1,max:300,default:60,increment:1},default_params:defaults};
    window.mount=()=>{
      useStore.setState(s=>({generationMode:'audio',audioSubMode:'speech',durationSeconds:30,
        params:{...s.params,...defaults,model:undefined,model_type:'fish_drama3_int4',
          prompt:'[calm] Hello. [whispering] Is anyone there?',activated_loras:[],loras_multipliers:''},
        models:[{model_type:'fish_drama3_int4',architecture:'fish_drama3',family:'tts',is_downloaded:true}],
        modelOptions:options,enabledModels:new Set(['fish_drama3_int4']),ttsVoiceCount:0,ttsVoices:[],
        activeWorkspace:'test',jobs:[],loadModelOptions:async()=>useStore.setState({modelOptions:options}),
        loadLoras:async()=>useStore.setState({availableLoras:[]})}));
    };
    window.mount();
    createRoot(document.getElementById('root')).render(<div style={{maxWidth:440,padding:20}}><FishSpeechSampling/></div>);
  `},bundle:true,write:false,jsx:'automatic',define:{'process.env.NODE_ENV':'"development"'},logLevel:'silent'});
  const assets=path.join(root,'ui/dist/assets');
  const css=fs.readFileSync(path.join(assets,fs.readdirSync(assets).find(name=>name.endsWith('.css'))),'utf8');
  const browser=await playwright.chromium.launch({headless:true,...(process.platform==='win32'?{
    executablePath:process.env.MAESTRO_CHROME || 'C:/Program Files/Google/Chrome/Application/chrome.exe'}:{})});
  try {
    for(const width of [390,1280]) {
      const page=await browser.newPage({viewport:{width,height:900}});
      page.setDefaultTimeout(7000);
      const errors=[],submissions=[];
      page.on('pageerror',error=>errors.push(error.stack||error.message));
      await page.route('**/*',route=>{
        const request=route.request(),url=new URL(request.url());
        assert.equal(url.origin,origin,'no live app or internet requests');
        if(url.pathname==='/') return route.fulfill({contentType:'text/html',body:'<meta name="viewport" content="width=device-width,initial-scale=1"><div id="root"></div>'});
        if(url.pathname==='/api/v1/generate') {
          submissions.push(request.postDataJSON());
          return route.fulfill({json:{job_id:'mock-fish-'+submissions.length,status:'held'}});
        }
        return route.fulfill({json:{}});
      });
      await page.goto(origin);
      await page.addStyleTag({content:css});
      await page.addScriptTag({content:bundle.outputFiles[0].text});
      const topK=page.getByLabel('Fish speech Top-k'),topP=page.getByLabel('Fish speech Top-p');
      await topK.waitFor();
      assert.equal(await topK.inputValue(),'0');
      await page.evaluate(()=>window.store.getState().startGeneration('queue'));
      assert.equal(submissions.at(-1).top_k,0,'uncapped zero reaches the generation request');
      await topK.fill('30');
      await topP.fill('0.9');
      await page.evaluate(()=>window.store.getState().startGeneration('queue'));
      assert.equal(submissions.at(-1).top_k,30,'explicit positive values are preserved');
      assert.equal(submissions.at(-1).top_p,0.9);
      await page.evaluate(async()=>{
        const state=window.store.getState();
        window.store.setState({selectedOutputMeta:{params:{...state.params,top_k:30,top_p:0.85}},selectedOutput:-1});
        await window.store.getState().loadSettingsFromOutput();
      });
      assert.equal(await topK.inputValue(),'30','Load Settings retains saved positive Top-k');
      assert.equal(await topP.inputValue(),'0.85');
      await topK.fill('0');
      assert.equal(await page.getByText('No cap',{exact:true}).count(),1);
      await page.screenshot({path:path.join(root,'.codex-tmp/fish-short-speech-20261023',`sampling-${width}.png`)});
      await page.evaluate(()=>window.store.setState({modelOptions:{architecture:'qwen3_tts_base'}}));
      await topK.waitFor({state:'detached'});
      assert.deepEqual(errors,[]);
      await page.close();
      console.log(`Fish sampling ${width}px: no-cap/default, positive controls, exact submission, saved settings and model scoping passed`);
    }
  } finally { await browser.close(); }
})().catch(error=>{console.error(error);process.exitCode=1;});
