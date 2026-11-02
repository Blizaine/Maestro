// Real picker components and store. All requests are intercepted; no live app.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const root = path.resolve(__dirname, '../..');
const esbuild = require(path.join(root, 'ui/node_modules/esbuild'));
const playwright = require(process.env.MAESTRO_PLAYWRIGHT || 'playwright');
const origin = 'http://lora-filters.test';
const names = ['portrait.safetensors', 'minimax_h3_turbo_v4.safetensors', 'Wan-Lightning-4steps.safetensors', 'renamed.safetensors'];
const rows = names.map(filename => ({filename,display_name:filename,trained_words:[],preview_url:null,
  civitai_model_id:null,recommended_weights:null,has_guide:false,nsfw:false,
  ...(filename === 'renamed.safetensors' ? {is_acceleration:true} : {})}));

(async () => {
  const bundle = await esbuild.build({stdin:{resolveDir:path.join(root,'ui'),loader:'tsx',contents:`
    import React from 'react'; import {createRoot} from 'react-dom/client';
    import {useStore} from './src/stores/useStore';
    import {LoraSelector} from './src/components/SettingsDrawer/LoraSelector';
    import {DirectorLoraSelector} from './src/components/SettingsDrawer/DirectorLoraSelector';
    import {AdvancedSettings} from './src/components/Sidebar/AdvancedSettings';
    import {LoraBrowser} from './src/components/LoraBrowser/LoraBrowser';
    import {isAccelerationLora} from './src/lib/loraAcceleration';
    window.store=useStore; window.classify=isAccelerationLora;
    const view=createRoot(document.getElementById('root'));
    window.mountFast=()=>{
      useStore.setState(s=>({sidebarMode:'studio',generationMode:'video',studioVideoWorkflow:'frames',
        params:{...s.params,model_type:'minimax_h3_fasth3',prompt:'A boat moves.',seed:42,video_length:124,
          activated_loras:[],loras_multipliers:'',minimax_h3_experimental_loras:false},
        models:[{model_type:'minimax_h3_fasth3',family:'minimax_h3'}],
        modelOptions:{model_type:'minimax_h3_fasth3',architecture:'minimax_h3',minimax_h3_fasth3:true,
          experimental_loras_available:true,loras_disabled:false,lock_inference_steps:true,lock_guidance_scale:true,no_negative_prompt:true},
        savedLoraPerMode:{},presets:[],loraWeights:{},availableLoras:[]}));
      view.render(<div data-sidebar><AdvancedSettings compact/></div>);
    };
    window.mountPickers=()=>{
      useStore.setState(s=>({params:{...s.params,model_type:'test_h3',activated_loras:['minimax_h3_turbo_v4.safetensors'],
        loras_multipliers:'0.35'},modelOptions:{guidance_max_phases:1},availableLoras:${JSON.stringify(names)},
        loraWeights:{'minimax_h3_turbo_v4.safetensors':[0.35]},savedLoraPerMode:{video:{
          activated_loras:['minimax_h3_turbo_v4.safetensors'],loras_multipliers:'0.35',
          loraWeights:{'minimax_h3_turbo_v4.safetensors':[0.35]},availableLoras:${JSON.stringify(names)}}}}));
      view.render(<div><section id="studio"><LoraSelector/></section><section id="director">
        <DirectorLoraSelector mode="video" modelType="test_h3"/></section></div>);
    };
    window.mountLibrary=()=>{
      useStore.setState({loraBrowserOpen:true,loraBrowserArch:null,loraBrowserDefaultDir:null,
        civitSearchResults:[],civitSearchCursor:null,civitSelectedModel:null});
      view.render(<LoraBrowser/>);
    };
  `},bundle:true,write:false,jsx:'automatic',define:{'process.env.NODE_ENV':'"development"'},logLevel:'silent'});
  const assets=path.join(root,'ui/dist/assets');
  const css=fs.readFileSync(path.join(assets,fs.readdirSync(assets).find(name=>name.endsWith('.css'))),'utf8');
  const browser=await playwright.chromium.launch({headless:true,...(process.platform==='win32'?{
    executablePath:process.env.MAESTRO_CHROME || 'C:/Program Files/Google/Chrome/Application/chrome.exe'}:{})});
  try {
    for(const width of [390,1280]) {
      const page=await browser.newPage({viewport:{width,height:900}});
      page.setDefaultTimeout(7000);
      const errors=[],requests=[];
      page.on('pageerror',error=>{errors.push(error.stack||error.message);console.error('Browser fixture error:',error.stack||error.message);});
      await page.route('**/*',route=>{
        const request=route.request(),url=new URL(request.url());
        assert.equal(url.origin,origin,'requests never reach the live app or internet');
        if(url.pathname==='/') return route.fulfill({contentType:'text/html',body:'<meta name="viewport" content="width=device-width,initial-scale=1"><div id="root"></div>'});
        requests.push(request.method()+' '+url.pathname);
        const json=body=>route.fulfill({json:body});
        if(url.pathname==='/api/v1/presets') return json({presets:[]});
        if(url.pathname==='/api/v1/media-flow/capabilities') return json({
          neural_rendering:{available:false,reason:'fixture',scales:[]},
          frame_generation:{available:false,reason:'fixture',factors:[]}});
        if(url.pathname==='/api/v1/loras/installed') return json({loras:rows.map(row=>({...row,directory:'test',name:row.filename,base_model:'Test',linked:false,size_bytes:100}))});
        if(url.pathname.endsWith('/details')) return json({loras:url.pathname.includes('fasth3')?[rows[0]]:rows,guidance_max_phases:1});
        if(url.pathname.includes('/guide/')) return json({guide:null});
        if(url.pathname.startsWith('/api/v1/loras/')) return json({loras:url.pathname.includes('fasth3')?[names[0]]:names,guidance_max_phases:1});
        if(url.pathname==='/api/v1/civitai/filters') return json({filters:[]});
        if(url.pathname==='/api/v1/civitai/downloads') return json({downloads:[]});
        if(url.pathname==='/api/v1/studio-preferences' && request.method()==='PUT') return json({});
        // Browser setup reads optional local catalogue endpoints. Empty
        // fixtures are sufficient; mutations other than preferences fail.
        if(request.method()==='GET') return json({filters:[],models:[],directories:[],items:[],loras:[],downloads:[],metadata:{}});
        throw new Error('Unexpected mutation '+request.method()+' '+url.pathname);
      });
      await page.goto(origin);
      await page.addStyleTag({content:css});
      await page.addScriptTag({content:bundle.outputFiles[0].text});
      await page.evaluate(()=>window.mountFast());
      await page.getByRole('button',{name:/Advanced settings/}).click();
      await page.getByText('LoRAs & presets',{exact:true}).click();
      const toggle=page.getByLabel('Enable LoRAs (experimental)');
      assert.equal(await toggle.isChecked(),false);
      assert.equal(await page.getByLabel('Turbo LoRA filter').count(),0,'FastH3 controls default to off');
      await toggle.check();
      await page.getByLabel('Turbo LoRA filter').waitFor();
      await page.locator('button[title="File: portrait.safetensors"]').first().click();
      assert.deepEqual(await page.evaluate(()=>window.store.getState().params.activated_loras),['portrait.safetensors']);
      const screenshotDir=path.join(root,'.codex-tmp/fasth3-lora-controls-20261023');fs.mkdirSync(screenshotDir,{recursive:true});
      await page.screenshot({path:path.join(screenshotDir,`fasth3-toggle-${width}.png`),fullPage:true});
      await toggle.uncheck();
      assert.deepEqual(await page.evaluate(()=>window.store.getState().params.activated_loras),['portrait.safetensors'],'turning off preserves the saved selection');
      await page.getByText('Saved LoRA selections are inactive while this is off.').waitFor();

      await page.evaluate(()=>window.mountPickers());
      const selectors=page.getByLabel('Turbo LoRA filter');
      await selectors.nth(1).waitFor();
      await selectors.first().selectOption('hide');
      await page.waitForFunction(()=>document.querySelectorAll('select[aria-label="Turbo LoRA filter"]')[1].value==='hide');
      for(const id of ['studio','director']) {
        const section=page.locator('#'+id);
        await section.locator('button[title="File: portrait.safetensors"]').waitFor();
        await section.locator('button[title="File: minimax_h3_turbo_v4.safetensors"]').waitFor();
        assert.equal(await section.locator('button[title="File: Wan-Lightning-4steps.safetensors"]').count(),0);
        await section.locator('button[title="File: renamed.safetensors"]').waitFor({state:'detached'});
        await section.getByText(/kept visible outside this filter/).waitFor();
      }
      assert.deepEqual(await page.evaluate(()=>window.store.getState().loraWeights['minimax_h3_turbo_v4.safetensors']),[0.35]);
      assert.equal(await page.evaluate(()=>localStorage.getItem('maestro_lora_acceleration_filter')),'hide');
      await page.screenshot({path:path.join(screenshotDir,`picker-filter-${width}.png`),fullPage:true});
      await selectors.nth(1).selectOption('only');
      await page.locator('#studio button[title="File: Wan-Lightning-4steps.safetensors"]').waitFor();
      assert.equal(await page.locator('#studio button[title="File: portrait.safetensors"]').count(),0);
      assert.deepEqual(await page.evaluate(()=>window.store.getState().params.activated_loras),['minimax_h3_turbo_v4.safetensors']);

      await page.evaluate(()=>window.mountLibrary());
      await page.getByRole('button',{name:'My LoRAs',exact:true}).click();
      await page.getByLabel('Turbo LoRA filter').waitFor();
      assert.equal(await page.getByLabel('Turbo LoRA filter').inputValue(),'only');
      await page.getByLabel('Turbo LoRA filter').selectOption('all');
      await page.getByText('portrait.safetensors',{exact:true}).first().waitFor();
      await page.getByLabel('Turbo LoRA filter').selectOption('hide');
      await page.getByText('portrait.safetensors',{exact:true}).first().waitFor();
      assert.equal(await page.getByText('Wan-Lightning-4steps.safetensors',{exact:true}).count(),0);
      assert.deepEqual(errors,[]);
      assert.ok(!requests.some(value=>value.includes('/generate')),'the browser fixture never renders');
      await page.screenshot({path:path.join(screenshotDir,`library-filter-${width}.png`),fullPage:true});
      await page.reload();
      await page.addStyleTag({content:css});
      await page.addScriptTag({content:bundle.outputFiles[0].text});
      await page.evaluate(()=>window.mountPickers());
      assert.equal(await page.getByLabel('Turbo LoRA filter').first().inputValue(),'hide','the shared filter survives page reload');
      await page.close();
    }
  } finally {await browser.close();}
  console.log('Experimental toggle and shared Studio/Director/installed Turbo filters passed on mobile and desktop.');
})().catch(error=>{console.error(error);process.exitCode=1;});
