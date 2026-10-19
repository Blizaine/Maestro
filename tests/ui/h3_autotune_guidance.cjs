// Real React duration guidance at desktop and phone widths; no generation requests.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const root = path.resolve(__dirname, '../..');
const esbuild = require(path.join(root, 'ui/node_modules/esbuild'));
const playwright = require(process.env.MAESTRO_PLAYWRIGHT || 'playwright');
const output = path.join(root, '.codex-tmp/h3-autotune-guidance-20261009');
(async()=>{
  const bundle=await esbuild.build({stdin:{contents:`
    import React from 'react'; import {createRoot} from 'react-dom/client';
    import {useStore} from './src/stores/useStore';
    import {H3DurationGuidance} from './src/components/Sidebar/H3DurationGuidance';
    window.store=useStore;
    function Fixture(){const frames=useStore(s=>s.params.video_length);
      return <div style={{width:'min(360px,100vw)',padding:12,boxSizing:'border-box'}}>
        <H3DurationGuidance frames={frames} selectedFrames={frames} minimumFrames={124}
          maximumFrames={719} continuationFrames={0}/></div>}
    window.mount=()=>createRoot(document.getElementById('root')).render(<Fixture/>);
  `,resolveDir:path.join(root,'ui'),loader:'tsx'},bundle:true,write:false,jsx:'automatic',
  define:{'process.env.NODE_ENV':'"development"'},logLevel:'silent'});
  const browser=await playwright.chromium.launch({headless:true,
    ...(process.platform==='win32'?{executablePath:process.env.MAESTRO_CHROME||'C:/Program Files/Google/Chrome/Application/chrome.exe'}:{})});
  try{
    fs.mkdirSync(output,{recursive:true});
    const page=await browser.newPage({viewport:{width:1000,height:900}});
    const errors=[];page.on('pageerror',e=>errors.push(e.message));
    await page.route('**/*',route=>route.fulfill({contentType:'text/html',body:'<div id="root"></div>'}));
    await page.goto('http://h3-auto-guidance.test');
    await page.addScriptTag({content:bundle.outputFiles[0].text});
    const assets=path.join(root,'ui/dist/assets');
    await page.addStyleTag({content:fs.readFileSync(path.join(assets,fs.readdirSync(assets).find(f=>f.endsWith('.css'))),'utf8')});
    await page.evaluate(()=>{
      const s=window.store.getState();
      window.store.setState({generationMode:'video',modelOptions:{model_type:'minimax_h3_ref2va_fused_turbo',
        architecture:'minimax_h3_ref2va',minimax_h3_fused_turbo:true,sla_attention_default:true,
        omni_reference:true,frames_minimum:124,frames_maximum:345,frames_steps:17},
        systemStats:{gpu:{vram_total_gb:12},ram:{total_gb:31.76}},servicesConfig:{auto_performance:true},
        params:{...s.params,resolution:'1920x1088',video_length:243,sliding_window_size:243,
          num_inference_steps:6,override_attention:'sla',prompt:'An armored kitten runs.',
          minimax_h3_references:[],activated_loras:[]}});
      window.mount();
    });
    const advice=page.getByTestId('h3-head-split-guidance');
    await advice.waitFor();
    assert.equal(await advice.getAttribute('data-recommended-split'),'1');
    assert.match(await advice.innerText(),/Low/);
    await page.evaluate(()=>{const s=window.store.getState();window.store.setState({
      params:{...s.params,video_length:362,sliding_window_size:362}})});
    await page.waitForFunction(()=>document.querySelector('[data-recommended-split="2"]'));
    assert.match(await advice.innerText(),/Medium/);
    await page.screenshot({path:path.join(output,'desktop-fused-medium.png')});
    await page.setViewportSize({width:390,height:844});
    await page.evaluate(()=>{const s=window.store.getState();window.store.setState({
      modelOptions:{...s.modelOptions,minimax_h3_fused_turbo:false,sla_attention_default:false},
      params:{...s.params,video_length:243,sliding_window_size:243,num_inference_steps:20,override_attention:'sage2'}})});
    await page.waitForFunction(()=>document.querySelector('[data-speed-caution="true"]'));
    assert.match(await advice.innerText(),/Very slow workload/);
    assert.match(await advice.innerText(),/shorter 720p/i);
    const box=await advice.boundingBox();assert.ok(box.x>=0&&box.x+box.width<=390,'Guidance fits a phone viewport');
    await page.evaluate(()=>window.store.setState({servicesConfig:{auto_performance:false}}));
    await page.waitForFunction(()=>document.querySelector('[data-testid="h3-head-split-guidance"]').textContent.includes('Auto-tune is off'));
    assert.match(await advice.innerText(),/saved setting stays in control/);
    await page.screenshot({path:path.join(output,'phone-pruned-speed-caution.png')});
    await page.evaluate(()=>{const s=window.store.getState();window.store.setState({systemStats:{...s.systemStats,gpu:{vram_total_gb:24}}})});
    assert.equal(await advice.count(),0,'12GB advice stays out of other VRAM tiers');
    assert.deepEqual(errors,[]);
    console.log('H3 Auto-tune React guidance: Low/Medium, Pruned speed caution, manual mode, phone layout and scoping passed');
  }finally{await browser.close()}
})().catch(e=>{console.error(e);process.exitCode=1});
