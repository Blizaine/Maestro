// Real React/Zustand duration popup on desktop and mobile; all API calls mocked.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const root = path.resolve(__dirname, '../..');
const esbuild = require(path.join(root, 'ui/node_modules/esbuild'));
const playwright = require(process.env.MAESTRO_PLAYWRIGHT || 'playwright');
const output = path.join(root, '.codex-tmp/h3-duration-guidance-20261004');
const modelOptions = {
  architecture: 'minimax_h3_fused_turbo', model_type: 'minimax_h3_fused_turbo', fps: 24,
  frames_minimum: 124, frames_maximum: 345, frames_steps: 17, sliding_window: true,
  sliding_window_defaults: {window_min: 124, window_max: 345, window_step: 17,
    window_default: 243, overlap_min:1, overlap_step:17, overlap_offset:1,
    overlap_default: 18, discard_last_frames: 0},
  sliding_window_memory_policy: {resolution_bands: [{min_pixels: 0, vram_tiers: [
    {max_vram_gb: 16, frames: 243}, {frames: 345}]}]},
};
(async () => {
  const bundle = await esbuild.build({stdin: {contents: `
    import React, {useState} from 'react'; import {createRoot} from 'react-dom/client';
    import {useStore} from './src/stores/useStore';
    import {OutputFormatControls} from './src/components/Sidebar/OutputFormatControls';
    import {SidebarLayoutContext} from './src/components/Sidebar/SidebarPanels';
    window.store = useStore;
    function Fixture() {
      const [sidebar,setSidebar] = useState(null);
      return <div ref={setSidebar} style={{position:'fixed',left:0,top:0,width:'min(360px,100vw)',height:'100vh'}}>
        <SidebarLayoutContext.Provider value={{sidebar,settings:null}}>
          <div style={{position:'absolute',bottom:24,left:16,right:16,display:'flex',gap:8}}><OutputFormatControls/></div>
        </SidebarLayoutContext.Provider>
      </div>;
    }
    window.mount = () => createRoot(document.getElementById('root')).render(<Fixture/>);
  `, resolveDir: path.join(root,'ui'),loader:'tsx'},bundle:true,write:false,jsx:'automatic',
    define:{'process.env.NODE_ENV':'"development"'},logLevel:'silent'});
  const browser = await playwright.chromium.launch({headless:true,
    ...(process.platform === 'win32' ? {executablePath:process.env.MAESTRO_CHROME || 'C:/Program Files/Google/Chrome/Application/chrome.exe'} : {})});
  try {
    fs.mkdirSync(output,{recursive:true});
    const page = await browser.newPage({viewport:{width:1000,height:800}});
    const errors=[]; page.on('pageerror',e=>errors.push(e.message));
    await page.route('**/*', route => route.fulfill({contentType:'text/html',body:'<div id="root"></div>'}));
    await page.goto('http://duration-guidance.test');
    await page.addScriptTag({content:bundle.outputFiles[0].text});
    const assets=path.join(root,'ui/dist/assets');
    await page.addStyleTag({content:fs.readFileSync(path.join(assets,fs.readdirSync(assets).find(f=>f.endsWith('.css'))),'utf8')});
    await page.evaluate(options=> {
      const store=window.store, s=store.getState();
      store.setState({modelOptions:options,generationMode:'video',studioVideoWorkflow:'frames',
        durationSeconds:243/24,slidingWindowSeconds:243/24,slidingWindowLocked:true,slidingWindowOverlap:18,
        h3WindowOverrides:{},systemStats:{gpu:{vram_total_gb:24}},
        params:{...s.params,model_type:options.model_type,resolution:'1280x704',prompt:'Two fighters spar in a gym.',
          image_mode:0,video_length:243,sliding_window_size:243,_duration_planning_mode:'duration',
          minimax_h3_extended_duration:false,minimax_h3_multi_window:false,minimax_h3_reference_sequence:false,
          minimax_h3_references:[],image_start:undefined,image_end:undefined,video_guide:undefined,audio_guide:undefined}});
      window.mount();
    },modelOptions);
    await page.getByRole('button',{name:/^Duration:/}).click();
    const dialog=page.getByRole('dialog',{name:'Duration & windows'});
    const guidance=dialog.getByTestId('h3-duration-guidance');
    await guidance.waitFor();
    assert.equal(await guidance.getAttribute('data-speed-path'),'fast');
    assert.equal(await guidance.getAttribute('data-memory-risk'),'within-profile');
    assert.match(await guidance.innerText(),/Faster path expected/);
    const initialPopup=await dialog.boundingBox();
    for (const label of [guidance.getByText('Faster path expected',{exact:true}),
      guidance.getByText(/Within GPU recommendation/)]) {
      const box=await label.boundingBox();
      assert.ok(box.y>=initialPopup.y && box.y+box.height<=initialPopup.y+initialPopup.height,
        'Speed and VRAM guidance are visible when Duration first opens');
    }
    await page.screenshot({path:path.join(output,'desktop-fast-guidance.png')});
    const slider=dialog.getByRole('slider',{name:'Window length',exact:true});
    const sliderBefore=await slider.boundingBox();
    const popupBefore=await dialog.boundingBox();
    await page.evaluate(()=>{const s=window.store.getState();s.setSlidingWindowSeconds(345/24);s.setDurationSeconds(345/24);});
    await page.waitForFunction(()=>document.querySelector('[data-speed-path="chunked"]'));
    assert.equal(await guidance.getAttribute('data-memory-risk'),'within-profile');
    assert.equal((await dialog.boundingBox()).height,popupBefore.height,'Guidance keeps popup height stable');
    assert.equal((await slider.boundingBox()).y,sliderBefore.y,'Guidance changes do not move the slider');
    await page.evaluate(()=>window.store.setState({systemStats:{gpu:{vram_total_gb:16}}}));
    await page.waitForFunction(()=>document.querySelector('[data-memory-risk="high"]'));
    assert.match(await guidance.innerText(),/16 GB GPU/);
    await page.evaluate(()=>window.store.setState({systemStats:null}));
    await page.waitForFunction(()=>document.querySelector('[data-memory-risk="unknown"]'));
    assert.match(await guidance.innerText(),/VRAM assessment unavailable/);

    await page.evaluate(()=>{
      const s=window.store.getState();window.store.setState({systemStats:{gpu:{vram_total_gb:24}},
        studioVideoWorkflow:'references',modelOptions:{...s.modelOptions,omni_reference:true,
          omni_sequence_memory_policy:{...s.modelOptions.sliding_window_memory_policy,reference_margin_steps:1}},
        params:{...s.params,minimax_h3_references:[{id:'v',type:'video',path:'/video.mp4',filename:'video.mp4',duration_seconds:10}],
          minimax_h3_reference_detail:'match'}});
      s.setSlidingWindowSeconds(243/24);s.setDurationSeconds(243/24);
    });
    await page.waitForFunction(()=>document.querySelector('[data-speed-path]')?.dataset.speedPath!=='fast');
    assert.notEqual(await guidance.getAttribute('data-speed-path'),'fast','Video reference removes confident faster advice');
    await page.evaluate(()=>{
      const s=window.store.getState();window.store.setState({studioVideoWorkflow:'frames',
        modelOptions:{...s.modelOptions,omni_reference:false},params:{...s.params,minimax_h3_references:[]}});
    });
    await dialog.getByRole('checkbox',{name:/Allow 30s clips/}).check();
    assert.equal(await slider.getAttribute('max'),'719');
    assert.equal(await guidance.getAttribute('data-speed-path'),'fast','Short actual clip retains accurate estimate under30s cap');
    assert.match(await guidance.innerText(),/for 10.1s per pass/);
    assert.match(await guidance.getByTestId('h3-extended-warning').innerText(),/selected window limit exceeds/);
    await page.evaluate(()=>window.store.getState().setDurationSeconds(30));
    await page.waitForFunction(()=>document.querySelector('[data-memory-risk="high"]'));
    assert.equal(await guidance.getAttribute('data-speed-path'),'chunked');
    assert.match(await guidance.getByTestId('h3-extended-warning').innerText(),/This pass exceeds.*14.4s/);
    assert.match(await guidance.innerText(),/30s experimental/);
    assert.equal(await page.evaluate(()=>window.store.getState().params.sliding_window_size),719);

    await page.setViewportSize({width:390,height:740});
    await guidance.scrollIntoViewIfNeeded();
    assert.ok(await dialog.evaluate(n=>n.scrollWidth<=n.clientWidth),'Guidance fits mobile popup width');
    const mobileBox=await dialog.boundingBox();
    assert.ok(mobileBox.x>=0 && mobileBox.x+mobileBox.width<=390,'Popup stays on mobile screen');
    await page.screenshot({path:path.join(output,'mobile-30s-guidance.png')});
    await guidance.getByText('Why this estimate?',{exact:true}).click();
    assert.match(await guidance.innerText(),/75,000/);
    await dialog.getByRole('checkbox',{name:/Allow 30s clips/}).uncheck();
    assert.equal(await slider.getAttribute('max'),'345');
    assert.equal(await guidance.getByTestId('h3-extended-warning').count(),0);
    // Holding one complete window must not move its slider during a mobile drag.
    await dialog.getByRole('button',{name:'Window',exact:true}).click();
    await dialog.getByRole('button',{name:'1',exact:true}).click();
    const settle=()=>page.evaluate(()=>new Promise(resolve=>requestAnimationFrame(()=>requestAnimationFrame(resolve))));
    for (const experimental of [false,true]) {
      if (experimental) await dialog.getByRole('checkbox',{name:/Allow 30s clips/}).check();
      assert.equal(await slider.getAttribute('max'),experimental ? '719' : '345');
      await slider.scrollIntoViewIfNeeded();
      const dragBox=await slider.boundingBox();
      const dragPopup=await dialog.boundingBox();
      await page.mouse.move(dragBox.x+8,dragBox.y+dragBox.height/2);
      await page.mouse.down();
      try {
        for (const fraction of [0,0.4,1,0.7,0]) {
          await page.mouse.move(dragBox.x+8+fraction*(dragBox.width-16),dragBox.y+dragBox.height/2);
          await settle();
          assert.equal(await dialog.getByRole('spinbutton',{name:'Window count',exact:true}).inputValue(),'1');
          const current=await page.evaluate(()=>{const s=window.store.getState();return [s.params.video_length,s.params.sliding_window_size];});
          assert.equal(current[0],current[1],'Single window uses the full length while dragging');
          if (fraction===1) assert.equal(current[0],experimental ? 719 : 345);
          assert.ok(Math.abs((await slider.boundingBox()).y-dragBox.y)<1,'Window slider stays under the pointer');
          assert.equal((await dialog.boundingBox()).height,dragPopup.height,'Window resizing keeps popup height stable');
        }
      } finally {await page.mouse.up();}
    }
    assert.match(await dialog.getByTestId('window-length-behavior').innerText(),/Window count stays fixed/);
    await dialog.getByRole('button',{name:'2',exact:true}).click();
    await slider.focus();
    await slider.press('End');
    await settle();
    assert.equal(await page.evaluate(()=>window.store.getState().params.video_length),1420,
      'Two extended Frames windows include one18-frame overlap');
    await dialog.getByRole('checkbox',{name:/Allow 30s clips/}).uncheck();
    await settle();
    assert.equal(await dialog.getByRole('spinbutton',{name:'Window count',exact:true}).inputValue(),'2');
    assert.equal(await page.evaluate(()=>window.store.getState().params.video_length),672,
      'Disabling the experiment preserves two complete native windows');
    await page.evaluate(()=>{
      const s=window.store.getState();window.store.setState({modelOptions:{...s.modelOptions,architecture:'ltx2'},
        params:{...s.params,model_type:'ltx2'}});
    });
    assert.equal(await guidance.count(),0,'H3 advisory stays out of other model families');
    assert.deepEqual(errors,[],'No React errors or duration loops');
    console.log('H3 duration popup: path changes, VRAM tiers, references, actual pass vs cap,30s warnings, mobile layout and family scoping passed');
  } finally {await browser.close();}
})().catch(error=>{console.error(error);process.exitCode=1;});
