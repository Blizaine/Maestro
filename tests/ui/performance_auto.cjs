// Isolated browser checks: recommendation details, learning, allocator restart
// feedback and manual memory controls disabling Auto. No live settings/GPU use.
const assert = require('node:assert/strict');
const path = require('node:path');
const fs = require('node:fs');
const root = path.resolve(__dirname, '../..');
const esbuild = require(path.join(root, 'ui/node_modules/esbuild'));
const playwright = require(process.env.MAESTRO_PLAYWRIGHT || 'playwright');

(async () => {
  const entry = `
    import React from 'react'; import {createRoot} from 'react-dom/client';
    import {useStore} from './src/stores/useStore';
    import {AutoPerformanceCard} from './src/components/SettingsDrawer/SystemSettingsPanel';
    import {MemorySettingsPanel} from './src/components/SettingsDrawer/MemorySettingsPanel';
    window.store = useStore;
    window.mount = (config) => {
      useStore.setState({systemConfig: config, servicesConfig: {auto_performance:true}});
      createRoot(document.getElementById('root')).render(<><AutoPerformanceCard/><MemorySettingsPanel/></>);
    };
  `;
  const bundle = await esbuild.build({stdin:{contents:entry, resolveDir:path.join(root,'ui'), loader:'tsx'},
    bundle:true, write:false, jsx:'automatic', define:{'process.env.NODE_ENV':'"development"'}, logLevel:'silent'});
  const browser = await playwright.chromium.launch({headless:true,
    ...(process.env.MAESTRO_CHROME ? {executablePath:process.env.MAESTRO_CHROME} : {})});
  try {
    const page = await browser.newPage({viewport:{width:390,height:844}});
    const errors = [];
    page.on('pageerror', error => errors.push(error.message));
    let auto = true;
    let applies = 0;
    const config = {vram_allocator:'vmm', vram_allocator_active:'vmm', read_ahead:false,
      smart_memory_pinning:true, perc_reserved_mem_max:0, attention_head_split:0,
      video_profile:4.5, image_profile:4.5, audio_profile:4.5, video_preload_mode:'default'};
    const detect = {hardware:{cuda_available:true,gpu_name:'RTX 3080',gpu_vram_gb:10,ram_gb:32},
      recommended:{_recommendation_label:'Profile 4.5 — partial residency',
        _recommendation_details:['Auto INT8 kernels and MMGP optimized allocator.']},
      auto_enabled:true, learning:{completed_renders:4,workloads:1,last_plan:{applied:true,
        profile:4.5,transformer_budget_mb:1792,source:'hardware',matching_renders:4,
        reasons:['Conservative host placement protects system RAM.'],warnings:['A shorter window may be needed.']}}};
    await page.route('http://maestro.test/**', async route => {
      const url = new URL(route.request().url());
      const endpoint = url.pathname;
      if (endpoint === '/') return route.fulfill({contentType:'text/html',body:'<div id="root" style="padding:16px"></div>'});
      if (endpoint.endsWith('/system-detect/apply')) { applies++; auto=true;
        return route.fulfill({json:{status:'ok', profile_changed:false,settings_reload:true,vram_allocator_restart_required:true}}); }
      if (endpoint.endsWith('/system-detect/reset-learning')) { detect.learning.completed_renders=0;
        return route.fulfill({json:{status:'ok'}}); }
      if (endpoint.endsWith('/system-detect')) return route.fulfill({json:detect});
      if (endpoint.endsWith('/services-config')) return route.fulfill({json:{auto_performance:auto}});
      if (endpoint.endsWith('/system-config')) {
        if (route.request().method()==='PUT') {
          Object.assign(config,route.request().postDataJSON()); auto=false;
          return route.fulfill({json:{status:'ok'}});
        }
        return route.fulfill({json:config});
      }
      return route.fulfill({json:{}});
    });
    await page.goto('http://maestro.test/');
    const css = fs.readdirSync(path.join(root,'ui/dist/assets')).find(name=>name.endsWith('.css'));
    await page.addStyleTag({content:fs.readFileSync(path.join(root,'ui/dist/assets',css),'utf8')});
    await page.addScriptTag({content:bundle.outputFiles[0].text});
    await page.evaluate(config => window.mount(config),config);
    await page.getByText('RTX 3080',{exact:true}).waitFor();
    assert.match(await page.locator('body').innerText(),/4 completed renders across 1 workloads/);
    await page.getByText('What Auto-tune manages',{exact:true}).click();
    assert.ok(await page.getByText('Auto INT8 kernels and MMGP optimized allocator.').isVisible());
    await page.getByText(/Last generation: Profile 4.5/).click();
    assert.ok(await page.getByText(/allowance covers transformer weights/).isVisible());
    await page.getByRole('button',{name:'Re-detect',exact:true}).click();
    await page.getByText('Auto-tune applied — restart Maestro to activate the memory allocator',{exact:true}).waitFor();
    assert.equal(applies,1);
    await page.getByLabel('Video VRAM Preload',{exact:true}).selectOption('manual');
    await page.getByText('Auto-tune off',{exact:true}).waitFor();
    assert.equal(auto,false);
    await page.getByRole('button',{name:'Clear comparisons',exact:true}).click();
    await page.getByText(/Local learning: 0 completed renders/).waitFor();
    assert.ok(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth));
    assert.deepEqual(errors,[]);
    console.log('PASS: mobile Auto-tune readout, learning reset, restart feedback, manual override ownership');
  } finally { await browser.close(); }
})().catch(error=>{console.error(error);process.exit(1);});
