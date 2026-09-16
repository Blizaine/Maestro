const assert = require('node:assert/strict');
const path = require('node:path');
const root = path.resolve(__dirname, '../..');
const esbuild = require(path.join(root, 'ui/node_modules/esbuild'));
const {chromium} = require(process.env.MAESTRO_PLAYWRIGHT || 'playwright');
const model = 'minimax_h3_ref2va_fused_turbo';
const options = {model_type:model, architecture:'minimax_h3_ref2va', omni_reference:true,
  frames_minimum:124, frames_maximum:345, frames_steps:17, fps:24};
(async () => {
  const bundle = await esbuild.build({stdin:{contents:`
    import React from 'react'; import {createRoot} from 'react-dom/client';
    import {useStore} from './src/stores/useStore';
    import {DirectorGpuClipLimit} from './src/components/Sidebar/DirectorGpuClipLimit';
    import {DirectorMusicClipLength} from './src/components/Sidebar/DirectorMusicClipLength';
    window.store = useStore;
    useStore.setState({selectedModelPerMode:{video:${JSON.stringify(model)}}, directorSeamless:false});
    createRoot(document.getElementById('root')).render(<><DirectorGpuClipLimit model=${JSON.stringify(model)}
      options={${JSON.stringify(options)}}/><DirectorMusicClipLength/></>);`,
    resolveDir:path.join(root,'ui'), loader:'tsx'}, bundle:true, write:false, jsx:'automatic', logLevel:'silent'});
  const browser = await chromium.launch({headless:true, ...(process.platform === 'win32' ? {
    executablePath:process.env.MAESTRO_CHROME || 'C:/Program Files/Google/Chrome/Application/chrome.exe',
  } : {})});
  try {
    const page = await browser.newPage({viewport:{width:390,height:844}});
    const errors = [], writes = [];
    page.on('pageerror', error => errors.push(error.message));
    await page.route('**/*', async route => {
      const url = new URL(route.request().url());
      if (url.pathname === '/') return route.fulfill({contentType:'text/html', body:'<div id="root"></div>'});
      let body = {};
      if (url.pathname.includes('model-options')) body = options;
      if (url.pathname.includes('music-clip-limits')) {
        const frames = route.request().postDataJSON().director_max_shot_frames || 243;
        body = {fps:24, frames_minimum:124, frame_step:17, hard_max_frames:345,
          recommended_frames:243, recommended_seconds:243/24, max_frames:frames, max_seconds:frames/24, auto:true};
      }
      if (route.request().method() === 'PUT') writes.push(route.request().postDataJSON());
      return route.fulfill({contentType:'application/json', body:JSON.stringify(body)});
    });
    await page.goto('http://fixture.local/');
    await page.addScriptTag({content:bundle.outputFiles[0].text});
    const group = page.getByRole('group', {name:'GPU clip limit',exact:true});
    const select = group.getByRole('combobox');
    await page.waitForFunction(() => document.querySelector('#director-gpu-clip-limit').textContent.includes('10.1'));
    assert.match(await select.textContent(), /14.4s/);
    await select.selectOption('345');
    await page.waitForFunction(() => document.body.textContent.includes('Above Auto'));
    const music = page.getByRole('group', {name:'Music video clip length'});
    await page.waitForFunction(() => document.querySelector('input[type=range]')?.value === '345');
    assert.equal(await music.getByRole('button', {name:'Auto',exact:true}).getAttribute('aria-pressed'), 'true');
    await page.waitForFunction(() => window.store.getState().directorVideoMaxShotFramesByModel.minimax_h3_ref2va_fused_turbo === 345);
    await select.selectOption('');
    await page.waitForFunction(() => document.querySelector('input[type=range]')?.value === '243');
    assert.equal(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth), true);
    assert.ok(writes.some(w => w.director_max_shot_frames_per_model?.[model] === 345));
    assert.deepEqual(errors, []);
    console.log('Director GPU cap, 14.4s selection, music Auto, persistence, and mobile width: passed');
  } finally {await browser.close();}
})().catch(error => {console.error(error); process.exitCode=1;});
