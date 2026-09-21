// Isolated desktop/mobile library and Advanced integration; no live user data.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const root = path.resolve(__dirname, '../..');
const esbuild = require(path.join(root, 'ui/node_modules/esbuild'));
const {chromium} = require(process.env.MAESTRO_PLAYWRIGHT || 'playwright');
(async () => {
  const bundle = await esbuild.build({stdin: {contents: `
    import React from 'react'; import {createRoot} from 'react-dom/client';
    import {useStore} from './src/stores/useStore';
    import {Yue2Controls} from './src/components/Sidebar/Yue2Controls';
    import {AdvancedSettings} from './src/components/Sidebar/AdvancedSettings';
    window.store = useStore;
    useStore.setState(s => ({sidebarMode:'studio', generationMode:'audio', audioSubMode:'music',
      params:{...s.params, model_type:'yue2', model_mode:2, custom_settings:{}, alt_prompt:'Dry close-mic rap', activated_loras:[], seed:-1},
      modelOptions:{model_type:'yue2', architecture:'yue2', audio_only:true, loras_disabled:true, no_negative_prompt:true}}));
    createRoot(document.getElementById('root')).render(<div data-sidebar><Yue2Controls/><AdvancedSettings compact/></div>);`,
    resolveDir: path.join(root, 'ui'), loader: 'tsx'}, bundle: true, write: false, jsx: 'automatic'});
  const cssFile = fs.readdirSync(path.join(root, 'ui/dist/assets')).find(n => n.endsWith('.css'));
  const css = fs.readFileSync(path.join(root, 'ui/dist/assets', cssFile), 'utf8');
  const browser = await chromium.launch({headless:true, ...(process.platform === 'win32' ? {
    executablePath:process.env.MAESTRO_CHROME || 'C:/Program Files/Google/Chrome/Application/chrome.exe'} : {})});
  try {
    for (const width of [1100, 390]) {
      const styles = Array.from({length:30}, (_, i) => ({id:'lora-'+i, name:`Artist session ${i} · adapted 100 · step-200`,
        trigger:i === 7 ? 'Distinct rap voice' : 'Other voice '+i, archived:false, in_selector:false, created_at:i + 1, adapted_pair:{step:100}}));
      const page = await browser.newPage({viewport:{width,height:950}});
      const errors = [], updates = []; let failPatch = false;
      page.on('pageerror', e => errors.push(e.message));
      await page.route('**/*', async route => {
        const url = new URL(route.request().url()), request = route.request();
        if (url.pathname === '/') return route.fulfill({contentType:'text/html', body:`<html data-theme="golden-hour"><style>${css}body{background:#151515;color:#eee}#root{width:min(450px,100%);padding:14px}</style><div id="root"></div><script>${bundle.outputFiles[0].text}</script></html>`});
        let result = {presets:[], loras:[], styles:[], projects:[]};
        if (url.pathname === '/api/v1/music-styles') result = {styles:styles.filter(s => url.searchParams.has('include_archived') || !s.archived)};
        else if (url.pathname.startsWith('/api/v1/music-styles/') && request.method() === 'PATCH') {
          if (failPatch) return route.fulfill({status:500, contentType:'application/json',body:JSON.stringify({detail:'Temporary library error'})});
          const item = styles.find(s => s.id === url.pathname.split('/').pop());
          Object.assign(item, request.postDataJSON()); updates.push({...item}); result = item;
        } else assert.ok(!url.pathname.endsWith('/generate') && !url.pathname.endsWith('/train'), 'No GPU jobs during library management');
        await route.fulfill({contentType:'application/json',body:JSON.stringify(result)});
      });
      await page.goto('http://music-library-fixture/');
      assert.equal(await page.getByLabel('Saved music style').count(), 0, 'No giant style dropdown in the composer');
      await page.getByRole('button', {name:'Advanced settings',exact:true}).click();
      const panel = page.getByRole('dialog',{name:'Advanced settings',exact:true});
      const section = panel.getByTestId('advanced-loras');
      await section.locator('summary').click();
      const list = panel.getByRole('group',{name:'Available music LoRAs'});
      await list.getByText('Add your favorites here from My music → Saved LoRAs.').waitFor();
      await panel.getByRole('button',{name:'My music',exact:true}).click();
      const dialog = page.getByRole('dialog',{name:'My music',exact:true});
      await dialog.getByRole('searchbox',{name:'Search music LoRAs'}).fill('Distinct rap');
      await dialog.getByRole('article').first().waitFor();
      assert.equal(await dialog.getByRole('article').count(), 1);
      const shortlistSwitch = dialog.getByRole('switch');
      failPatch = true;
      await shortlistSwitch.click();
      await dialog.getByRole('alert').filter({hasText:'Temporary library error'}).waitFor();
      assert.equal(await shortlistSwitch.isChecked(),false,'Failed listing stays off');
      assert.equal(await page.evaluate(() => !!window.store.getState().params.custom_settings.artist_id),false);
      failPatch = false;
      await shortlistSwitch.click();
      await page.waitForFunction(() => document.querySelector('[role="switch"]')?.checked);
      assert.equal(updates.at(-1).in_selector,true);
      assert.equal(await page.evaluate(() => !!window.store.getState().params.custom_settings.artist_id),false,'Shortlisting does not activate the LoRA');
      assert.equal(await dialog.isVisible(),true,'Shortlisting keeps My music open');
      await dialog.getByRole('searchbox',{name:'Search music LoRAs'}).fill('Artist session 12');
      await shortlistSwitch.click();
      await page.waitForFunction(() => document.querySelector('[role="switch"]')?.checked);
      await dialog.getByRole('button',{name:'Close My music'}).click();
      await list.getByRole('checkbox',{name:styles[7].name,exact:true}).waitFor();
      assert.equal(await list.getByRole('checkbox').count(),2,'Only the two shortlisted LoRAs are offered');
      const active = list.getByRole('checkbox',{name:styles[7].name,exact:true});
      await active.check();
      await page.waitForFunction(() => window.store.getState().params.custom_settings.artist_id === 'lora-7');
      await panel.getByText('Trigger added automatically:',{exact:false}).waitFor();
      assert.equal(await page.evaluate(() => window.store.getState().params.alt_prompt), 'Dry close-mic rap', 'Selecting leaves the user style intact; backend adds the trigger');
      assert.equal(await page.getByLabel('Composition planning',{exact:true}).isDisabled(),true);
      assert.equal(await section.locator('summary').getByLabel('1 active',{exact:true}).count(),1,'Music adapter counts despite generic LoRAs being disabled');
      await panel.getByLabel('Music LoRA strength',{exact:true}).fill('1.25');
      assert.equal(await page.evaluate(() => window.store.getState().params.custom_settings.artist_strength),1.25);
      await panel.getByRole('searchbox',{name:'Search available music LoRAs'}).fill('Other voice 12');
      await list.getByRole('checkbox',{name:styles[12].name,exact:true}).check();
      assert.deepEqual(await page.evaluate(() => window.store.getState().params.custom_settings.artist_loras),
        [{id:'lora-7',strength:1.25},{id:'lora-12',strength:1}]);
      await panel.getByRole('searchbox',{name:'Search available music LoRAs'}).fill('');
      assert.equal(await active.isChecked(),true,'Adding a LoRA preserves the previous selection and strength');
      await panel.getByRole('group',{name:`Selected ${styles[12].name}`,exact:true}).getByLabel('Music LoRA strength',{exact:true}).fill('0.5');
      assert.equal(await section.locator('summary').getByLabel('2 active',{exact:true}).count(),1);
      await panel.getByText('Experimental mix.',{exact:true}).waitFor();
      assert.ok(await panel.evaluate(e => e.scrollWidth <= e.clientWidth+1),'Multiple sliders fit mobile width');
      fs.mkdirSync(path.join(root,'.codex-tmp/music-lora-ui'),{recursive:true});
      await page.screenshot({animations:'disabled',path:path.join(root,`.codex-tmp/music-lora-ui/multi-selector-${width}.png`)});
      await page.evaluate(() => window.store.getState().setMusicInstrumental(true));
      await panel.getByText('Instrumental LoRA · active at strength 1',{exact:true}).waitFor();
      assert.equal(await active.isDisabled(),true);
      assert.equal(await panel.getByLabel('Music LoRA strength',{exact:true}).first().isDisabled(),true);
      assert.equal(await section.locator('summary').getByLabel('1 active',{exact:true}).count(),1,
        'The built-in instrumental adapter counts; paused artist adapters do not');
      await page.screenshot({animations:'disabled',path:path.join(root,`.codex-tmp/music-lora-ui/instrumental-selector-${width}.png`)});
      await page.evaluate(() => window.store.getState().setMusicInstrumental(false));
      assert.equal(await active.isChecked(),true,'Instrumental preserves the artist selection');
      assert.equal(await active.isDisabled(),false);
      await panel.getByRole('button',{name:`Disable ${styles[7].name}`,exact:true}).click();
      assert.deepEqual(await page.evaluate(() => window.store.getState().params.custom_settings.artist_loras),[{id:'lora-12',strength:0.5}],
        'Disabling one LoRA keeps the other with its strength');
      await active.check();
      await list.getByRole('checkbox',{name:styles[12].name,exact:true}).uncheck();
      await panel.getByRole('button',{name:'My music',exact:true}).click();
      await dialog.getByRole('searchbox',{name:'Search music LoRAs'}).fill('Distinct rap');
      failPatch = true;
      await shortlistSwitch.click();
      await dialog.getByRole('alert').filter({hasText:'Temporary library error'}).waitFor();
      assert.equal(await shortlistSwitch.isChecked(),true,'Failed unlisting preserves saved state');
      assert.equal(await page.evaluate(() => window.store.getState().params.custom_settings.artist_id),'lora-7');
      failPatch = false;
      await shortlistSwitch.click();
      await page.waitForFunction(() => !window.store.getState().params.custom_settings.artist_id);
      assert.equal(updates.at(-1).in_selector,false);
      await shortlistSwitch.click();
      await page.waitForFunction(() => document.querySelector('[role="switch"]')?.checked);
      assert.equal(await page.evaluate(() => !!window.store.getState().params.custom_settings.artist_id),false);
      await dialog.getByRole('button',{name:'Close My music'}).click();
      await active.check();
      await panel.getByRole('button',{name:'My music',exact:true}).click();
      await dialog.getByRole('searchbox',{name:'Search music LoRAs'}).fill('Distinct rap');
      failPatch = true;
      await dialog.getByRole('button',{name:'Remove',exact:true}).click();
      await dialog.getByRole('alert').filter({hasText:'Temporary library error'}).waitFor();
      assert.equal(await page.evaluate(() => window.store.getState().params.custom_settings.artist_id),'lora-7','Failure must not deselect the adapter');
      failPatch = false;
      await dialog.getByRole('button',{name:'Remove',exact:true}).click();
      await dialog.getByRole('button',{name:'Undo removal'}).waitFor();
      await page.waitForFunction(() => !window.store.getState().params.custom_settings.artist_id);
      assert.equal(updates.at(-1).archived,true);
      assert.equal(await dialog.getByRole('article').count(),0);
      await dialog.getByRole('button',{name:'Undo removal'}).click();
      await dialog.getByRole('article').waitFor();
      assert.equal(updates.at(-1).archived,false);
      await dialog.getByRole('button',{name:'Remove',exact:true}).click();
      await dialog.getByRole('button',{name:'Undo removal'}).waitFor();
      await dialog.getByRole('checkbox',{name:'Show removed LoRAs'}).check();
      await dialog.getByRole('button',{name:'Restore',exact:true}).click();
      await shortlistSwitch.waitFor();
      assert.equal(updates.at(-1).archived,false);
      assert.equal(await shortlistSwitch.isChecked(),true,'Restore keeps the shortlist choice');
      assert.equal(await page.evaluate(() => !!window.store.getState().params.custom_settings.artist_id),false,'Restoring does not activate the LoRA');
      assert.ok(await dialog.evaluate(e => e.scrollWidth <= e.clientWidth+1),'Library fits phone width');
      fs.mkdirSync(path.join(root,'.codex-tmp/music-lora-ui'),{recursive:true});
      await page.screenshot({path:path.join(root,`.codex-tmp/music-lora-ui/library-${width}.png`)});
      await dialog.getByRole('button',{name:'Close My music'}).click();
      await active.check();
      assert.ok(await panel.evaluate(e => e.scrollWidth <= e.clientWidth+1),'Selector fits phone width');
      await page.screenshot({animations:'disabled',path:path.join(root,`.codex-tmp/music-lora-ui/selector-${width}.png`)});
      await active.uncheck();
      assert.equal(await page.getByLabel('Composition planning',{exact:true}).isDisabled(),false);
      assert.equal(await section.locator('summary').getByLabel('1 active',{exact:true}).count(),0);
      assert.equal(await list.getByRole('checkbox').count(),2,'Disabling does not remove the shortlist entry');
      // A legacy or restored job must not silently lose its selected adapter.
      await page.evaluate(() => {const {params} = window.store.getState(); window.store.setState({params:{...params,custom_settings:{artist_id:'lora-3',artist_strength:0.75}}});});
      const restored = list.getByRole('checkbox',{name:styles[3].name,exact:true});
      await restored.waitFor();
      assert.equal(await restored.isChecked(),true);
      assert.equal(await panel.getByLabel('Music LoRA strength',{exact:true}).inputValue(),'0.75');
      await restored.click();
      assert.equal(await restored.count(),0,'Unchecked unlisted restored adapter leaves the shortlist');
      await page.reload();
      await page.getByRole('button',{name:'Advanced settings',exact:true}).click();
      if (!await section.locator('summary').evaluate(el => el.parentElement.open)) await section.locator('summary').click();
      await active.waitFor();
      assert.equal(await list.getByRole('checkbox').count(),2,'Shortlist survives reopening independently of active params');
      assert.equal(await active.isChecked(),false);
      assert.deepEqual(errors,[]); await page.close();
    }
    console.log('Music LoRA library: desktop/mobile shortlist persistence, multiple selections and independent strengths, restored jobs, search, remove/restore, error recovery and badges passed.');
  } finally {await browser.close();}
})().catch(e => {console.error(e);process.exitCode=1;});
