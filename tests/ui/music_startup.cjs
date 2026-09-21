// Real store and music controls; every request is isolated from the running app.
const assert = require('node:assert/strict');
const path = require('node:path');
const root = path.resolve(__dirname, '../..');
const esbuild = require(path.join(root, 'ui/node_modules/esbuild'));
const {chromium} = require(process.env.MAESTRO_PLAYWRIGHT || 'playwright');
const music = 'yue2', other = 'minimax_music3', speech = 'kugelaudio_0_open';
const models = [music, other, speech].map(model_type => ({model_type, name:model_type,
  family:'tts', architecture:model_type, is_downloaded:true, audio_only:true}));

(async () => {
  const bundle = await esbuild.build({stdin:{contents:`
    import React from 'react'; import {createRoot} from 'react-dom/client';
    import {useStore} from './src/stores/useStore';
    import {MusicControls} from './src/components/Sidebar/MusicControls';
    import * as musicStyles from './src/lib/musicStyles';
    window.store = useStore;
    window.musicStyles = musicStyles;
    createRoot(document.getElementById('root')).render(<MusicControls/>);
    window.boot = useStore.getState().loadModels();`, resolveDir:path.join(root,'ui'), loader:'tsx'},
    bundle:true, write:false, jsx:'automatic', logLevel:'silent', define:{'process.env.NODE_ENV':'"development"'}});
  const browser = await chromium.launch({headless:true, ...(process.platform === 'win32' ? {
    executablePath:process.env.MAESTRO_CHROME || 'C:/Program Files/Google/Chrome/Application/chrome.exe',
  } : {})});
  try {
    for (const fresh of [false, true]) {
      let prefs = fresh ? {configured:false} : {configured:true, generation_mode:'audio', audio_sub_mode:'music',
        selected_model_per_mode:{audio:other}, selected_model_per_audio_sub_mode:{music:other}};
      let visibility = fresh ? {configured:false} : {configured:true, defaults_version:14,
        enabled_models:[other,speech], initialized_mature_models:[]};
      let failOptions = false, holdOptions = false, releaseOptions;
      const errors = [], pending = new Set(), submissions = [];
      const open = async () => {
        // A new browser context also proves that persistence is server-side.
        const context = await browser.newContext();
        const page = await context.newPage();
        page.on('pageerror', e => errors.push(e.message));
        await page.route('**/*', async route => {
          pending.add(route);
          const key = new URL(route.request().url()).pathname;
          const request = route.request();
          let result = {styles:[], projects:[], loras:[], presets:[], items:[], overrides:{}};
          try {
            if (key === '/') return await route.fulfill({contentType:'text/html', body:'<div id="root"></div>'});
            if (key === '/api/v1/models') result = {models, families:[{id:'tts', label:'Audio', order:1}]};
            if (key === '/api/v1/model-visibility') {
              if (request.method() === 'PUT') visibility = {configured:true, ...request.postDataJSON()};
              result = visibility;
            }
            if (key === '/api/v1/studio-preferences') {
              if (request.method() === 'PUT') prefs = {...prefs, configured:true, ...request.postDataJSON()};
              result = prefs;
            }
            if (key.includes('/model-options/')) {
              if (holdOptions) await new Promise(resolve => {releaseOptions = resolve;});
              if (failOptions) return await route.fulfill({status:503, body:'Temporarily unavailable'});
              const id = key.split('/').pop();
              result = {model_type:id, architecture:id, audio_only:true, yue2_composition:id === music,
                duration_slider:{min:1,max:600,default:120}, fps:1, guidance_max_phases:1};
            }
            if (key.includes('/defaults/')) result = {num_inference_steps:32, guidance_scale:1};
            if (key === '/api/v1/generate') {
              submissions.push(request.postDataJSON());
              result = {job_id:`music-${submissions.length}`, status:'held'};
            }
            if (key.startsWith('/api/v1/status/')) result = {status:'held', progress:0, output_files:[]};
            await route.fulfill({json:result});
          } finally {pending.delete(route);}
        });
        await page.goto('http://music-startup.test');
        await page.addScriptTag({content:bundle.outputFiles[0].text});
        await page.evaluate(() => window.boot);
        return {page, context};
      };
      const settle = async page => {
        await page.waitForFunction(() => !window.store.getState().modelOptionsLoading);
        await page.waitForTimeout(180); // serialized preference writes after store actions
        assert.equal(pending.size, 0);
      };
      let {page, context} = await open();
      await page.evaluate(() => window.store.getState().setDirectorVideoMaxShotFrames('minimax_h3_ref2va_fused_turbo', 345));
      if (fresh) await page.evaluate(() => {
        window.store.getState().setGenerationMode('audio');
        window.store.getState().setAudioSubMode('music');
      });
      await settle(page);
      assert.equal(await page.evaluate(() => window.store.getState().params.model_type), music);
      assert.ok(visibility.enabled_models.includes(music));
      assert.equal(prefs.music_defaults_version, 1);
      assert.equal(prefs.director_music_model, music);
      await page.getByLabel('Composition planning', {exact:true}).waitFor();
      const composition = page.getByLabel('Composition planning', {exact:true});
      assert.equal(await composition.inputValue(), '2', 'Direct generation is the initial selection');
      // Exercise the actual queue payload, not just the visible select value.
      await page.evaluate(() => window.store.getState().setParams({prompt:'[Verse]\nMorning light', alt_prompt:'Acoustic pop'}));
      for (const mode of [2, 0, 1]) {
        await composition.selectOption(String(mode));
        await page.evaluate(() => window.store.getState().startGeneration('queue'));
        assert.equal(submissions.at(-1)?.model_mode, mode, 'The submitted composition mode matches the visible choice');
      }
      await composition.selectOption('2');
      await page.evaluate(() => {
        window.musicStyles.toggleMusicStyle({id:'rapper',name:'Rapper',trigger:'Rap voice'},true);
        window.musicStyles.toggleMusicStyle({id:'singer',name:'Singer',trigger:'Pop voice'},true);
        window.musicStyles.setMusicStyleStrength('rapper',0.75);
        window.musicStyles.setMusicStyleStrength('singer',1.25);
      });
      assert.equal(await composition.isDisabled(),true);
      await page.evaluate(() => window.store.getState().startGeneration('queue'));
      const selections = [{id:'rapper',strength:0.75},{id:'singer',strength:1.25}];
      assert.deepEqual(submissions.at(-1).custom_settings.artist_loras,selections,'Both selections reach the actual queued request');
      const saved = JSON.parse(JSON.stringify(submissions.at(-1)));
      await page.evaluate(async params => {
        window.musicStyles.selectMusicStyle();
        window.store.setState({selectedOutputMeta:{source:'sidecar',params},selectedOutput:-1});
        await window.store.getState().loadSettingsFromOutput();
      }, saved);
      await settle(page);
      assert.deepEqual(await page.evaluate(() => window.store.getState().params.custom_settings.artist_loras),selections,
        'Loading a saved song restores every LoRA and its exact strength');
      await page.evaluate(() => window.store.getState().startGeneration('queue'));
      assert.deepEqual(submissions.at(-1).custom_settings.artist_loras,selections,'Restored settings survive resubmission');
      await page.evaluate(async params => {
        window.store.setState({selectedOutputMeta:{source:'sidecar',params:{...params,custom_settings:{artist_id:'legacy',artist_strength:0.6}}}});
        await window.store.getState().loadSettingsFromOutput();
      }, saved);
      await settle(page);
      assert.deepEqual(await page.evaluate(() => window.musicStyles.selectedMusicStyles(window.store.getState().params.custom_settings)),
        [{id:'legacy',strength:0.6}],'Old single-LoRA outputs still restore correctly');
      await page.getByRole('checkbox', {name:'Instrumental', exact:true}).check();
      assert.equal(await composition.inputValue(),'0','Instrumental visibly uses full score planning');
      assert.equal(await composition.isDisabled(),true);
      await page.evaluate(() => window.store.getState().startGeneration('queue'));
      const instrumental = JSON.parse(JSON.stringify(submissions.at(-1)));
      assert.equal(instrumental.model_mode,0);
      assert.equal(instrumental.custom_settings.instrumental,true);
      assert.equal(instrumental.custom_settings.artist_id,'legacy','Artist choices remain saved while paused');
      assert.equal(instrumental.custom_settings.abc,'');
      assert.equal(instrumental.audio_prompt_type,'');
      await page.getByRole('checkbox', {name:'Instrumental', exact:true}).uncheck();
      assert.equal(await composition.inputValue(),'2','Turning Instrumental off restores the vocal planning selection');
      await page.evaluate(async params => {
        window.store.setState({selectedOutputMeta:{source:'sidecar',params:{...params,_music_instrumental:undefined,prompt:'[intro]\n[outro]'}}});
        await window.store.getState().loadSettingsFromOutput();
      },instrumental);
      await settle(page);
      assert.equal(await page.getByRole('checkbox', {name:'Instrumental',exact:true}).isChecked(),true,
        'An instrumental with a section plan restores from its runtime flag');
      await page.getByRole('checkbox', {name:'Instrumental',exact:true}).uncheck();
      assert.equal(await composition.inputValue(),'2','Restored artist LoRAs resume in their required direct mode');
      await page.getByRole('checkbox', {name:'Instrumental',exact:true}).check();
      await page.evaluate(() => window.musicStyles.selectMusicStyle({id:'solo',name:'Solo',trigger:'Solo'}));
      assert.equal(await page.getByRole('checkbox', {name:'Instrumental',exact:true}).isChecked(),false,
        'Choosing a training audition switches back to vocal generation');
      await page.evaluate(() => window.musicStyles.selectMusicStyle());
      await page.getByRole('button',{name:'My music',exact:true}).click();
      await page.getByRole('dialog',{name:'My music',exact:true}).waitFor();
      await page.evaluate(id => {
        window.store.getState().selectModel(id);
        window.store.getState().setDirectorMusicModel(id);
      }, other);
      await settle(page);
      assert.equal(await page.getByLabel('Composition planning', {exact:true}).count(), 0);
      assert.equal(prefs.selected_model_per_audio_sub_mode.music, other);
      await page.evaluate(id => window.store.getState().toggleModelEnabled(id), music);
      await settle(page);
      await context.close();

      ({page, context} = await open());
      await settle(page);
      assert.equal(await page.evaluate(() => window.store.getState().directorVideoMaxShotFramesByModel.minimax_h3_ref2va_fused_turbo), 345,
        'Manual GPU clip limit survives a fresh browser using durable preferences');
      assert.equal(await page.evaluate(() => window.store.getState().params.model_type), other, 'Later choice survives a new browser and app restart');
      assert.equal(await page.evaluate(() => window.store.getState().directorMusicModel), other);
      assert.equal(await page.evaluate(id => window.store.getState().enabledModels.has(id), music), false, 'A deliberate disable is not undone');
      await page.evaluate(id => {
        window.store.getState().toggleModelEnabled(id);
        window.store.getState().selectModel(id);
      }, music);
      await settle(page);
      await context.close();

      holdOptions = true;
      ({page, context} = await open());
      await page.getByLabel('Composition planning', {exact:true}).waitFor();
      assert.equal(await page.getByLabel('Composition planning', {exact:true}).inputValue(), '2',
        'Default composition does not wait for the model-options response');
      assert.equal(await page.getByRole('button',{name:'My music',exact:true}).count(), 1, 'Controls show while capabilities are still loading');
      failOptions = true; holdOptions = false; releaseOptions();
      await settle(page);
      assert.equal(await page.getByLabel('Composition planning', {exact:true}).count(), 1, 'A failed options request cannot hide YuE2 controls');
      assert.equal(await page.evaluate(() => window.store.getState().modelOptions), null);
      // Stale capabilities from YuE2 must not keep its controls on another model.
      await page.evaluate(({music,other}) => window.store.setState(s => ({
        params:{...s.params,model_type:other}, modelOptions:{model_type:music,yue2_composition:true},
      })), {music,other});
      assert.equal(await page.getByLabel('Composition planning', {exact:true}).count(), 0);
      await context.close();
      assert.deepEqual(errors, []);
    }
    console.log('Music startup: fresh/update defaults, durable choices, delayed/failed options and stale capabilities passed.');
  } finally {await browser.close();}
})().catch(e => {console.error(e); process.exitCode = 1;});
