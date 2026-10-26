// Real duration controls/store, mocked API only: no generation or live requests.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const root = path.resolve(__dirname, '../..');
const esbuild = require(path.join(root, 'ui/node_modules/esbuild'));
const playwright = require(process.env.MAESTRO_PLAYWRIGHT || 'playwright');
const origin = 'http://fish-duration.test';
const output = path.join(root,'.codex-tmp/fish-auto-duration-20261023');
fs.mkdirSync(output,{recursive:true});
const options = {model_type:'fish_drama3_int4',architecture:'fish_drama3',audio_only:true,
  speech_auto_duration:true,duration_is_maximum:true,max_voice_count:3,
  audio_mode_from_voice_count:true,guidance_max_phases:0,
  duration_slider:{min:0,max:300,default:0,increment:1}};
const defaults = JSON.parse(fs.readFileSync(path.join(root,'app/defaults/fish_drama3_int4.json'),'utf8'));

(async () => {
  const bundle = await esbuild.build({stdin:{resolveDir:path.join(root,'ui'),loader:'tsx',contents:`
    import React from 'react'; import {createRoot} from 'react-dom/client';
    import {useStore} from './src/stores/useStore';
    import {OutputFormatControls} from './src/components/Sidebar/OutputFormatControls';
    window.store=useStore;
    const options=${JSON.stringify(options)}, defaults=${JSON.stringify(defaults)};
    useStore.setState(s=>({generationMode:'audio',audioSubMode:'speech',durationSeconds:0,
      params:{...s.params,...defaults,model:undefined,model_type:'fish_drama3_int4',
        prompt:'[calm] Hello. [whispering] Is anyone there?',activated_loras:[],loras_multipliers:''},
      models:[{model_type:'fish_drama3_int4',architecture:'fish_drama3',family:'tts',is_downloaded:true}],
      modelsLoaded:true,modelsLoading:false,modelOptions:options,enabledModels:new Set(['fish_drama3_int4']),
      selectedModelPerMode:{audio:'fish_drama3_int4'},ttsVoiceCount:0,ttsVoices:[],
      activeWorkspace:'test',jobs:[],loadLoras:async()=>useStore.setState({availableLoras:[]})}));
    createRoot(document.getElementById('root')).render(<div style={{padding:20}}><OutputFormatControls/></div>);
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
      let savedDuration=0, legacy=false;
      page.on('pageerror',error=>errors.push(error.stack||error.message));
      await page.route('**/*',route=>{
        const request=route.request(),url=new URL(request.url());
        assert.equal(url.origin,origin,'no live app or internet requests');
        if(url.pathname==='/') return route.fulfill({contentType:'text/html',body:'<meta name="viewport" content="width=device-width,initial-scale=1"><div id="root"></div>'});
        if(url.pathname==='/api/v1/generate') {
          submissions.push(request.postDataJSON());
          return route.fulfill({json:{job_id:'mock-duration-'+submissions.length,status:'held'}});
        }
        if(url.pathname.startsWith('/api/v1/model-options/')) return route.fulfill({json:legacy
          ? {...options,speech_auto_duration:false,duration_slider:{min:1,max:300,default:60,increment:1}}
          : options});
        if(url.pathname.startsWith('/api/v1/defaults/')) return route.fulfill({json:{...defaults,duration_seconds:savedDuration}});
        return route.fulfill({json:{}});
      });
      await page.goto(origin);
      await page.addStyleTag({content:css});
      await page.addScriptTag({content:bundle.outputFiles[0].text});
      const submit=async(expected)=>{
        const before=submissions.length;
        await page.evaluate(()=>window.store.getState().startGeneration('queue'));
        assert.equal(submissions.length,before+1,'one mocked request per submission');
        assert.equal(submissions.at(-1).duration_seconds,expected,'exact seconds reach API');
      };
      await page.getByRole('button',{name:'Duration: Auto',exact:true}).click();
      const dialog=page.getByRole('dialog',{name:'Speech length'});
      const auto=dialog.getByRole('button',{name:'Auto',exact:true});
      const manual=dialog.getByRole('button',{name:'Maximum length',exact:true});
      assert.equal(await auto.getAttribute('aria-pressed'),'true');
      assert.equal(await dialog.getByRole('textbox').count(),0,'Auto needs no duration entry');
      await submit(0);
      await page.screenshot({path:path.join(output,`auto-${width}.png`)});
      await manual.click();
      const input=dialog.getByRole('textbox',{name:'Maximum speech duration timecode'});
      await input.fill('6'); await input.press('Enter');
      assert.equal(await page.evaluate(()=>window.store.getState().durationSeconds),6);
      assert.equal(await page.evaluate(()=>window.store.getState().params.duration_seconds),6);
      await submit(6);
      await auto.click(); await submit(0);
      await manual.click(); await submit(6);
      assert.equal(await page.getByRole('button',{name:'Duration: Max 6s',exact:true}).count(),1);
      await page.screenshot({path:path.join(output,`manual-${width}.png`)});
      await dialog.getByRole('button',{name:'Close Speech length',exact:true}).click();

      // Restore both Auto and manual limits from actual sidecar settings.
      for(const duration of [0,6]) {
        await page.evaluate(async value=>{
          const state=window.store.getState();
          window.store.setState({selectedOutputMeta:{params:{...state.params,duration_seconds:value}},selectedOutput:-1});
          await window.store.getState().loadSettingsFromOutput();
        },duration);
        assert.equal(await page.evaluate(()=>window.store.getState().durationSeconds),duration,'saved zero/positive duration restored');
        await submit(duration);
      }

      // Fetch actual per-model defaults on selection, rather than inheriting
      // another model's maximum. Legacy saved values remain manual.
      for(const duration of [0,30]) {
        savedDuration=duration;
        await page.evaluate(()=>{
          window.store.setState(s=>({params:{...s.params,duration_seconds:120},durationSeconds:120}));
          window.store.getState().selectModel('fish_drama3_int4');
        });
        await page.waitForFunction(value=>window.store.getState().durationSeconds===value
          &&window.store.getState().params.duration_seconds===value,duration);
        await submit(duration);
      }

      // A newly built UI connected to an older running backend offers no Auto.
      legacy=true;
      await page.evaluate(()=>window.store.getState().loadModelOptions('fish_drama3_int4'));
      await page.getByRole('button',{name:'Duration: Max 30s',exact:true}).click();
      assert.equal(await dialog.getByRole('button',{name:'Auto',exact:true}).count(),0);
      await page.evaluate(()=>window.store.getState().setDurationSeconds(6));
      await submit(6);
      await dialog.getByRole('button',{name:'Close Speech length',exact:true}).click();
      await page.evaluate(()=>window.store.getState().setDurationSeconds(0));
      assert.equal(await page.evaluate(()=>window.store.getState().durationSeconds),1,'Auto zero not sent to old backend');
      await submit(1);

      // Other audio generators keep their existing duration policy.
      await page.evaluate(()=>window.store.setState(s=>({durationSeconds:6,
        params:{...s.params,model_type:'other_tts'},enabledModels:new Set(['other_tts']),
        models:[{model_type:'other_tts',architecture:'other_tts',family:'tts',is_downloaded:true}],
        modelOptions:{model_type:'other_tts',architecture:'other_tts',audio_only:true,
          duration_slider:{min:1,max:300,default:60,increment:1}}})));
      await submit(60);
      assert.equal(await page.getByRole('button',{name:'Duration: Auto · 6s',exact:true}).count(),1);
      assert.deepEqual(errors,[]);
      await page.close();
      console.log('Fish duration '+width+'px: Auto, exact manual6, toggle, saved settings, per-model defaults, legacy backend and other-model scoping passed');
    }
  } finally { await browser.close(); }
})().catch(error=>{console.error(error);process.exitCode=1;});
