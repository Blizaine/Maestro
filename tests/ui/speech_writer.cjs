// Real speech writer controls/store. Every API request is mocked.
const assert=require('node:assert/strict');
const fs=require('node:fs');
const path=require('node:path');
const root=path.resolve(__dirname,'../..');
const esbuild=require(path.join(root,'ui/node_modules/esbuild'));
const {chromium}=require(process.env.MAESTRO_PLAYWRIGHT||'playwright');
const origin='http://speech-writer.test';
const output=path.join(root,'.codex-tmp/fish-longform-20261023');
fs.mkdirSync(output,{recursive:true});

(async()=>{
  const bundle=await esbuild.build({stdin:{resolveDir:path.join(root,'ui'),loader:'tsx',contents:`
    import React from 'react';import {createRoot} from 'react-dom/client';
    import {useStore} from './src/stores/useStore';
    import {SpeechWriterSettings} from './src/components/Sidebar/SpeechWriterSettings';
    import {OutputFormatControls} from './src/components/Sidebar/OutputFormatControls';
    import {PromptInput} from './src/components/Sidebar/PromptInput';
    window.store=useStore;
    const options={model_type:'fish_drama3_int4',architecture:'fish_drama3',audio_only:true,
      max_voice_count:3,audio_mode_from_voice_count:true,speech_auto_duration:true,speech_longform:true,speech_writer_controls:true,
      duration_slider:{min:0,max:3600,default:0,increment:1}};
    useStore.setState(s=>({generationMode:'audio',audioSubMode:'speech',durationSeconds:0,
      params:{...s.params,model_type:'fish_drama3_int4',prompt:'Write a ten-minute story about a lighthouse.',
        activated_loras:[],loras_multipliers:''},modelOptions:options,
      models:[{model_type:'fish_drama3_int4',architecture:'fish_drama3',family:'tts',is_downloaded:true}],
      enabledModels:new Set(['fish_drama3_int4']),ttsVoices:[],ttsVoiceCount:0,
      activeWorkspace:'test',jobs:[],loadLoras:async()=>{},
      loadModelOptions:async()=>useStore.setState({modelOptions:options})}));
    createRoot(document.getElementById('root')).render(<div style={{padding:20,maxWidth:440}}><SpeechWriterSettings/><PromptInput/><OutputFormatControls/></div>);
  `},bundle:true,write:false,jsx:'automatic',logLevel:'silent',define:{'process.env.NODE_ENV':'"development"'}});
  const browser=await chromium.launch({headless:true,...(process.platform==='win32'?{executablePath:process.env.MAESTRO_CHROME||'C:/Program Files/Google/Chrome/Application/chrome.exe'}:{})});
  try{
    for(const width of [390,1280]){
      const page=await browser.newPage({viewport:{width,height:900}});
      const errors=[],drafts=[],jobs=[];
      let streamReads=0;
      page.on('pageerror',error=>errors.push(error.message));
      await page.route('**/*',route=>{
        const request=route.request(),url=new URL(request.url());
        assert.equal(url.origin,origin,'no live or internet requests');
        if(url.pathname==='/')return route.fulfill({contentType:'text/html',body:'<meta name="viewport" content="width=device-width,initial-scale=1"><div id="root"></div>'});
        if(url.pathname==='/api/v1/llm/enhance-prompt'){
          drafts.push(request.postDataJSON());
          return route.fulfill({json:{enhanced:'Speaker 1: The lighthouse keeper watched the sea.',warnings:[]}});
        }
        if(url.pathname==='/api/v1/generate'){
          jobs.push(request.postDataJSON());return route.fulfill({json:{job_id:'mock-speech',status:'held'}});
        }
        if(url.pathname==='/api/v1/llm/status')return route.fulfill({json:{loaded:true}});
        if(url.pathname==='/api/v1/llm/stream-status'){
          streamReads++;return route.fulfill({json:{text:streamReads===1?'Private plan':'Next script section',done:true}});
        }
        return route.fulfill({json:{}});
      });
      const boot=async()=>{
        const index=fs.readFileSync(path.join(root,'ui/dist/index.html'),'utf8');
        const css=index.match(/href="(\/assets\/[^" ]+\.css)"/);
        assert.ok(css,'built UI stylesheet');
        await page.addStyleTag({content:fs.readFileSync(path.join(root,'ui/dist',css[1]),'utf8')});
        await page.addScriptTag({content:bundle.outputFiles[0].text});
      };
      await page.goto(origin);await boot();
      await page.evaluate(()=>window.store.setState({isEnhancing:true}));
      await page.waitForResponse(response=>response.url().endsWith('/api/v1/llm/stream-status')&&streamReads>=3);
      assert.equal(await page.getByText('Writing...',{exact:true}).count(),1,'progress continues between section calls');
      await page.evaluate(()=>window.store.setState({isEnhancing:false}));
      const instructions=page.getByRole('textbox',{name:'Speech writing instructions'});
      const target=page.getByRole('spinbutton',{name:'Speech draft length in minutes'});
      await instructions.fill('Warm, wry narrator; concise dialogue with distinct personalities.');
      await target.fill('12');
      for(const mode of ['monologue','dialogue_fast']){
        await page.evaluate(mode=>window.store.getState().enhancePrompt(mode),mode);
        assert.equal(drafts.at(-1).tts_target_minutes,12);
        assert.equal(drafts.at(-1).duration_seconds,0,'Auto duration reaches the writer without a guessed target');
        assert.equal(drafts.at(-1).tts_writer_instructions,'Warm, wry narrator; concise dialogue with distinct personalities.');
        assert.ok(!drafts.at(-1).prompt.includes('Warm, wry'),'writing instructions are separate from spoken brief');
      }
      const saved=await page.evaluate(()=>JSON.parse(localStorage.getItem('maestro_speech_writer')));
      assert.equal(saved._speech_writer_target_minutes,12);
      await page.reload();await boot();
      assert.equal(await instructions.inputValue(),saved._speech_writer_instructions,'preferences survive refresh');
      assert.equal(await target.inputValue(),'12');

      await page.getByRole('button',{name:'Duration: Auto',exact:true}).click();
      const dialog=page.getByRole('dialog',{name:'Speech length'});
      await dialog.getByRole('button',{name:'Maximum length',exact:true}).click();
      await dialog.getByRole('button',{name:'20m',exact:true}).click();
      await page.evaluate(()=>window.store.getState().enhancePrompt('monologue'));
      assert.equal(drafts.at(-1).duration_seconds,1200,'selected clip duration reaches Write');
      assert.equal(drafts.at(-1).tts_target_minutes,12,'explicit writer override remains independent');
      await page.evaluate(()=>window.store.getState().startGeneration('queue'));
      assert.equal(jobs.at(-1).duration_seconds,1200,'long manual maximum reaches generation');
      assert.equal(jobs.at(-1)._speech_writer_target_minutes,12,'writer preferences saved with output settings');
      await dialog.getByRole('button',{name:'Close Speech length',exact:true}).click();

      await target.fill('0');
      await page.evaluate(()=>{
        window.store.getState().setParam('prompt','Tell a story about a lighthouse.');
        // A restored/stale parameter must not replace the currently visible20m.
        window.store.setState(s=>({params:{...s.params,duration_seconds:0}}));
      });
      await page.evaluate(()=>window.store.getState().enhancePrompt('monologue'));
      assert.equal(drafts.at(-1).duration_seconds,1200,'Write follows the visible duration');
      assert.equal(drafts.at(-1).tts_target_minutes,0,'zero keeps clip duration as the default');
      await page.evaluate(()=>window.store.getState().setDurationSeconds(0));
      await page.evaluate(()=>window.store.getState().enhancePrompt('monologue'));
      assert.equal(drafts.at(-1).duration_seconds,0,'Auto follows the brief again');
      assert.equal(drafts.at(-1).tts_target_minutes,0);

      await page.evaluate(async()=>{
        const s=window.store.getState();
        window.store.setState({selectedOutputMeta:{params:{...s.params,_speech_writer_instructions:'Quiet documentary narrator',
          _speech_writer_target_minutes:10,duration_seconds:0}},selectedOutput:-1});
        await window.store.getState().loadSettingsFromOutput();
      });
      assert.equal(await instructions.inputValue(),'Quiet documentary narrator');
      assert.equal(await target.inputValue(),'10');
      await page.screenshot({path:path.join(output,'writer-'+width+'.png')});
      await page.evaluate(()=>window.store.setState({audioSubMode:'music'}));
      assert.equal(await instructions.count(),0,'controls hidden outside speech');
      await page.evaluate(()=>window.store.getState().enhancePrompt());
      assert.ok(!('tts_writer_instructions' in drafts.at(-1)));
      assert.ok(!('tts_target_minutes' in drafts.at(-1)));
      assert.ok(!('duration_seconds' in drafts.at(-1)),'speech duration does not leak into music writing');
      await page.evaluate(()=>window.store.getState().startGeneration('queue'));
      assert.ok(!('_speech_writer_instructions' in jobs.at(-1)),'speech preferences excluded from music output');
      assert.deepEqual(errors,[]);
      await page.evaluate(()=>window.store.setState(s=>({audioSubMode:'speech',modelOptions:{...s.modelOptions,speech_writer_controls:false}})));
      assert.equal(await instructions.count(),0,'old backend cannot silently ignore visible controls');
      await page.close();
      console.log('Speech writer '+width+'px: clip-duration default, Auto/override, saved style, refresh/sidecar restoration and music isolation passed');
    }
  }finally{await browser.close();}
})().catch(error=>{console.error(error);process.exitCode=1;});
