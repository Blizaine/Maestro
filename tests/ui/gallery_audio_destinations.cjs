// Focused mounted-sidecar coverage for audio gallery destinations.
// All API requests are intercepted; no live uploads or generations occur.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const root = path.resolve(__dirname, '../..');
const esbuild = require(path.join(root, 'ui/node_modules/esbuild'));
const {chromium} = require(process.env.MAESTRO_PLAYWRIGHT || 'playwright');

(async () => {
  const bundle = await esbuild.build({stdin:{resolveDir:path.join(root,'ui'),loader:'tsx',contents:`
    import React from 'react'; import {createRoot} from 'react-dom/client';
    import {useStore} from './src/stores/useStore';
    import {useGalleryInputs,sendToGalleryInput} from './src/lib/galleryInputs';
    import {VoiceRefSection} from './src/components/Sidebar/VoiceRefSection';
    import {AudioModeSection} from './src/components/Sidebar/AudioModeSection';
    import {OmniReferenceSection} from './src/components/Sidebar/OmniReferenceSection';
    import {MixerControls} from './src/components/Sidebar/MixerControls';
    import {Yue2Controls} from './src/components/Sidebar/Yue2Controls';
    const components={voice:VoiceRefSection,audioMode:AudioModeSection,omni:OmniReferenceSection,mixer:MixerControls,yue2:Yue2Controls};
    const initial=useStore.getState(); let rootNode=null;
    window.store=useStore; window.galleryInputs=useGalleryInputs;
    window.mount=(name,patch={})=>{
      if(rootNode) rootNode.unmount();
      useGalleryInputs.setState({targets:[],receiving:false});
      const state={...initial,...patch,params:{...initial.params,...(patch.params||{})}};
      useStore.setState(state);
      rootNode=createRoot(document.getElementById('root'));
      rootNode.render(React.createElement(components[name]));
    };
    window.send=async(label,file)=>{
      const target=useGalleryInputs.getState().targets.find(item=>item.label===label);
      if(!target) throw new Error('Missing target: '+label);
      await sendToGalleryInput(target.id,file);
    };
    window.targets=()=>useGalleryInputs.getState().targets.map(({kind,label,disabledReason})=>({kind,label,disabledReason}));
    window.mount('voice');
    // The browser's audio-duration probe is irrelevant to routing; resolve it
    // deterministically so MixerControls still runs its real upload handler.
    window.Audio=class extends EventTarget {duration=1;set src(_value){queueMicrotask(()=>this.dispatchEvent(new Event('loadedmetadata')))}};
  `},bundle:true,write:false,jsx:'automatic',define:{'process.env.NODE_ENV':'"development"'},logLevel:'silent'});
  const assets=path.join(root,'ui/dist/assets');
  const css=fs.readFileSync(path.join(assets,fs.readdirSync(assets).find(name=>name.endsWith('.css'))),'utf8');
  const browser=await chromium.launch({headless:true,...(process.platform==='win32'?{executablePath:process.env.MAESTRO_CHROME||'C:/Program Files/Google/Chrome/Application/chrome.exe'}:{})});
  try {
    const page=await browser.newPage({viewport:{width:700,height:900}});
    const errors=[],uploads=[];
    page.on('pageerror',error=>errors.push(error.message));
    await page.route('**/*',async route=>{
      const request=route.request(), url=new URL(request.url()), key=url.pathname;
      const json=value=>route.fulfill({json:value});
      if(key==='/') return route.fulfill({contentType:'text/html',body:'<meta name="viewport" content="width=device-width,initial-scale=1"><div id="root"></div>'});
      if(key==='/api/v1/upload-audio'||key==='/api/v1/upload') {
        const body=request.postDataBuffer()?.toString('latin1')||'';
        const name=/filename="([^"]+)"/.exec(body)?.[1]||'fixture.wav';uploads.push({endpoint:key,name});
        return json({filename:name,path:'/uploads/'+name,url:'/api/v1/file/'+name,duration_seconds:1,has_audio:true});
      }
      if(key==='/api/v1/characters') return json({characters:[]});
      if(key.includes('/model-options/')) return json({});
      return json({});
    });
    await page.goto('http://gallery-audio-destinations.test');
    await page.addStyleTag({content:css});
    await page.addScriptTag({content:bundle.outputFiles[0].text});
    const mount=(name,patch)=>page.evaluate(args=>window.mount(args.name,args.patch),{name,patch});
    const targetLabels=()=>page.evaluate(()=>window.targets().map(target=>target.label));
    const send=label=>page.evaluate(async label=>window.send(label,new File(['audio fixture'],label+'.wav',{type:'audio/wav'})),label);
    const hasTarget=async label=>{
      await page.waitForFunction(label=>window.targets().some(target=>target.label===label),label);
      assert.ok((await targetLabels()).includes(label),`Expected active target: ${label}`);
    };

    const ltx={model_type:'ltx-fixture',family:'ltx2',architecture:'ltx2'};
    await mount('voice',{models:[ltx],params:{model_type:ltx.model_type},studioVideoWorkflow:'frames',servicesConfig:{voice_reference_enabled:true},directorVoiceRef:null});
    await hasTarget('LTX voice reference');
    await send('LTX voice reference');
    await page.waitForFunction(()=>window.store.getState().directorVoiceRef?.name==='LTX voice reference.wav');
    assert.ok(!(await targetLabels()).includes('LTX voice reference'),'filled voice slot does not stay registered as an add destination');

    const audioOptions={model_type:'fixture_tts',architecture:'generic_tts',audio_only:true,any_audio_prompt:true,max_voice_count:2,
      audio_prompt_type_sources:{selection:['A','AB'],default:'AB'}};
    await mount('audioMode',{modelOptions:audioOptions,modelOptionsLoading:false,generationMode:'audio',ttsVoiceCount:2,
      ttsVoices:[{name:'Voice 1',filename:null,path:null},{name:'Voice 2',filename:null,path:null}],params:{model_type:'fixture_tts',audio_prompt_type:'AB'}});
    await hasTarget('voice 1 reference');await hasTarget('voice 2 reference');
    await send('voice 2 reference');
    await page.waitForFunction(()=>window.store.getState().params.audio_guide2==='/uploads/voice 2 reference.wav');
    assert.equal(uploads.at(-1).endpoint,'/api/v1/upload-audio','voice reference uses the existing audio upload handler');

    const videoReference={id:'video-ref',type:'video',path:'/uploads/source.mp4',filename:'source.mp4',url:'/api/v1/file/source.mp4',role:'',has_audio:false};
    const omniOptions={omni_reference_limits:{image:3,video:2,audio:1,total:4}};
    await mount('omni',{modelOptions:omniOptions,params:{minimax_h3_references:[videoReference]}});
    await hasTarget('reference audio');
    await page.getByRole('button',{name:'Edit Video 1 reference'}).click();
    await hasTarget('Audio for Video 1');
    await send('Audio for Video 1');
    await page.waitForFunction(()=>window.store.getState().params.minimax_h3_references[0]?.audio_path==='/uploads/Audio for Video 1.wav');
    await send('reference audio');
    await page.waitForFunction(()=>window.store.getState().params.minimax_h3_references.some(ref=>ref.type==='audio'));
    await page.waitForFunction(()=>window.targets().find(target=>target.label==='reference audio')?.disabledReason);
    assert.equal(uploads.filter(item=>item.endpoint==='/api/v1/upload-audio').length,3,'Omni add/attach routes use its existing upload handlers');

    await mount('mixer',{params:{audio_mixer_tracks:[]}});
    await hasTarget('mixer base track');
    await send('mixer base track');
    await page.getByText('mixer base track.wav',{exact:true}).waitFor();
    await page.getByRole('button',{name:/Add/}).click();
    await hasTarget('mixer overlay track 1');
    await send('mixer overlay track 1');
    await page.getByText('mixer overlay track 1.wav',{exact:true}).waitFor();
    assert.equal(uploads.at(-1).endpoint,'/api/v1/upload','mixer uses the existing generic upload handler');

    await mount('yue2',{musicInstrumental:false,params:{model_mode:0,audio_guide:undefined,audio_prompt_type:''}});
    assert.ok(!(await targetLabels()).includes('YuE2 source song'),'collapsed score/source section has no hidden target');
    await page.getByText('Score and source song',{exact:true}).click();
    await hasTarget('YuE2 source song');
    await send('YuE2 source song');
    await page.waitForFunction(()=>window.store.getState().params.audio_guide==='/uploads/YuE2 source song.wav');
    assert.equal(await page.evaluate(()=>window.store.getState().params.audio_prompt_type),'A');
    assert.equal(errors.length,0,errors.join('\n'));
    console.log('Audio destinations passed: LTX voice ref, TTS voice slots, Omni audio reference and video soundtrack, mixer base/overlay, collapsed YuE2 target gating, and existing upload handlers.');
  } finally {await browser.close();}
})().catch(error=>{console.error(error);process.exitCode=1;});
